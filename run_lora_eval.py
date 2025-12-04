import argparse
import os
import numpy as np
import torch
import json
import gc
import time
from sklearn.preprocessing import StandardScaler

# Import provided modules
from neuroprobe.braintreebank_subject import BrainTreebankSubject
import neuroprobe.Mod_train_test_splits as neuroprobe_train_test_splits
import neuroprobe.config as neuroprobe_config
from examples.eval_utils_Mod import preprocess_data, log, subset_electrodes, combine_regions, get_region_labels
from lora_utils import MambaLoraWrapper

# Configuration constants
HF_PATH = "/ocean/projects/cis250217p/ylei5/Neuroprobe/ckpts/mamba-1.4b-hf"
# Path to the checkpoint trained in train_mask_evalutils.py
RECON_CKPT_PATH = "/ocean/projects/cis250217p/ylei5/Neuroprobe/ckpts/mask_evalutils/model.safetensors" 

# Patching config must match pre-training
T_PATCH = 16
F_PATCH = 16
PATCH_DIM = T_PATCH * F_PATCH 

def get_data_labels(dataset, subject_labels, preprocess_type, preprocess_params, start_idx, end_idx):
    """
    Helper to extract and preprocess data from a dataset object.
    """
    # X shape: (N, Electrodes, Freqs, TimeBins)
    # Note: Accessing dataset items might be slow if not cached; standard implementation accesses disk.
    X_list = []
    y_list = []
    
    # Iterate safely
    for i in range(len(dataset)):
        try:
            item = dataset[i]
            # Slicing time dimension: [start_idx : end_idx]
            # Ensure tensor is float
            neural_data = item[0][:, start_idx:end_idx].unsqueeze(0).float()
            
            processed = preprocess_data(
                neural_data, 
                subject_labels, 
                preprocess_type, 
                preprocess_params
            )
            
            X_list.append(processed)
            y_list.append(item[1])
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue
            
    if len(X_list) == 0:
        return np.array([]), np.array([])

    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)
    return X, y

def parse_id_list(id_str):
    """Parses a comma-separated string of IDs into a list of integers."""
    if not id_str:
        return []
    return [int(x.strip()) for x in id_str.split(',')]

def main():
    parser = argparse.ArgumentParser()
    # Standard Neuroprobe args
    parser.add_argument('--eval_name', type=str, default='onset,speech,pitch', help='List of tasks')
    
    # Changed to support lists
    parser.add_argument('--subject_ids', type=str, required=True, help='Comma-separated list of subject IDs (e.g., "1,1,2")')
    parser.add_argument('--trial_ids', type=str, required=True, help='Comma-separated list of trial IDs (e.g., "1,2,1")')
    
    parser.add_argument('--save_dir', type=str, default='eval_results_lora')
    
    # Split configuration
    splits_options = ['WithinSession', 'CrossSession', 'CrossSubject']
    parser.add_argument('--split_type', type=str, choices=splits_options, default='WithinSession')
    
    # Preprocessing args
    parser.add_argument('--nperseg', type=int, default=512)
    parser.add_argument('--poverlap', type=float, default=0.75)
    parser.add_argument('--lite', action='store_true', default=True, help="Use lite electrodes")
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    
    # Parse IDs
    subject_ids = parse_id_list(args.subject_ids)
    trial_ids = parse_id_list(args.trial_ids)
    
    if len(subject_ids) != len(trial_ids):
        raise ValueError(f"Number of subject_ids ({len(subject_ids)}) must match number of trial_ids ({len(trial_ids)})")

    # Setup Preprocessing
    preprocess_type = 'stft_abs' 
    preprocess_parameters = {
        "type": preprocess_type,
        "stft": {
            "nperseg": args.nperseg,
            "poverlap": args.poverlap,
            "window": "hann",
            "max_frequency": 150,
            "min_frequency": 0
        }
    }
    
    # For CrossSubject, we need the Training Subject (Fixed)
    train_subject = None
    if args.split_type == 'CrossSubject':
        print("Loading Training Subject for CrossSubject split...")
        train_subject_id = neuroprobe_config.DS_DM_TRAIN_SUBJECT_ID
        train_subject = BrainTreebankSubject(train_subject_id, allow_corrupted=False, cache=True, dtype=torch.float32)
        train_electrodes = neuroprobe_config.NEUROPROBE_LITE_ELECTRODES[train_subject.subject_identifier] if args.lite else train_subject.electrode_labels
        train_subject.set_electrode_subset(train_electrodes)

    # Define Time Bins
    bins_start = -0.5
    bins_end = 1.0 
    sr = neuroprobe_config.SAMPLING_RATE
    data_idx_from = int((bins_start + 0.5) * sr) 
    data_idx_to = int((bins_end + 0.5) * sr)

    eval_names = args.eval_name.split(',')

    # --- Main Loop over (Subject, Trial) Pairs ---
    for s_idx, (curr_sub_id, curr_trial_id) in enumerate(zip(subject_ids, trial_ids)):
        print(f"\n[{s_idx+1}/{len(subject_ids)}] Processing Subject {curr_sub_id}, Trial {curr_trial_id}")
        
        # Load Current Test Subject
        try:
            subject = BrainTreebankSubject(curr_sub_id, cache=True, dtype=torch.float32)
            subset_electrodes(subject, lite=args.lite)
            subject.load_neural_data(curr_trial_id)
        except Exception as e:
            print(f"Failed to load Subject {curr_sub_id} Trial {curr_trial_id}: {e}")
            continue

        for eval_task in eval_names:
            print(f"--- Task: {eval_task} | Split: {args.split_type} ---")
            
            # 1. Generate Splits
            folds = []
            split_train_subject = None
            
            try:
                if args.split_type == 'WithinSession':
                    #
                    folds = neuroprobe_train_test_splits.generate_splits_within_session(
                        subject, curr_trial_id, eval_task, 
                        dtype=torch.float32, lite=args.lite, binary_tasks=True
                    )
                    split_train_subject = subject
                    
                elif args.split_type == 'CrossSession':
                    #
                    folds = neuroprobe_train_test_splits.generate_splits_cross_session(
                        subject, curr_trial_id, eval_task, 
                        dtype=torch.float32, lite=args.lite, binary_tasks=True
                    )
                    split_train_subject = subject
                    
                elif args.split_type == 'CrossSubject':
                    #
                    all_subjects = {
                        curr_sub_id: subject,
                        neuroprobe_config.DS_DM_TRAIN_SUBJECT_ID: train_subject,
                    }
                    folds = neuroprobe_train_test_splits.generate_splits_cross_subject(
                        all_subjects, curr_sub_id, curr_trial_id, eval_task,
                        dtype=torch.float32, lite=args.lite, binary_tasks=True
                    )
                    split_train_subject = train_subject
            except Exception as e:
                print(f"Skipping task {eval_task} due to split generation error: {e}")
                continue

            task_results = []

            for fold_idx, fold in enumerate(folds):
                print(f"Fold {fold_idx + 1}/{len(folds)}")
                
                # 2. Extract Data
                train_ds_obj = fold["train_dataset"]
                test_ds_obj = fold["test_dataset"]
                val_ds_obj = fold["val_dataset"]

                # For CrossSubject, val usually comes from train distribution (Training Subject)
                # For Within/CrossSession, val comes from the same subject
                val_subject_ref = split_train_subject if args.split_type == 'CrossSubject' else subject

                print("  Preprocessing...")
                # Load Train
                X_train, y_train = get_data_labels(train_ds_obj, split_train_subject.electrode_labels, preprocess_type, preprocess_parameters, data_idx_from, data_idx_to)
                if len(X_train) == 0:
                    print("  No training data found, skipping fold.")
                    continue
                
                # Load Test
                X_test, y_test = get_data_labels(test_ds_obj, subject.electrode_labels, preprocess_type, preprocess_parameters, data_idx_from, data_idx_to)
                
                # Load Val
                X_val, y_val = get_data_labels(val_ds_obj, val_subject_ref.electrode_labels, preprocess_type, preprocess_parameters, data_idx_from, data_idx_to)

                # 3. Handle CrossSubject Region Averaging
                if args.split_type == 'CrossSubject':
                    print("  Combining Regions (CrossSubject)...")
                    regions_train = get_region_labels(split_train_subject)
                    regions_test = get_region_labels(subject)
                    
                    # Align Train & Test
                    X_train, X_test, common_regions = combine_regions(X_train, X_test, regions_train, regions_test)
                    
                    # Align Val (which comes from Train Subject) to same common regions
                    # We pass X_val as "train" and X_test as "test" just to use the logic to project X_val
                    X_val_aligned, _, _ = combine_regions(X_val, X_test, regions_train, regions_test)
                    X_val = X_val_aligned

                # 4. Standardization
                print("  Standardizing...")
                scaler = StandardScaler()
                N_tr, C, F, T = X_train.shape
                
                X_train = scaler.fit_transform(X_train.reshape(N_tr, -1)).reshape(N_tr, C, F, T)
                if len(X_val) > 0:
                    X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape[0], C, F, T)
                if len(X_test) > 0:
                    X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape[0], C, F, T)
                
                # 5. Train & Eval (Mamba LoRA)
                clf = MambaLoraWrapper(
                    hf_path=HF_PATH,
                    recon_ckpt_path=RECON_CKPT_PATH,
                    patch_dim=PATCH_DIM,
                    t_patch=T_PATCH,
                    f_patch=F_PATCH,
                    batch_size=16,
                    epochs=5,
                    lr=1e-3,
                    seed=args.seed
                )

                print(f"  Training LoRA...")
                clf.fit(X_train, y_train, X_val, y_val)
                
                test_acc = clf.score(X_test, y_test)
                test_probs = clf.predict_proba(X_test)

                # Calculate AUC
                test_auc = 0.5
                try:
                    from sklearn.metrics import roc_auc_score
                    if len(np.unique(y_test)) == 2:
                        test_auc = roc_auc_score(y_test, test_probs[:, 1])
                    else:
                        y_test_oh = np.eye(len(clf.classes_))[y_test]
                        test_auc = roc_auc_score(y_test_oh, test_probs, multi_class='ovr')
                except Exception as e:
                    print(f"  AUC calc failed: {e}")

                print(f"  > Fold {fold_idx+1} Result: Acc={test_acc:.4f}, AUC={test_auc:.4f}")
                
                task_results.append({
                    "fold": fold_idx,
                    "accuracy": float(test_acc),
                    "auc": float(test_auc)
                })

                # Cleanup
                del clf, X_train, X_val, X_test, scaler
                gc.collect()
                torch.cuda.empty_cache()

            # Save Results per Subject-Trial-Task
            os.makedirs(args.save_dir, exist_ok=True)
            filename = f"{curr_sub_id}_{curr_trial_id}_{eval_task}_{args.split_type}_lora.json"
            save_path = os.path.join(args.save_dir, filename)
            with open(save_path, 'w') as f:
                json.dump(task_results, f, indent=4)
            print(f"Saved results to {save_path}")

        # Cleanup Subject Data from Memory
        del subject
        gc.collect()

if __name__ == "__main__":
    main()
