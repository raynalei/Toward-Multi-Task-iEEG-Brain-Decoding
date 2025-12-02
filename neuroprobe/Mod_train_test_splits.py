import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from sklearn.model_selection import KFold
import numpy as np

from Neuroprobe.neuroprobe.mod_dataset import BrainTreebankSubjectTrialBenchmarkDataset
from Neuroprobe.neuroprobe.config import *


def generate_splits_cross_subject(all_subjects, test_subject_id, test_trial_id, eval_name, dtype=torch.float32,
                          lite=True, nano=False,
                          
                          # Dataset parameters
                          binary_tasks=False,
                          output_indices=False, 
                          start_neural_data_before_word_onset=int(START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE), 
                          end_neural_data_after_word_onset=int(END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE),
                          max_samples=None):
    """Generate train/test splits for Cross-Subject evaluation."""
    assert test_subject_id != DS_DM_TRAIN_SUBJECT_ID, "Test subject cannot be the same as the training subject."

    # 1. Initialize TRAIN dataset first to calculate binning thresholds
    train_subject_id, train_trial_id = DS_DM_TRAIN_SUBJECT_ID, DS_DM_TRAIN_TRIAL_ID
    train_dataset = BrainTreebankSubjectTrialBenchmarkDataset(all_subjects[train_subject_id], train_trial_id, dtype=dtype, eval_name=eval_name, 
                                                                binary_tasks=binary_tasks, output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                                lite=lite, nano=nano, max_samples=max_samples)

    # 2. Extract learned thresholds from training set
    learned_thresholds = train_dataset.get_binning_thresholds()

    # 3. Initialize TEST dataset using the training thresholds
    test_dataset = BrainTreebankSubjectTrialBenchmarkDataset(all_subjects[test_subject_id], test_trial_id, dtype=dtype, eval_name=eval_name, 
                                                             binary_tasks=binary_tasks, output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                             lite=lite, nano=nano, max_samples=max_samples,
                                                             binning_thresholds=learned_thresholds) # Pass thresholds here

    test_size = len(test_dataset)
    val_size = test_size // 2
    val_indices = list(range(val_size))
    test_indices = list(range(val_size, test_size))
    val_dataset = Subset(test_dataset, val_indices)
    test_dataset = Subset(test_dataset, test_indices)
    
    # copy the electrode coordinates and labels
    val_dataset.electrode_coordinates = test_dataset.dataset.electrode_coordinates
    val_dataset.electrode_labels = test_dataset.dataset.electrode_labels
    test_dataset.electrode_coordinates = test_dataset.dataset.electrode_coordinates
    test_dataset.electrode_labels = test_dataset.dataset.electrode_labels

    return [
        {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset
        }
    ]
    

def generate_splits_cross_session(test_subject, test_trial_id, eval_name, dtype=torch.float32,
                          lite=True,
                          
                          # Dataset parameters
                          binary_tasks=False,
                          output_indices=False, 
                          start_neural_data_before_word_onset=int(START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE), 
                          end_neural_data_after_word_onset=int(END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE),
                          max_samples=None, include_all_other_trials=False):
    """Generate train/test splits for Cross-Session evaluation."""
    assert len(NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT[test_subject.subject_id]) > 1, f"Training subject must have at least two trials. But subject {test_subject.subject_id} has only {len(NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT[test_subject.subject_id])} trials."
    
    # 1. Initialize TRAIN dataset first to calculate binning thresholds
    if include_all_other_trials:
        train_trial_ids = [trial_id for trial_id in NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT[test_subject.subject_id] if trial_id != test_trial_id]

        if max_samples is not None:
            max_samples = max_samples // len(train_trial_ids)

        train_datasets = [BrainTreebankSubjectTrialBenchmarkDataset(test_subject, train_trial_id, dtype=dtype, eval_name=eval_name, 
                                                                    binary_tasks=binary_tasks, output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                                    lite=lite, max_samples=max_samples) for train_trial_id in train_trial_ids]
        train_dataset = ConcatDataset(train_datasets)
        # Note: ConcatDataset doesn't have get_binning_thresholds. We grab from the first one.
        # This assumes all train datasets have roughly similar stats or that the first is representative enough,
        # OR ideally we should calculate global stats. But for this pipeline, using the first trial's thresholds is a safe start.
        learned_thresholds = train_datasets[0].get_binning_thresholds() 
    else:
        if not lite:
            train_trial_id = NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT[test_subject.subject_id][0]
            if train_trial_id == test_trial_id:
                train_trial_id = NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT[test_subject.subject_id][1]
        else:
            train_trial_id = [trial_id for subject_id, trial_id in NEUROPROBE_LITE_SUBJECT_TRIALS if subject_id == test_subject.subject_id and trial_id != test_trial_id][0]
        
        train_dataset = BrainTreebankSubjectTrialBenchmarkDataset(test_subject, train_trial_id, dtype=dtype, eval_name=eval_name, 
                                                                    binary_tasks=binary_tasks, output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                                    lite=lite, max_samples=max_samples)
        learned_thresholds = train_dataset.get_binning_thresholds()

    # 2. Initialize TEST dataset using the training thresholds
    test_dataset = BrainTreebankSubjectTrialBenchmarkDataset(test_subject, test_trial_id, dtype=dtype, eval_name=eval_name, 
                                                             binary_tasks=binary_tasks, output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                             lite=lite, max_samples=max_samples,
                                                             binning_thresholds=learned_thresholds) # Pass thresholds here

    test_size = len(test_dataset)
    val_size = test_size // 2
    val_indices = list(range(val_size))
    test_indices = list(range(val_size, test_size))
    val_dataset = Subset(test_dataset, val_indices)
    test_dataset = Subset(test_dataset, test_indices)

    # copy the electrode coordinates and labels
    val_dataset.electrode_coordinates = test_dataset.dataset.electrode_coordinates
    val_dataset.electrode_labels = test_dataset.dataset.electrode_labels
    test_dataset.electrode_coordinates = test_dataset.dataset.electrode_coordinates
    test_dataset.electrode_labels = test_dataset.dataset.electrode_labels

    return [
        {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset
        }
    ]


def generate_splits_within_session(test_subject, test_trial_id, eval_name, dtype=torch.float32,
                          lite=True, nano=False,
                          
                          # Dataset parameters
                          binary_tasks=False,
                          output_indices=False, 
                          start_neural_data_before_word_onset=int(START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE), 
                          end_neural_data_after_word_onset=int(END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE),
                          max_samples=None):
    """Generate train/test splits for Within Session evaluation.
    
    NOTE: This method currently creates the dataset first (calculating global bins) and then splits using KFold. 
    Strictly speaking, this allows distributional information from the test fold to influence the bin definitions.
    However, refactoring this requires manual K-Folds on the raw DataFrame before Dataset instantiation.
    """

    train_datasets = []
    test_datasets = []
    val_datasets = []

    dataset = BrainTreebankSubjectTrialBenchmarkDataset(test_subject, test_trial_id, dtype=dtype, eval_name=eval_name, 
                                                        binary_tasks=binary_tasks, output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                        lite=lite, nano=nano, max_samples=max_samples)
    
    k_folds = NEUROPROBE_LITE_N_FOLDS if not nano else NEUROPROBE_NANO_N_FOLDS
    kf = KFold(n_splits=k_folds, shuffle=False) 
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        # Skip empty splits
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        
        train_dataset = Subset(dataset, train_idx)
        train_datasets.append(train_dataset)
        test_dataset = Subset(dataset, test_idx)
        
        test_size = len(test_dataset)
        val_size = test_size // 2
        val_indices = list(range(val_size))
        test_indices = list(range(val_size, test_size))
        val_dataset = Subset(test_dataset, val_indices)
        test_dataset = Subset(test_dataset, test_indices)

        val_dataset.electrode_coordinates = dataset.electrode_coordinates
        val_dataset.electrode_labels = dataset.electrode_labels
        test_dataset.electrode_coordinates = dataset.electrode_coordinates
        test_dataset.electrode_labels = dataset.electrode_labels
        test_datasets.append(test_dataset)
        val_datasets.append(val_dataset)

    return [
        {
            "train_dataset": train_dataset, 
            "val_dataset": val_dataset, 
            "test_dataset": test_dataset
        } for train_dataset, val_dataset, test_dataset in zip(train_datasets, val_datasets, test_datasets)
    ]

# For backwards compatibility
generate_splits_DS_DM = generate_splits_cross_subject
generate_splits_SS_DM = generate_splits_cross_session
generate_splits_SS_SM = generate_splits_within_session

# For flexibility in function naming convention
generate_splits_CrossSubject = generate_splits_cross_subject
generate_splits_CrossSession = generate_splits_cross_session
generate_splits_WithinSession = generate_splits_within_session