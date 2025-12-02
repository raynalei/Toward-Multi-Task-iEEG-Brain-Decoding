import os
os.environ.setdefault(
    "ROOT_DIR_BRAINTREEBANK",
    "/ocean/projects/cis250217p/shared/data/braintreebank"  
)
import torch
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from torchinfo import summary

from utils_mask_evalutils import EEGMaskDataset, collate_mask
from models_mask_recon import MambaMaskRecon

# nohup python train_mask_evalutils.py > log.txt 2>&1 &

HF_DIR = "/ocean/projects/cis250217p/ylei5/Neuroprobe/ckpts/mamba-1.4b-hf"
SAVE_DIR = "/ocean/projects/cis250217p/ylei5/Neuroprobe/ckpts/mask_evalutils"

# Training config
BS      = 1
EPOCHS  = 5
LR      = 1e-3
TP, FP  = 16, 16         # patch size (time, freq)
MASK_P  = 0.3

# SUBJECT_ID = [1,1,2,2,3,3,4,4,7,7,10,10]
# TRIAL_ID   = [1,2,0,4,0,1,0,1,0,1,0,1]

SUBJECT_ID = [1,1]
TRIAL_ID   = [1,2]
def main():

    preprocess = "laplacian-stft_abs"
    preprocess_parameters = {
        "stft": {
            "nperseg": 512,
            "poverlap": 0.75,
            "window": "hann",
            "max_frequency": 150,
            "min_frequency": 0,
        }
    }
    print ("Start creating dataset over multiple subjects and trials")
    # ================================
    # Create dataset over multiple subjects and trials
    # ================================
    ds = EEGMaskDataset(
        subject_ids=SUBJECT_ID,
        trial_ids=TRIAL_ID,
        preprocess=preprocess,
        preprocess_parameters=preprocess_parameters,
        mask_prob=MASK_P,
        t_patch=TP,
        f_patch=FP,
        window_size_seconds=1.0,
        stride_seconds=None,   # None -> stride == window_size
        lite=True,
        electrodes="lite",
    )

    # Determine input patch dimension for the Mamba projection layer
    sample = ds[0]["cont_feats"]          # shape: (num_patches, patch_dim)
    in_dim = sample.size(1)

    # ================================
    # Load Mamba + Mask Reconstruction model
    # ================================
    print("Load Mamba + Mask Reconstruction model")#1h
    model = MambaMaskRecon(
        hf_dir=HF_DIR,
        in_dim=in_dim
    )
    
    def count_params(model):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total:,}")
        print(f"Trainable params: {trainable:,}")
        print(f"Frozen params: {total-trainable:,}")

    print("total parameters are:",count_params(model))


    print("hf_device_map =", getattr(model, "hf_device_map", None))

    first_param = next(model.parameters())
    print("first_param.device =", first_param.device)

 

 

    # ================================
    # Training Arguments
    # ================================
    args = TrainingArguments(
        output_dir=SAVE_DIR,
        per_device_train_batch_size=BS,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=20,
        save_strategy="epoch",
        remove_unused_columns=False,   # IMPORTANT: custom batch dict keys
    )

    # ================================
    # Custom loss (uses cont_feats + labels)
    # ================================
    class MaskTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            out = model(inputs["cont_feats"], inputs["labels"])
            loss = out["loss"]
            return (loss, out) if return_outputs else loss


        
    trainer = MaskTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collate_mask,
    )
    print("Start training")

    trainer.train()
    trainer.save_model(SAVE_DIR)
    print("Saved:", SAVE_DIR)


if __name__ == "__main__":
    main()
