import torch
import torch.nn as nn
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType
from transformers import MambaConfig, MambaModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import copy
from safetensors.torch import load_file
# Import patchify logic to ensure input consistency with pre-training
from utils_mask_evalutils import patchify_grid #

class MambaDownstreamLora(nn.Module):
    def __init__(self, hf_path, recon_ckpt_path, num_classes, in_dim, lora_rank=8, lora_alpha=16):
        """
        Args:
            hf_path: Path to the base HuggingFace Mamba model (e.g. state-spaces/mamba-1.4b).
            recon_ckpt_path: Path to the saved 'reconstruction' checkpoint.
            num_classes: Number of output classes for the downstream task.
            in_dim: Input dimension of the patches (must match pre-training).
        """
        super().__init__()
        
        # 1. Load Base Mamba Backbone
        # We use the transformer/hf compatible version to easily integrate with PEFT
        self.backbone = MambaModel.from_pretrained(hf_path)
        
        # 2. Load Reconstruction Weights (Unsupervised Learning Phase)
        # Assuming the ckpt was saved via trainer.save_model or torch.save
        if recon_ckpt_path:
            print(f"Loading reconstruction weights from {recon_ckpt_path}")
            
            if recon_ckpt_path.endswith(".safetensors"):
                state_dict = load_file(recon_ckpt_path)
            else:
                state_dict = torch.load(recon_ckpt_path, map_location="cpu")

            backbone_keys = {k.replace("backbone.", ""): v for k, v in state_dict.items() if "backbone" in k}
            if not backbone_keys:
                backbone_keys = state_dict
            
            missing, unexpected = self.backbone.load_state_dict(backbone_keys, strict=False)
            print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        # 3. Input Projection (matches MambaMaskRecon)
        self.d_model = self.backbone.config.hidden_size
        self.input_proj = nn.Linear(in_dim, self.d_model)

        # 4. Classification Head (New)
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, num_classes)
        )

        # 5. Apply LoRA
        # Target modules usually include the linear projections inside Mamba blocks
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["x_proj", "embeddings", "in_proj", "out_proj"], 
            task_type=None, # Generic task
            inference_mode=False,
            dropout=0.1
        )
        self.backbone = get_peft_model(self.backbone, peft_config)
        self.backbone.print_trainable_parameters()

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Patch_Dim)
        
        # Project patches to d_model
        x = self.input_proj(x) # (B, L, D)
        
        # Mamba Forward
        outputs = self.backbone(inputs_embeds=x)
        hidden_states = outputs.last_hidden_state # (B, L, D)

        # Pooling: Mean pool over sequence length (time/freq patches)
        # You could also take the last token, but mean is stable for classification
        pooled = hidden_states.mean(dim=1) # (B, D)
        
        logits = self.classifier(pooled) # (B, Num_Classes)
        return logits

class MambaLoraWrapper:
    """
    Scikit-learn compatible wrapper for the Mamba LoRA model.
    Follows the API used in eval_population_mod.py
    """
    def __init__(self, 
                 hf_path, 
                 recon_ckpt_path, 
                 patch_dim,
                 t_patch=16, f_patch=16, #
                 batch_size=32, 
                 lr=1e-4, 
                 epochs=10, 
                 device='cuda',
                 seed=42):
        self.hf_path = hf_path
        self.recon_ckpt_path = recon_ckpt_path
        self.patch_dim = patch_dim
        self.t_patch = t_patch
        self.f_patch = f_patch
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.seed = seed
        self.model = None
        self.classes_ = None

    def _format_input(self, X):
        """
        Converts (N, Electrodes, Freq, Time) or (N, Freq, Time) to (N, Seq_Len, Patch_Dim).
        Uses patchify_grid from utils_mask_evalutils.py.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Handle dimensions. 
        # Ideally X is (Batch, Electrodes, Freq, Time).
        # We need to flatten Electrodes/Freq/Time into a sequence of patches.
        
        formatted_batch = []
        for i in range(X_tensor.shape[0]):
            sample = X_tensor[i] # (C, F, T)
            
            # If multi-channel, we can either:
            # 1. Treat channels as part of the patch dimension (Patch_Dim = T_p * F_p * C)
            # 2. Stack channels into the sequence (Seq_Len = Patches * C)
            # Given MambaMaskRecon usually takes 1 channel or mixes them, 
            # we will average channels here for simplicity, OR flatten.
            # Let's average channels to match single-electrode pretraining logic usually found in Braintreebank
            if sample.ndim == 3:
                sample = sample.mean(dim=0) # (F, T)
            
            # Ensure (T, F) for patchify_grid
            if sample.shape[0] != self.f_patch * self.t_patch: 
                 # Assuming input is (Freq, Time), transpose to (Time, Freq) for patchify
                 sample = sample.transpose(0, 1) 
            
            # patchify_grid expects (T, F).
            # Returns (num_patches, t_patch * f_patch)
            patches = patchify_grid(sample, t_patch=self.t_patch, f_patch=self.f_patch)
            formatted_batch.append(patches)
            
        return torch.stack(formatted_batch) # (N, L, Patch_Dim)

    def fit(self, X_train, y_train, X_val, y_val):
        torch.manual_seed(self.seed)
        self.classes_ = np.unique(y_train)
        num_classes = len(self.classes_)
        
        # Format Data
        X_train_seq = self._format_input(X_train)
        X_val_seq = self._format_input(X_val)
        
        # Verify Patch Dim matches
        current_patch_dim = X_train_seq.shape[-1]
        assert current_patch_dim == self.patch_dim, \
            f"Data patch dim {current_patch_dim} != Model input dim {self.patch_dim}"

        y_train_t = torch.tensor(y_train, dtype=torch.long)
        y_val_t = torch.tensor(y_val, dtype=torch.long)

        train_ds = TensorDataset(X_train_seq, y_train_t)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        # Initialize Model
        self.model = MambaDownstreamLora(
            self.hf_path, 
            self.recon_ckpt_path, 
            num_classes, 
            self.patch_dim
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        best_val_score = -np.inf
        best_state = None

        print(f"Starting LoRA Finetuning for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for bx, by in train_dl:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation logic matching eval_utils_Mod.py
            val_acc = self.score(X_val, y_val) # Note: score calls predict -> _format_input internally
            print(f"Epoch {epoch+1}: Loss {train_loss/len(train_dl):.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_val_score:
                best_val_score = val_acc
                best_state = copy.deepcopy(self.model.state_dict())

        if best_state:
            self.model.load_state_dict(best_state)
        
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_seq = self._format_input(X)
        ds = TensorDataset(X_seq)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        
        probs = []
        with torch.no_grad():
            for bx in dl:
                bx = bx[0].to(self.device)
                logits = self.model(bx)
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        
        return np.concatenate(probs, axis=0)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)