import torch, torch.nn as nn
from transformers import AutoModel

class EEGAdapter(nn.Module):
    def __init__(self, in_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
        )
    def forward(self, x): return self.net(x)

class MambaMaskRecon(nn.Module):
    def __init__(self, hf_dir, in_dim, dtype=torch.float16, device="cuda"):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.backbone = AutoModel.from_pretrained(
            hf_dir,
            torch_dtype=dtype,
            device_map=None,
            low_cpu_mem_usage=True,
        ).to(device)
        self.backbone.requires_grad_(False)

        d_model = self.backbone.get_input_embeddings().embedding_dim

        self.adapter = EEGAdapter(in_dim, d_model).to(device=device, dtype=dtype)
        self.recon_head = nn.Linear(d_model, in_dim).to(device=device, dtype=dtype)

    def forward(self, cont_feats, labels):

        print("cont_feats shape:", cont_feats.shape)

        cont_feats = cont_feats.to(device=self.device, dtype=self.dtype)
        labels     = labels.to(device=self.device)

        # adapter
        h = self.adapter(cont_feats)       # (B, T, d_model), dtype = self.dtype

        # backbone (frozen)
        with torch.no_grad():
            out = self.backbone(
                inputs_embeds=h,
                output_hidden_states=False,
                use_cache=False,
            )
            hid = out.last_hidden_state.to(self.dtype)
            # reconstruction
        recon = self.recon_head(hid)       

        # mask loss
        if labels.dim() == 2:
            labels_expanded = labels.unsqueeze(-1).expand_as(recon)
        else:
            labels_expanded = labels

        loss = torch.nn.functional.mse_loss(
            recon[labels_expanded],
            cont_feats[labels_expanded],
        )

        return {"loss": loss, "recon": recon}

# Total params: 1,377,425,664
# Trainable params: 5,247,232
# Frozen params: 1,372,178,432