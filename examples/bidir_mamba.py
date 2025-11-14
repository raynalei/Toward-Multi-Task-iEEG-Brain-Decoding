# Bidirectional (layer-wise alternating direction) Mamba backbone + classification head
# (consistent with existing MambaClassifier input/output)
# Dependencies: create_block from mixer_seq_simple.py

import math
import torch
import torch.nn as nn

try:
    # Use Triton fused Add+Norm fast path if available
    from mamba_ssm.ops.triton.layer_norm import layer_norm_fn, RMSNorm
    _FUSED_LN_AVAILABLE = True
except Exception:
    layer_norm_fn, RMSNorm = None, None
    _FUSED_LN_AVAILABLE = False

from mixer_seq_simple import create_block  

class BidirectionalMambaEncoder(nn.Module):
    """
    Accepts (B, T, d_model) → outputs (B, T, d_model)
    Alternating directions between layers: even layers forward, odd layers backward (configurable).
    Maintains consistency with mixer_seq_simple.MixerModel residual protocol.
    """
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_intermediate: int = 256,
        ssm_cfg: dict | None = None,
        rms_norm: bool = False,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        alternating_directions: bool = True,
        norm_epsilon: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_intermediate = d_intermediate
        self.ssm_cfg = ssm_cfg or {"layer": "Mamba2", "d_state": 16, "d_conv": 4, "expand": 2}
        self.rms_norm = rms_norm
        self.fused_add_norm = fused_add_norm and _FUSED_LN_AVAILABLE
        self.residual_in_fp32 = residual_in_fp32
        self.alternating_directions = alternating_directions

        # Build layers
        self.layers = nn.ModuleList([
            create_block(
                d_model=self.d_model,
                d_intermediate=self.d_intermediate,
                ssm_cfg=self.ssm_cfg,
                attn_layer_idx=[],          # All Mamba; pass indices externally if you want mixed attention
                attn_cfg={},
                norm_epsilon=norm_epsilon,
                rms_norm=self.rms_norm,
                residual_in_fp32=self.residual_in_fp32,
                fused_add_norm=self.fused_add_norm,
                layer_idx=i,
                device=None,
                dtype=None,
            ) for i in range(self.n_layers)
        ])

        # Final normalization
        if self.rms_norm and _FUSED_LN_AVAILABLE:
            self.norm_f = RMSNorm(self.d_model, eps=norm_epsilon)
            self._is_rms = True
        else:
            self.norm_f = nn.LayerNorm(self.d_model, eps=norm_epsilon)
            self._is_rms = False

    @torch.no_grad()
    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None):
        # Align with original MixerModel interface (for streaming inference if needed)
        return {i: layer.allocate_inference_cache(
            batch_size=batch_size, max_seqlen=max_seqlen, dtype=dtype
        ) for i, layer in enumerate(self.layers)}

    def _maybe_flip(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        # Flip sequence dimension T (dim=1)
        return torch.flip(x, dims=[1])

    def forward(self, x: torch.Tensor, inference_params=None, **mixer_kwargs) -> torch.Tensor:
        """
        x: (B, T, d_model)
        returns: (B, T, d_model)
        """
        residual = None

        for i, layer in enumerate(self.layers):
            use_backward = self.alternating_directions and (i % 2 == 1)

            if use_backward:
                # Flip main branch and residual before entering "backward layer"
                x = self._maybe_flip(x)
                residual = self._maybe_flip(residual)

            x, residual = layer(
                x, residual, inference_params=inference_params, **mixer_kwargs
            )

            if use_backward:
                # Flip back to forward direction after layer to align subsequent layers
                x = self._maybe_flip(x)
                residual = self._maybe_flip(residual)

        # Final Add+Norm (consistent with mixer_seq_simple.MixerModel)
        if not self.fused_add_norm:
            residual = (x + residual) if residual is not None else x
            x = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            x = layer_norm_fn(
                x, self.norm_f.weight, self.norm_f.bias, eps=self.norm_f.eps,
                residual=residual, prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=self._is_rms
            )
        return x


class Mamba(nn.Module):
    """
    Classification head: maintains complete input/output contract consistency with the module
    returned by existing MambaClassifier._create_model.
    - Input shape description: input_shape=(C,T) or (C,F,T)
    - Frontend: Linear projection to d_model, treating time dimension as sequence dimension
    - Backbone: BidirectionalMambaEncoder (alternating directions)
    - Pooling: mean over time dimension
    - Output: Linear to n_classes
    """
    def __init__(
        self,
        input_shape,
        n_classes: int,
        d_model: int = 64,
        d_intermediate: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
        ssm_cfg: dict | None = None,
        rms_norm: bool = False,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        alternating_directions: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        if len(input_shape) == 2:
            # (C, T)
            in_feat = input_shape[0]
            self.seq_len = input_shape[1]
            self._is_2d = True
        else:
            # (C, F, T)
            in_feat = input_shape[0] * input_shape[1]
            self.seq_len = input_shape[2]
            self._is_2d = False

        # Project (C) or (C*F) to d_model → (B, T, d_model)
        self.input_proj = nn.Linear(in_feat, d_model)

        self.encoder = BidirectionalMambaEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_intermediate=d_intermediate,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            alternating_directions=alternating_directions,
        )

        self.fc = nn.Linear(d_model, n_classes)

        self.reset_parameters()

    def reset_parameters(self):
        # Simple robust initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T) or (B, C, F, T)
        returns: (B, n_classes)
        """
        if x.ndim == 3:
            # (B, C, T) → (B, T, C) → Linear
            x = x.transpose(1, 2)                    # (B, T, C)
            x = self.input_proj(x)                   # (B, T, d_model)
        else:
            # (B, C, F, T) → (B, T, C*F) → Linear
            b, c, f, t = x.shape
            x = x.transpose(1, 3).reshape(b, t, c * f)
            x = self.input_proj(x)                   # (B, T, d_model)

        x = self.dropout(x)
        x = self.encoder(x)                          # (B, T, d_model)
        x = x.mean(dim=1)                            # (B, d_model)
        x = self.fc(x)                               # (B, n_classes)
        return x
