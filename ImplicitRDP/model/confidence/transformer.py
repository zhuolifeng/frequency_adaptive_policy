import copy
import math
from typing import Optional
import torch
import torch.nn as nn

from ImplicitRDP.model.common.module_attr_mixin import ModuleAttrMixin


class _SinePositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (B, T, C) tensor, got shape {x.shape}")
        seq_len = x.shape[1]
        device = x.device
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(seq_len, self.d_model, device=device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe.unsqueeze(0)


class TransformerConfidenceHead(ModuleAttrMixin):
    def __init__(self,
                 n_emb: int = 768,
                 hidden_dim: Optional[int] = None,
                 obs_encoder_n_layers: int = 2,
                 obs_encoder_hidden_dim: Optional[int] = None,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 use_learnable_pos_emb: bool = False,
                 max_obs_tokens: int = 100,
                 max_cond_tokens: int = 100,
                 pooling: str = 'mean'):  # 'last', 'mean', 'max'
        super().__init__()

        self.n_emb = n_emb
        self.hidden_dim = hidden_dim or n_emb
        self.use_learnable_pos_emb = use_learnable_pos_emb
        self.pooling = pooling

        ff_dim = obs_encoder_hidden_dim or (4 * n_emb)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )

        # Position encoding: learnable or sinusoidal
        if use_learnable_pos_emb:
            self.conf_pos_emb = nn.Parameter(torch.zeros(1, max_obs_tokens, n_emb))
            self.conf_pos_enc = None
        else:
            self.conf_pos_enc = _SinePositionalEncoding(n_emb)
            self.conf_pos_emb = None

        # encoder and projection for confidence head
        self.conf_encoder = nn.TransformerEncoder(
            copy.deepcopy(encoder_layer),
            num_layers=obs_encoder_n_layers
        )
        self.conf_proj = nn.Sequential(
            nn.LayerNorm(n_emb),
            nn.Linear(n_emb, n_emb)
        )

        # Head takes only obs features
        self.head = nn.Sequential(
            nn.Linear(n_emb, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, TransformerConfidenceHead):
            # Initialize learnable position embeddings (same as diffusion models)
            if getattr(module, 'conf_pos_emb', None) is not None:
                nn.init.normal_(module.conf_pos_emb, mean=0.0, std=0.02)

    def _encode_conf_obs(self,
                    encoder: nn.TransformerEncoder,
                    pos_enc: Optional[_SinePositionalEncoding],
                    pos_emb: Optional[nn.Parameter],
                    obs_tokens: torch.Tensor) -> torch.Tensor:
        # Apply position encoding for confidence head
        if pos_enc is not None:
            obs_tokens = pos_enc(obs_tokens)
        elif pos_emb is not None:
            seq_len = obs_tokens.shape[1]
            pos_emb_slice = pos_emb[:, :seq_len, :]
            obs_tokens = obs_tokens + pos_emb_slice

        encoded = encoder(obs_tokens)

        # Pool sequence features
        if self.pooling == 'last':
            return encoded[:, -1, :]  # (B, n_emb)
        elif self.pooling == 'mean':
            return encoded.mean(dim=1)  # (B, n_emb)
        elif self.pooling == 'max':
            return encoded.max(dim=1)[0]  # (B, n_emb)
        else:
            return encoded.mean(dim=1)  # default to mean

    def forward(self,
                obs_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_tokens: (B, N, n_emb)
        """
        obs_feat = self.conf_proj(
            self._encode_conf_obs(
                self.conf_encoder,
                self.conf_pos_enc,
                self.conf_pos_emb,
                obs_tokens  # (B, N, n_emb)
            )
        )

        # Head takes only obs features
        log_sigma_raw = self.head(obs_feat).squeeze(-1)

        return log_sigma_raw

    def get_optim_groups(self, weight_decay: float = 1e-3):
        groups = [
            {"params": self.head.parameters(), "weight_decay": weight_decay},
            {"params": self.conf_encoder.parameters(), "weight_decay": weight_decay},
            {"params": self.conf_proj.parameters(), "weight_decay": weight_decay},
        ]

        if getattr(self, 'conf_pos_emb', None) is not None:
            groups.append({"params": [self.conf_pos_emb], "weight_decay": 0.0})

        return groups

