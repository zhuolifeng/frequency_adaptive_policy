"""
Multi-frequency Transformer for improved MeanFlow (iMF).

Single-stage training backbone that outputs dual predictions:
  u: average velocity field (for MeanFlow equation)
  v: instantaneous velocity field (auxiliary head for JVP)

Architecture:
  shared_blocks (n_layer - aux_head_depth layers)
    └─> u_head_blocks (aux_head_depth layers) → u output
    └─> v_head_blocks (aux_head_depth layers) → v output

Condition tokens: [time_emb | h_emb | obs_tokens]
  - time_emb: diffusion time t
  - h_emb: time difference h = t - r (iMF specific)
  - No freq_emb or distill_emb (simplified from CTM/CD backbones)
"""
from typing import Union, Optional
import logging

import torch
import torch.nn as nn

from ImplicitRDP.model.common.module_attr_mixin import ModuleAttrMixin
from ImplicitRDP.model.diffusion.positional_embedding import SinusoidalPosEmb
from ImplicitRDP.model.diffusion.transformer_utils import (
    CustomizedTransformerDecoderLayer,
)

logger = logging.getLogger(__name__)


class TransformerForIMFMultiFreq(ModuleAttrMixin):
    """Transformer backbone for multi-frequency improved MeanFlow (iMF).

    Key differences from TransformerForDiffusionMultiFreq:
    1. h_emb replaces distill_emb — embeds time difference h = t - r
    2. No freq_emb — removed (was already disabled in CTM yaml)
    3. Dual-head decoder — shared backbone branches into u-head and v-head
    4. Returns (u, v) instead of single velocity
    """

    def __init__(
        self,
        shape_meta: dict,
        n_obs_steps: int = 2,
        cond_dim: int = 768,
        n_layer: int = 7,
        n_head: int = 4,
        n_emb: int = 768,
        max_cond_tokens: int = 800,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.1,
        time_as_cond: bool = True,
        obs_as_cond: bool = False,
        n_cond_layers: int = 0,
        # Multi-freq specific
        horizon_10hz: int = 16,
        horizon_50hz: int = 80,
        freq_ratio: int = 5,
        action_offset_10hz: int = 8,
        temporal_offset_10hz: int = 4,
        # iMF dual-head specific
        aux_head_depth: int = 4,
    ) -> None:
        super().__init__()

        input_dim = shape_meta["action"]["shape"][0]
        output_dim = shape_meta["action"]["shape"][0]

        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond

        assert aux_head_depth < n_layer, (
            f"aux_head_depth ({aux_head_depth}) must be < n_layer ({n_layer})"
        )
        shared_depth = n_layer - aux_head_depth

        # ---- Input embedding ----
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.drop = nn.Dropout(p_drop_emb)

        # ---- Discrete shared PE table on the 50hz grid ----
        max_action_index = action_offset_10hz + (horizon_10hz - 1) * freq_ratio
        max_temporal_index = temporal_offset_10hz + (horizon_10hz - 1) * freq_ratio
        pos_emb_len = max(horizon_50hz, max_action_index + 1, max_temporal_index + 1)
        self.pos_emb = nn.Parameter(torch.zeros(1, pos_emb_len, n_emb))

        # ---- Timestep embeddings ----
        self.time_emb = SinusoidalPosEmb(n_emb)   # diffusion time t
        self.h_emb = SinusoidalPosEmb(n_emb)       # time difference h = t - r

        # ---- Condition obs embedding ----
        self.cond_obs_emb = None
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        # ---- Condition positional embedding (learnable) ----
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, max_cond_tokens, n_emb))

        # ---- Condition encoder ----
        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=n_cond_layers
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(n_emb, 4 * n_emb),
                nn.Mish(),
                nn.Linear(4 * n_emb, n_emb),
            )

        # ---- Shared decoder blocks ----
        decoder_kwargs = dict(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.shared_blocks = nn.ModuleList(
            [CustomizedTransformerDecoderLayer(**decoder_kwargs) for _ in range(shared_depth)]
        )

        # ---- Dual-head decoder blocks ----
        self.u_head_blocks = nn.ModuleList(
            [CustomizedTransformerDecoderLayer(**decoder_kwargs) for _ in range(aux_head_depth)]
        )
        self.v_head_blocks = nn.ModuleList(
            [CustomizedTransformerDecoderLayer(**decoder_kwargs) for _ in range(aux_head_depth)]
        )

        # ---- Dual output heads ----
        self.u_ln = nn.LayerNorm(n_emb)
        self.u_head = nn.Linear(n_emb, output_dim)
        self.v_ln = nn.LayerNorm(n_emb)
        self.v_head = nn.Linear(n_emb, output_dim)

        # ---- Constants ----
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.horizon_10hz = horizon_10hz
        self.horizon_50hz = horizon_50hz
        self.freq_ratio = freq_ratio
        self.action_offset_10hz = action_offset_10hz
        self.temporal_offset_10hz = temporal_offset_10hz
        self.n_emb = n_emb
        self.aux_head_depth = aux_head_depth

        # ---- Init weights ----
        self.apply(self._init_weights)
        logger.info(
            "TransformerForIMFMultiFreq parameters: %e",
            sum(p.numel() for p in self.parameters()),
        )

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GRU):
            for attr in dir(module):
                if attr.startswith("weight_"):
                    torch.nn.init.normal_(
                        getattr(module, attr), mean=0.0, std=0.02
                    )
                if attr.startswith("bias_"):
                    torch.nn.init.zeros_(getattr(module, attr))
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                "in_proj_weight",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
            ]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForIMFMultiFreq):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    # ------------------------------------------------------------------
    # PE index helpers (unchanged from TransformerForDiffusionMultiFreq)
    # ------------------------------------------------------------------

    def _get_action_pos_indices(
        self, T: int, freq_mode: str, device: torch.device
    ) -> torch.Tensor:
        if freq_mode == "50hz":
            if T > self.horizon_50hz:
                raise ValueError(
                    f"Unsupported 50hz sequence length T={T}. "
                    f"Expected T <= {self.horizon_50hz}."
                )
            return torch.arange(T, device=device)
        if freq_mode == "10hz":
            if T > self.horizon_10hz:
                raise ValueError(
                    f"Unsupported 10hz sequence length T={T}. "
                    f"Expected T <= {self.horizon_10hz}."
                )
            return self.action_offset_10hz + torch.arange(T, device=device) * self.freq_ratio
        raise ValueError(f"Unsupported freq_mode={freq_mode}. Expected '10hz' or '50hz'.")

    def _get_temporal_pos_indices(
        self, T: int, freq_mode: str, device: torch.device
    ) -> torch.Tensor:
        if freq_mode == "50hz":
            if T > self.horizon_50hz:
                raise ValueError(
                    f"Unsupported 50hz sequence length T={T}. "
                    f"Expected T <= {self.horizon_50hz}."
                )
            return torch.arange(T, device=device)
        if freq_mode == "10hz":
            if T > self.horizon_10hz:
                raise ValueError(
                    f"Unsupported 10hz sequence length T={T}. "
                    f"Expected T <= {self.horizon_10hz}."
                )
            return self.temporal_offset_10hz + torch.arange(T, device=device) * self.freq_ratio
        raise ValueError(f"Unsupported freq_mode={freq_mode}. Expected '10hz' or '50hz'.")

    @staticmethod
    def _expand_timesteps(timestep, batch_size: int, device: torch.device):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.float32, device=device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(device)
        else:
            timestep = timestep.to(device)
        return timestep.expand(batch_size)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        h_timestep: Union[torch.Tensor, float, int],
        cond: Optional[torch.Tensor] = None,
        temporal_cond: Optional[torch.Tensor] = None,
        freq_mode: str = "10hz",
        **kwargs,
    ):
        """
        Args:
            sample: (B, T, input_dim) — noisy action tokens
            timestep: (B,) or scalar — continuous diffusion time t in [0, 1]
            h_timestep: (B,) or scalar — continuous time difference h = t - r
            cond: (B, N_obs, cond_dim) — obs tokens
            temporal_cond: (B, T, n_emb) — RNN-encoded force tokens
            freq_mode: "10hz" or "50hz"
        Returns:
            u: (B, T, output_dim) — predicted average velocity field
            v: (B, T, output_dim) — predicted instantaneous velocity field
        """
        B, T, _ = sample.shape
        device = sample.device

        # 1. Timestep embeddings
        timesteps = self._expand_timesteps(timestep, B, device)
        time_token = self.time_emb(timesteps).unsqueeze(1)       # (B, 1, n_emb)

        h_timesteps = self._expand_timesteps(h_timestep, B, device)
        h_token = self.h_emb(h_timesteps).unsqueeze(1)           # (B, 1, n_emb)

        # 2. Build condition embedding: [time | h | obs]
        cond_emb = torch.cat([time_token, h_token], dim=1)       # (B, 2, n_emb)
        if self.cond_obs_emb is not None and cond is not None:
            cond_obs_emb = self.cond_obs_emb(cond)               # (B, N_obs, n_emb)
            cond_emb = torch.cat([cond_emb, cond_obs_emb], dim=1)

        tc = cond_emb.shape[1]
        cond_emb = self.drop(cond_emb + self.cond_pos_emb[:, :tc, :])
        cond_emb = self.encoder(cond_emb)                        # (B, tc, n_emb)

        # 3. Action token embedding with shared PE
        action_pos_idx = self._get_action_pos_indices(T, freq_mode, device)
        input_emb = self.input_emb(sample)                       # (B, T, n_emb)
        input_emb = self.drop(input_emb + self.pos_emb[:, action_pos_idx, :])

        # 4. Temporal condition with shared PE (unchanged from original)
        temporal_cond_emb = temporal_cond
        if temporal_cond_emb is not None:
            assert temporal_cond_emb.shape[1] == T, (
                f"temporal_cond ({temporal_cond_emb.shape[1]}) "
                f"must match sample ({T})"
            )
            temporal_pos_idx = self._get_temporal_pos_indices(T, freq_mode, device)
            temporal_cond_emb = self.drop(
                temporal_cond_emb + self.pos_emb[:, temporal_pos_idx, :]
            )

        # 5. Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=device), diagonal=1
        ).bool()

        # 6. Assemble memory and memory_mask
        if temporal_cond_emb is not None:
            memory = torch.cat([cond_emb, temporal_cond_emb], dim=1)
            memory_mask = torch.zeros(T, tc + T, dtype=torch.bool, device=device)
            memory_mask[:, tc:] = causal_mask
        else:
            memory = cond_emb
            memory_mask = None

        # 7. Shared decoder blocks
        tgt = input_emb
        for block in self.shared_blocks:
            tgt = block(
                tgt, memory,
                temporal_memory=None,
                tgt_mask=causal_mask,
                memory_mask=memory_mask,
                temporal_memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                temporal_memory_key_padding_mask=None,
            )

        # 8. Dual-head decoder blocks (branch from shared output)
        u_tgt = tgt
        v_tgt = tgt

        for block in self.u_head_blocks:
            u_tgt = block(
                u_tgt, memory,
                temporal_memory=None,
                tgt_mask=causal_mask,
                memory_mask=memory_mask,
                temporal_memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                temporal_memory_key_padding_mask=None,
            )

        for block in self.v_head_blocks:
            v_tgt = block(
                v_tgt, memory,
                temporal_memory=None,
                tgt_mask=causal_mask,
                memory_mask=memory_mask,
                temporal_memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                temporal_memory_key_padding_mask=None,
            )

        # 9. Output projections
        u = self.u_head(self.u_ln(u_tgt))    # (B, T, output_dim)
        v = self.v_head(self.v_ln(v_tgt))    # (B, T, output_dim)

        return u, v

    # ------------------------------------------------------------------
    # Optimizer groups (unchanged from TransformerForDiffusionMultiFreq)
    # ------------------------------------------------------------------

    def get_optim_groups(self, weight_decay: float = 1e-3):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            torch.nn.Linear,
            torch.nn.MultiheadAttention,
            torch.nn.GRU,
        )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if pn.endswith("bias") or pn.startswith("bias"):
                    no_decay.add(fpn)
                elif (
                    pn.endswith("weight") or pn.startswith("weight")
                ) and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif (
                    pn.endswith("weight") or pn.startswith("weight")
                ) and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")
        no_decay.add("cond_pos_emb")
        no_decay.add("_dummy_variable")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, (
            "parameters %s made it into both decay/no_decay sets!"
            % (str(inter_params),)
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(param_dict.keys() - union_params),)
        )

        return [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
