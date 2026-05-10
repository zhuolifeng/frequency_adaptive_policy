"""
Multi-frequency Transformer for RDT-style flow-matching distillation.

This backbone keeps the shared 50Hz positional grid from
TransformerForDiffusionMultiFreq, but does not inject frequency or distillation
mode tokens. The frequency mode only selects action/temporal position indices.
"""
from typing import Optional, Tuple, Union
import logging

import torch
import torch.nn as nn

from ImplicitRDP.model.common.module_attr_mixin import ModuleAttrMixin
from ImplicitRDP.model.diffusion.positional_embedding import SinusoidalPosEmb
from ImplicitRDP.model.diffusion.transformer_utils import (
    CustomizedTransformerDecoderLayer,
)

logger = logging.getLogger(__name__)


class TransformerForRDTDistillMultiFreq(ModuleAttrMixin):
    """Transformer backbone for 10Hz FM and 50Hz one-step total velocity."""

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
        horizon_10hz: int = 16,
        horizon_50hz: int = 80,
        freq_ratio: int = 5,
        action_offset_10hz: int = 8,
        temporal_offset_10hz: int = 4,
    ) -> None:
        super().__init__()

        input_dim = shape_meta["action"]["shape"][0]
        output_dim = shape_meta["action"]["shape"][0]

        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond

        self.input_emb = nn.Linear(input_dim, n_emb)
        self.drop = nn.Dropout(p_drop_emb)

        max_action_index = action_offset_10hz + (horizon_10hz - 1) * freq_ratio
        max_temporal_index = temporal_offset_10hz + (horizon_10hz - 1) * freq_ratio
        pos_emb_len = max(horizon_50hz, max_action_index + 1, max_temporal_index + 1)
        self.pos_emb = nn.Parameter(torch.zeros(1, pos_emb_len, n_emb))

        self.time_emb = SinusoidalPosEmb(n_emb)

        self.cond_obs_emb = None
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = nn.Parameter(torch.zeros(1, max_cond_tokens, n_emb))

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
                encoder_layer=encoder_layer,
                num_layers=n_cond_layers,
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(n_emb, 4 * n_emb),
                nn.Mish(),
                nn.Linear(4 * n_emb, n_emb),
            )

        self.decoder_blocks = nn.ModuleList(
            [
                CustomizedTransformerDecoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.horizon_10hz = horizon_10hz
        self.horizon_50hz = horizon_50hz
        self.freq_ratio = freq_ratio
        self.action_offset_10hz = action_offset_10hz
        self.temporal_offset_10hz = temporal_offset_10hz
        self.n_emb = n_emb

        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e",
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
                    torch.nn.init.normal_(getattr(module, attr), mean=0.0, std=0.02)
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
        elif isinstance(module, TransformerForRDTDistillMultiFreq):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

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

    def get_optim_groups(self, weight_decay: float = 1e-3):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention, torch.nn.GRU)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias") or pn.startswith("bias"):
                    no_decay.add(fpn)
                elif (pn.endswith("weight") or pn.startswith("weight")) and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif (pn.endswith("weight") or pn.startswith("weight")) and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")
        no_decay.add("cond_pos_emb")
        no_decay.add("_dummy_variable")
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, (
            f"parameters {param_dict.keys() - union_params} were not separated into either decay/no_decay sets!"
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

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Optional[torch.Tensor] = None,
        temporal_cond: Optional[torch.Tensor] = None,
        freq_mode: str = "10hz",
        **kwargs,
    ):
        B, T, _ = sample.shape
        device = sample.device

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device)
        timesteps = timesteps.expand(B)
        time_emb = self.time_emb(timesteps).unsqueeze(1)

        cond_emb = time_emb
        if self.cond_obs_emb is not None and cond is not None:
            cond_obs_emb = self.cond_obs_emb(cond)
            cond_emb = torch.cat([cond_emb, cond_obs_emb], dim=1)

        tc = cond_emb.shape[1]
        cond_emb = self.drop(cond_emb + self.cond_pos_emb[:, :tc, :])
        cond_emb = self.encoder(cond_emb)

        action_pos_idx = self._get_action_pos_indices(T, freq_mode, device)
        input_emb = self.input_emb(sample)
        input_emb = self.drop(input_emb + self.pos_emb[:, action_pos_idx, :])

        temporal_cond_emb = temporal_cond
        if temporal_cond_emb is not None:
            assert temporal_cond_emb.shape[1] == T, (
                f"temporal_cond ({temporal_cond_emb.shape[1]}) must match sample ({T})"
            )
            temporal_pos_idx = self._get_temporal_pos_indices(T, freq_mode, device)
            temporal_cond_emb = self.drop(
                temporal_cond_emb + self.pos_emb[:, temporal_pos_idx, :]
            )

        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        if temporal_cond_emb is not None:
            memory = torch.cat([cond_emb, temporal_cond_emb], dim=1)
            memory_mask = torch.zeros(T, tc + T, dtype=torch.bool, device=device)
            memory_mask[:, tc:] = causal_mask
        else:
            memory = cond_emb
            memory_mask = None

        tgt = input_emb
        for block in self.decoder_blocks:
            tgt = block(
                tgt,
                memory,
                temporal_memory=None,
                tgt_mask=causal_mask,
                memory_mask=memory_mask,
                temporal_memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                temporal_memory_key_padding_mask=None,
            )

        x = self.ln_f(tgt)
        return self.head(x)
