"""
Multi-frequency Transformer for Flow Matching (v3).

Supports split forward: T=16 (10hz) or T=80 (50hz) per call, never concatenated.
Key design:
  - Discrete shared PE with modality-specific offsets on the 50hz grid
  - freq_emb: Embedding(2, n_emb) to distinguish 10hz(0) vs 50hz(1)
  - distill_emb: Embedding(2, n_emb) to distinguish standard FM(0) vs distillation(1)
  - Cond structure always: [time_emb(1) | freq(1) | distill(1) | obs(N)]
  - Standard causal mask (no block-diagonal)
"""
from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from ImplicitRDP.model.diffusion.positional_embedding import SinusoidalPosEmb
from ImplicitRDP.model.diffusion.transformer_utils import CustomizedTransformerDecoderLayer
from ImplicitRDP.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)


class TransformerForDiffusionMultiFreq(ModuleAttrMixin):
    """
    Transformer backbone for multi-frequency flow matching.

    Differences from TransformerForDiffusion:
    1. pos_emb is a shared table on the 50hz grid with modality-specific
       alignment offsets for 10hz inputs
    2. freq_emb: Embedding(2, n_emb) added as condition token
    3. distill_emb: Embedding(2, n_emb) added as condition token
    4. forward takes freq_mode and distill_mode parameters
    5. Standard causal mask only (no block-diagonal)
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
    ) -> None:
        super().__init__()

        input_dim = shape_meta["action"]["shape"][0]
        output_dim = shape_meta["action"]["shape"][0]

        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond

        # ---- Input embedding ----
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.drop = nn.Dropout(p_drop_emb)

        # ---- Discrete shared PE table on the 50hz grid ----
        max_action_index = action_offset_10hz + (horizon_10hz - 1) * freq_ratio
        max_temporal_index = temporal_offset_10hz + (horizon_10hz - 1) * freq_ratio
        pos_emb_len = max(horizon_50hz, max_action_index + 1, max_temporal_index + 1)
        self.pos_emb = nn.Parameter(torch.zeros(1, pos_emb_len, n_emb))

        # ---- Diffusion timestep embedding ----
        self.time_emb = SinusoidalPosEmb(n_emb)

        # ---- Frequency embedding ----
        self.freq_emb = nn.Embedding(2, n_emb)  # 0=10hz, 1=50hz

        # ---- Distillation mode embedding ----
        self.distill_emb = nn.Embedding(2, n_emb)  # 0=standard FM, 1=distillation

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

        # ---- Decoder blocks ----
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

        # ---- Output head ----
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # ---- Constants ----
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.horizon_10hz = horizon_10hz
        self.horizon_50hz = horizon_50hz
        self.freq_ratio = freq_ratio
        self.action_offset_10hz = action_offset_10hz
        self.temporal_offset_10hz = temporal_offset_10hz
        self.n_emb = n_emb

        # ---- Init weights ----
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
        elif isinstance(module, TransformerForDiffusionMultiFreq):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def _get_action_pos_indices(
        self, T: int, freq_mode: str, device: torch.device
    ) -> torch.Tensor:
        """Get action-token PE indices on the shared 50hz grid."""
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
        """Get temporal-condition PE indices on the shared 50hz grid."""
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

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Optional[torch.Tensor] = None,
        temporal_cond: Optional[torch.Tensor] = None,
        freq_mode: str = "10hz",
        distill_mode: int = 0,
        **kwargs,
    ):
        """
        Args:
            sample: (B, T, input_dim) - noisy action tokens. T=16 (10hz) or T=80 (50hz)
            timestep: (B,) or scalar - diffusion/flow ODE timestep
            cond: (B, N_obs, cond_dim) - obs tokens
            temporal_cond: (B, T, n_emb) - RNN-encoded force tokens
            freq_mode: "10hz" or "50hz"
            distill_mode: 0 (standard FM) or 1 (distillation)
        Returns:
            (B, T, output_dim) - predicted velocity field
        """
        B, T, _ = sample.shape
        device = sample.device

        # 1. Diffusion timestep embedding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device)
        timesteps = timesteps.expand(B)
        time_emb = self.time_emb(timesteps).unsqueeze(1)  # (B, 1, n_emb)

        # 2. Frequency token
        freq_id = torch.tensor(
            [0 if freq_mode == "10hz" else 1], device=device
        )
        freq_token = self.freq_emb(freq_id).unsqueeze(0).expand(B, -1, -1)  # (B, 1, n_emb)

        # 3. Distillation mode token
        distill_id = torch.tensor([distill_mode], device=device)
        distill_token = self.distill_emb(distill_id).unsqueeze(0).expand(B, -1, -1)  # (B, 1, n_emb)

        # 4. Condition embedding: [time_emb | freq | distill | obs_tokens]
        cond_emb = torch.cat([time_emb, freq_token, distill_token], dim=1)
        if self.cond_obs_emb is not None and cond is not None:
            cond_obs_emb = self.cond_obs_emb(cond)  # (B, N_obs, n_emb)
            cond_emb = torch.cat([cond_emb, cond_obs_emb], dim=1)

        tc = cond_emb.shape[1]
        cond_emb = self.drop(
            cond_emb + self.cond_pos_emb[:, :tc, :]
        )
        cond_emb = self.encoder(cond_emb)  # (B, tc, n_emb)

        # 5. Get PE indices for this frequency and prefix length
        action_pos_idx = self._get_action_pos_indices(T, freq_mode, device)  # (T,)

        # 6. Process action tokens with shared PE
        input_emb = self.input_emb(sample)  # (B, T, n_emb)
        input_emb = self.drop(input_emb + self.pos_emb[:, action_pos_idx, :])

        # 7. Process temporal_cond with shared PE
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

        # 8. Build standard causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=device), diagonal=1
        ).bool()

        # 9. Assemble memory and memory_mask
        if temporal_cond_emb is not None:
            memory = torch.cat([cond_emb, temporal_cond_emb], dim=1)
            # cross_mask: cond part fully visible, temporal part causal
            memory_mask = torch.zeros(
                T, tc + T, dtype=torch.bool, device=device
            )
            memory_mask[:, tc:] = causal_mask
        else:
            memory = cond_emb
            memory_mask = None

        # 10. Decoder
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

        # 11. Output head
        x = self.ln_f(tgt)
        x = self.head(x)  # (B, T, output_dim)
        return x

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """Separate parameters into decay / no-decay groups."""
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
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    no_decay.add(fpn)
                elif (
                    pn.endswith("weight") or pn.startswith("weight")
                ) and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif (
                    pn.endswith("weight") or pn.startswith("weight")
                ) and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Special cases
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

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups
