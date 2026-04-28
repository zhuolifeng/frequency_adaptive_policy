"""
Flow Matching + Confidence Head Policy.

Stage 1: Train FM model (velocity matching loss).
Stage 2: Freeze FM, train Confidence Head using one-step FM action MSE.
Inference: One-step (or multi-step) FM + Confidence output.
"""
import copy
from typing import Dict, List, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import reduce

from loguru import logger

from ImplicitRDP.model.common.normalizer import LinearNormalizer
from ImplicitRDP.policy.base_image_policy import BaseImagePolicy
from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from ImplicitRDP.model.confidence.transformer import TransformerConfidenceHead
from ImplicitRDP.model.force.rnn import RNN


class FMConfidencePolicy(BaseImagePolicy):
    """
    Flow Matching (Rectified Flow) policy with a Confidence Head.

    Two-stage training:
      Stage 1 - train the FM backbone (velocity matching loss).
      Stage 2 - freeze FM, train confidence head on one-step FM action MSE.
    """

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: torch.nn.Module,
        model: TransformerForDiffusion,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: int = 1,
        # flow matching time sampling
        beta_a: float = 1.5,
        beta_b: float = 1.0,
        # arch
        n_emb: int = 768,
        use_rnn_obs_encoder: bool = False,
        rnn_obs_encoder_n_layer: int = 1,
        rnn_obs_encoder_hidden_dim: int = 512,
        use_amp_for_inference: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        # confidence head
        confidence_hidden_dim: Optional[int] = None,
        confidence_gamma: float = 1.0,
        confidence_alpha: float = 0.2,
        confidence_log_sigma_min: float = -5.0,
        confidence_log_sigma_max: float = 5.0,
        confidence_clamp_log_sigma: bool = False,
        confidence_obs_encoder_n_layers: int = 2,
        confidence_obs_encoder_hidden_dim: Optional[int] = None,
        use_confidence_head: bool = True,
        confidence_head_type: str = "transformer",
        confidence_transformer_n_heads: int = 8,
        confidence_transformer_dropout: float = 0.1,
        confidence_action_dims: int = 6,
        normalize_confidence_loss: bool = False,
        # two-stage training
        training_stage: int = 1,
        **kwargs,
    ):
        super().__init__()

        # ---- action / obs shapes ----
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = horizon

        # ---- extended obs keys ----
        all_extented_obs_keys = (
            list(shape_meta["extended_obs"].keys())
            if "extended_obs" in shape_meta
            else []
        )
        self.extented_obs_keys = sorted(all_extented_obs_keys)
        temporal_cond_dim = sum(
            shape_meta["extended_obs"][k]["shape"][-1] for k in self.extented_obs_keys
        )

        # ---- RNN obs encoder (optional) ----
        if use_rnn_obs_encoder:
            assert len(self.extented_obs_keys) > 0, (
                "extended_obs is required for RNNObsEncoder"
            )
            self.rnn_obs_encoder = RNN(
                input_dim=temporal_cond_dim,
                hidden_dim=rnn_obs_encoder_hidden_dim,
                layer_num=rnn_obs_encoder_n_layer,
                n_emb=n_emb,
            )
        else:
            self.rnn_obs_encoder = None

        # ---- core modules ----
        self.shape_meta = shape_meta
        self.obs_encoder = obs_encoder
        self.model = model  # TransformerForDiffusion used as FM backbone
        self.normalizer = LinearNormalizer()

        # ---- scalars ----
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.n_emb = n_emb
        self.num_inference_steps = num_inference_steps
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.kwargs = kwargs

        # ---- AMP ----
        self.use_amp_for_inference = use_amp_for_inference
        if use_amp_for_inference:
            if amp_dtype == "bf16" and torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
            elif amp_dtype == "fp16":
                self.amp_dtype = torch.float16
            else:
                self.amp_dtype = torch.float16
        else:
            self.amp_dtype = None

        # ---- confidence head ----
        self.confidence_gamma = confidence_gamma
        self.confidence_alpha = confidence_alpha
        self.confidence_log_sigma_min = confidence_log_sigma_min
        self.confidence_log_sigma_max = confidence_log_sigma_max
        self.use_confidence_head = use_confidence_head
        self.confidence_action_dims = confidence_action_dims
        self.normalize_confidence_loss = normalize_confidence_loss
        self.confidence_head_type = (confidence_head_type or "transformer").lower()

        hidden_dim = confidence_hidden_dim or n_emb
        self.confidence_head = None
        if self.use_confidence_head:
            if self.confidence_head_type == "transformer":
                self.confidence_head = TransformerConfidenceHead(
                    n_emb=n_emb,
                    hidden_dim=hidden_dim,
                    obs_encoder_n_layers=confidence_obs_encoder_n_layers,
                    obs_encoder_hidden_dim=confidence_obs_encoder_hidden_dim,
                    n_heads=confidence_transformer_n_heads,
                    dropout=confidence_transformer_dropout,
                    log_sigma_min=confidence_log_sigma_min,
                    log_sigma_max=confidence_log_sigma_max,
                    clamp_log_sigma=confidence_clamp_log_sigma,
                )
            else:
                raise ValueError(
                    f"Unsupported confidence_head_type '{confidence_head_type}'. "
                    "Choose from 'transformer'."
                )

        # ---- training stage ----
        self.training_stage = training_stage
        self.progress = 0.0

        logger.info(
            "FMConfidencePolicy initialized "
            f"(stage={training_stage}, confidence_head={use_confidence_head})"
        )

    # ------------------------------------------------------------------
    # Normalizer
    # ------------------------------------------------------------------
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def set_training_stage(self, stage: int):
        assert stage in (1, 2), f"Invalid stage {stage}, must be 1 or 2"
        self.training_stage = stage

    def update_progress(self, progress: float):
        self.progress = progress

    # ------------------------------------------------------------------
    # Temporal condition helpers (from FlowMatchingTransformerImagePolicy)
    # ------------------------------------------------------------------
    def get_temporal_cond(
        self,
        extended_obs_dict,
        extended_obs_last_step=None,
        extend_obs_pad_after_n=None,
        allow_partial: bool = False,
        post_process: bool = False,
    ):
        if extended_obs_dict is None or len(self.extented_obs_keys) == 0:
            return None

        temporal_cond = []
        for extented_obs_key in self.extented_obs_keys:
            if extented_obs_key not in extended_obs_dict:
                if allow_partial:
                    continue
                raise KeyError(f"Missing extended obs key: {extented_obs_key}")

            extented_obs = extended_obs_dict[extented_obs_key]
            obs_dim = len(
                self.shape_meta["extended_obs"][extented_obs_key]["shape"]
            )

            if extended_obs_last_step is not None:
                if obs_dim == 1:
                    extented_obs = extented_obs[..., -extended_obs_last_step:, :]
                elif obs_dim == 2:
                    extented_obs = extented_obs[
                        ..., -extended_obs_last_step:, :, :
                    ]
                else:
                    raise ValueError("Only support 1d or 2d extended obs")

            if extend_obs_pad_after_n is not None:
                if obs_dim == 1:
                    padding_obs = extended_obs_dict[extented_obs_key][
                        ..., -1:, :
                    ].repeat(1, extend_obs_pad_after_n, 1)
                elif obs_dim == 2:
                    padding_obs = extended_obs_dict[extented_obs_key][
                        ..., -1:, :, :
                    ].repeat(1, extend_obs_pad_after_n, 1, 1)
                else:
                    raise ValueError("Only support 1d or 2d extended obs")
                extented_obs = torch.cat([padding_obs, extented_obs], dim=-2)

            extented_obs = self.normalizer[
                f"extended_{extented_obs_key}"
            ].normalize(extented_obs)
            temporal_cond.append(extented_obs)

        if len(temporal_cond) == 0:
            return None

        temporal_cond = torch.cat(temporal_cond, dim=-1)

        if post_process and self.rnn_obs_encoder is not None:
            temporal_cond = self.rnn_obs_encoder(temporal_cond)

        return temporal_cond

    # ------------------------------------------------------------------
    # Obs tokens
    # ------------------------------------------------------------------
    def _predict_obs_tokens(
        self,
        obs_dict: Dict[str, torch.Tensor],
        extended_obs_dict=None,
        return_expanded_obs_tokens: bool = False,
        return_temporal_cond: bool = False,
    ):
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])

        temporal_cond = None
        if self.rnn_obs_encoder is not None and extended_obs_dict is not None:
            temporal_cond = self.get_temporal_cond(
                extended_obs_dict, allow_partial=True, post_process=True
            )

        if return_temporal_cond:
            return obs_tokens, temporal_cond

        obs_flat = obs_tokens.reshape(B, -1)
        if return_expanded_obs_tokens:
            obs_flat = obs_flat.unsqueeze(1).expand(-1, self.action_horizon, -1)
        return obs_flat

    def predict_obs_tokens(self, *args, **kwargs):
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_obs_tokens(*args, **kwargs)

    # ------------------------------------------------------------------
    # FM Euler ODE sampler (from FlowMatchingTransformerImagePolicy)
    # ------------------------------------------------------------------
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        cond=None,
        temporal_cond=None,
        noisy_trajectory=None,
        generator=None,
        num_inference_steps: Optional[int] = None,
        **kwargs,
    ):
        """Euler ODE solver for Rectified Flow: t=1 (noise) -> t=0 (data)."""
        model = self.model
        steps = num_inference_steps if num_inference_steps is not None else self.num_inference_steps

        if noisy_trajectory is None:
            trajectory = torch.randn(
                size=condition_data.shape,
                dtype=condition_data.dtype,
                device=condition_data.device,
                generator=generator,
            )
        else:
            trajectory = noisy_trajectory

        B = trajectory.shape[0]

        # RNN computed once outside the Euler loop
        if self.rnn_obs_encoder is not None and temporal_cond is not None:
            temporal_cond_tokens = self.rnn_obs_encoder(temporal_cond)
        else:
            temporal_cond_tokens = None

        t_schedule = torch.linspace(
            1.0, 0.0, steps + 1, device=trajectory.device
        )
        for i in range(steps):
            
            trajectory[condition_mask] = condition_data[condition_mask]

            t_cur = t_schedule[i]
            dt = t_schedule[i + 1] - t_schedule[i]

            t_tensor = torch.full(
                (B,), t_cur * 1000, device=trajectory.device
            ).long()

            model_output = model(
                trajectory,
                t_tensor,
                cond=cond,
                temporal_cond=(
                    temporal_cond_tokens
                    if temporal_cond_tokens is not None
                    else temporal_cond
                ),
            )

            trajectory = trajectory + dt * model_output

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    # ------------------------------------------------------------------
    # Noisy trajectory helper
    # ------------------------------------------------------------------
    def sample_noisy_trajectory(self, B=1, generator=None):
        noisy_trajectory = torch.randn(
            size=(B, self.action_horizon, self.action_dim),
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )
        return noisy_trajectory.reshape(B, -1)

    # ------------------------------------------------------------------
    # Inference: predict action
    # ------------------------------------------------------------------
    def _predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        dataset_obs_temporal_downsample_ratio: int = 1,
        extended_obs_dict: Dict[str, torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict

        obs_tokens = self.predict_obs_tokens(obs_dict)
        B = obs_tokens.shape[0]
        obs_tokens = obs_tokens.reshape(B, -1, self.n_emb)

        if self.rnn_obs_encoder is not None:
            assert extended_obs_dict is not None
            temporal_cond = self.get_temporal_cond(extended_obs_dict)
        else:
            temporal_cond = None

        if temporal_cond is None:
            cond_data = torch.zeros(
                (B, self.action_horizon, self.action_dim),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            cond_data = torch.zeros(
                (B, temporal_cond.shape[1], self.action_dim),
                device=self.device,
                dtype=self.dtype,
            )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            cond=obs_tokens,
            temporal_cond=temporal_cond,
            num_inference_steps=num_inference_steps,
            **self.kwargs,
        )

        assert nsample.shape == (B, cond_data.shape[1], self.action_dim)
        action_pred = self.normalizer["action"].unnormalize(nsample)

        To = self.n_obs_steps * dataset_obs_temporal_downsample_ratio
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {"action": action, "action_pred": action_pred}

    def predict_action(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_action(*args, **kwargs)

    # ------------------------------------------------------------------
    # Batch inference from pre-computed obs tokens
    # ------------------------------------------------------------------
    def _predict_from_obs_tokens_and_noisy_trajectory(
        self,
        obs_tokens_and_noisy_trajectory_list: List[torch.Tensor],
        extended_obs_dict: Union[
            Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]
        ],
        extended_obs_last_step_list: List[int],
        dataset_obs_temporal_downsample_ratio: int,
        extend_obs_pad_after: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        obs_tokens_and_noisy_trajectory = torch.cat(
            obs_tokens_and_noisy_trajectory_list, dim=0
        )
        B = obs_tokens_and_noisy_trajectory.shape[0]
        obs_tokens = obs_tokens_and_noisy_trajectory[
            :, : -self.action_horizon * self.action_dim
        ]
        noisy_trajectory = obs_tokens_and_noisy_trajectory[
            :, -self.action_horizon * self.action_dim :
        ]
        obs_tokens = obs_tokens.to(self.device).reshape(B, -1, self.n_emb)
        noisy_trajectory = noisy_trajectory.to(self.device).reshape(
            B, self.action_horizon, self.action_dim
        )

        max_extended_obs_last_step = max(extended_obs_last_step_list)
        assert extended_obs_dict is not None
        if extend_obs_pad_after:
            extend_obs_pad_after_n = (
                self.action_horizon - max_extended_obs_last_step
            )
        else:
            extend_obs_pad_after_n = None

        if isinstance(extended_obs_dict, list):
            temporal_cond = [
                self.get_temporal_cond(
                    extended_obs_dict[i],
                    max_extended_obs_last_step,
                    extend_obs_pad_after_n=extend_obs_pad_after_n,
                )
                for i in range(len(extended_obs_dict))
            ]
        else:
            temporal_cond = self.get_temporal_cond(
                extended_obs_dict,
                max_extended_obs_last_step,
                extend_obs_pad_after_n=extend_obs_pad_after_n,
            )

        temporal_cond_list = []
        for i in range(B):
            if isinstance(temporal_cond, list):
                cond = temporal_cond[i]
            else:
                cond = temporal_cond
            if extended_obs_last_step_list[i] < max_extended_obs_last_step:
                padding_obs = cond[:, -1:, :].repeat(
                    1,
                    max_extended_obs_last_step - extended_obs_last_step_list[i],
                    1,
                )
                temporal_cond_list.append(
                    torch.cat(
                        [
                            cond[:, -extended_obs_last_step_list[i] :, :],
                            padding_obs,
                        ],
                        dim=-2,
                    )
                )
            else:
                temporal_cond_list.append(cond)
        temporal_cond = torch.cat(temporal_cond_list, dim=0)

        cond_data = torch.zeros(
            (B, temporal_cond.shape[1], self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        noisy_trajectory = noisy_trajectory[:, : temporal_cond.shape[1], :]

        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            cond=obs_tokens,
            temporal_cond=temporal_cond,
            noisy_trajectory=noisy_trajectory,
            **self.kwargs,
        )

        assert nsample.shape == (B, temporal_cond.shape[1], self.action_dim)
        action_pred = self.normalizer["action"].unnormalize(nsample)

        To = self.n_obs_steps * dataset_obs_temporal_downsample_ratio
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        results = []
        for i in range(B):
            if extend_obs_pad_after:
                results.append(
                    {"action": action[i : i + 1], "action_pred": action_pred[i : i + 1]}
                )
            else:
                results.append(
                    {
                        "action": action[
                            i : i + 1, : extended_obs_last_step_list[i] - start, :
                        ],
                        "action_pred": action_pred[
                            i : i + 1, : extended_obs_last_step_list[i], :
                        ],
                    }
                )
        return results

    def predict_from_obs_tokens_and_noisy_trajectory(self, *args, **kwargs):
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_from_obs_tokens_and_noisy_trajectory(
                *args, **kwargs
            )

    # ------------------------------------------------------------------
    # Confidence helpers
    # ------------------------------------------------------------------
    def _compute_confidence_logits(
        self, obs_tokens: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self.confidence_head is None:
            return None
        return self.confidence_head(obs_tokens)

    def _compute_confidence_loss(
        self,
        obs_tokens: torch.Tensor,
        fm_loss_vec_conf: torch.Tensor,
        loss_mask=None,
    ):
        if self.confidence_head is None:
            return None, None, None, None

        log_sigma_raw = self._compute_confidence_logits(obs_tokens)
        if log_sigma_raw is None:
            return None, None, None, None

        log_sigma = (
            log_sigma_raw[loss_mask] if loss_mask is not None else log_sigma_raw
        )
        fm_loss_masked = (
            fm_loss_vec_conf[loss_mask]
            if loss_mask is not None
            else fm_loss_vec_conf
        )

        sigma = 1.0 + torch.exp(log_sigma)
        sigma_stats = sigma
        log_sigma_term = torch.log(sigma)

        gamma = self.confidence_gamma
        alpha = self.confidence_alpha
        conf_loss = gamma * fm_loss_masked * sigma - alpha * log_sigma_term
        conf_loss = conf_loss.mean()

        return conf_loss, log_sigma, sigma_stats, fm_loss_masked

    # ------------------------------------------------------------------
    # FM model_forward (training, from FlowMatchingTransformerImagePolicy)
    # ------------------------------------------------------------------
    def model_forward(self, batch):
        """Rectified Flow forward pass for velocity matching training."""
        assert "valid_mask" not in batch

        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        trajectory = nactions
        B = nactions.shape[0]

        if self.rnn_obs_encoder is not None:
            assert "extended_obs" in batch
            temporal_cond = self.get_temporal_cond(batch["extended_obs"])
        else:
            temporal_cond = None

        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])

        # Sample t ~ Beta
        t = torch.distributions.Beta(self.beta_a, self.beta_b).sample((B,)).to(
            trajectory.device
        )
        t = t * 0.999 + 0.001

        x1 = torch.randn(trajectory.shape, device=trajectory.device)

        t_expanded = t.view(B, 1, 1)
        x_t = t_expanded * x1 + (1 - t_expanded) * trajectory

        timesteps = (t * 1000).long()

        if self.rnn_obs_encoder is not None and temporal_cond is not None:
            temporal_cond_tokens = self.rnn_obs_encoder(temporal_cond)
        else:
            temporal_cond_tokens = None

        pred = self.model(
            x_t,
            timesteps,
            cond=obs_tokens,
            temporal_cond=(
                temporal_cond_tokens
                if temporal_cond_tokens is not None
                else temporal_cond
            ),
        )

        target = x1 - trajectory
        return trajectory, x1, t, pred, target, obs_tokens

    # ------------------------------------------------------------------
    # One-step FM MSE (for confidence training)
    # ------------------------------------------------------------------
    def _compute_fm_one_step_mse(
        self,
        nactions: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond_tokens: Optional[torch.Tensor],
        loss_mask=None,
        n_samples: int = 8,
        num_denoise_steps: int = 2,
    ):
        """
        Multi-step Euler MSE on actions (distill=0).
        Average over n_samples noise draws to get a stable MSE target.

        Each noise draw runs `num_denoise_steps`-step Euler ODE from t=1→0,
        matching actual inference behavior. Total forward passes per batch:
        n_samples * num_denoise_steps (default: 8 * 10 = 80).

        Why multiple samples?
        ---------------------
        ACT is deterministic: same obs → same prediction → same MSE.
        FM is stochastic: same obs + different noise → different MSE.
        A single noise draw gives a target dominated by noise variance,
        causing the confidence head to collapse to a constant.

        By averaging n_samples multi-step predictions, the noise variance
        is reduced by ~1/n_samples. The remaining MSE signal reflects
        the FM model's true capability on each observation — easy obs
        consistently get low average MSE, hard obs consistently get
        high average MSE.
        """
        B = nactions.shape[0]
        device = nactions.device

        # Accumulate per-dim MSE over n_samples noise draws
        per_sample_loss_all_accum = torch.zeros_like(nactions)

        with torch.no_grad():
            for _ in range(n_samples):
                x1 = torch.randn_like(nactions)

                # Multi-step Euler ODE: t=1.0 → t=0.0
                trajectory = x1
                t_schedule = torch.linspace(
                    1.0, 0.0, num_denoise_steps + 1, device=device
                )
                for i in range(num_denoise_steps):
                    t_cur = t_schedule[i]
                    dt = t_schedule[i + 1] - t_schedule[i]
                    t_tensor = torch.full(
                        (B,), t_cur * 1000, device=device
                    ).long()
                    pred_v = self.model(
                        trajectory,
                        t_tensor,
                        cond=obs_tokens,
                        temporal_cond=temporal_cond_tokens,
                    )
                    trajectory = trajectory + dt * pred_v

                x0_pred = trajectory

                per_sample_loss_all_accum += F.mse_loss(
                    x0_pred, nactions, reduction="none"
                )

        # Average over noise samples
        per_sample_loss_all = per_sample_loss_all_accum / n_samples

        per_sample_loss = reduce(per_sample_loss_all, "b t a -> b", "mean")

        # Confidence dims (first confidence_action_dims)
        conf_dims = min(
            self.confidence_action_dims, per_sample_loss_all.shape[-1]
        )
        sel_loss = per_sample_loss_all[..., :conf_dims]
        if self.normalize_confidence_loss:
            gt_sq = nactions[..., :conf_dims] ** 2
            sel_loss = sel_loss / (gt_sq + 1e-8)
        per_sample_loss_conf = reduce(sel_loss, "b t a -> b", "mean")

        # Virtual dims (remaining)
        if per_sample_loss_all.shape[-1] > conf_dims:
            sel_virtual = per_sample_loss_all[..., conf_dims:]
            per_sample_loss_virtual = reduce(
                sel_virtual, "b t a -> b", "mean"
            )
        else:
            per_sample_loss_virtual = torch.zeros(
                B, device=device, dtype=per_sample_loss_all.dtype
            )

        if loss_mask is None:
            loss_mask = torch.ones(B, dtype=torch.bool, device=device)
        masked_loss = per_sample_loss[loss_mask]
        loss = masked_loss.mean()

        return loss, per_sample_loss, per_sample_loss_conf, per_sample_loss_virtual

    # ------------------------------------------------------------------
    # compute_loss: two-stage
    # ------------------------------------------------------------------
    def compute_loss(self, batch, loss_mask=None):
        # ==================== Stage 1: FM velocity matching ====================
        if self.training_stage == 1:
            trajectory, x1, t, pred, target, obs_tokens = self.model_forward(
                batch
            )
            B = pred.shape[0]
            device = pred.device

            if loss_mask is None:
                loss_mask = torch.ones(B, dtype=torch.bool, device=device)

            loss_all = F.mse_loss(pred, target, reduction="none")
            loss_all = loss_all.type(loss_all.dtype)
            loss_per = reduce(loss_all, "b ... -> b (...)", "mean")
            loss_per = loss_per[loss_mask]
            loss_per = loss_per.mean(dim=1)
            fm_loss = loss_per.sum() / (loss_mask.sum() + 1e-8)

            # Front 6 dims velocity matching loss
            front6_loss_all = loss_all[..., :6]
            front6_loss_per = reduce(front6_loss_all, "b ... -> b (...)", "mean")
            front6_loss_per = front6_loss_per[loss_mask].mean(dim=1)
            fm_loss_front6 = front6_loss_per.sum() / (loss_mask.sum() + 1e-8)

            # Remaining dims velocity matching loss
            if loss_all.shape[-1] > 6:
                remain_loss_all = loss_all[..., 6:]
                remain_loss_per = reduce(remain_loss_all, "b ... -> b (...)", "mean")
                remain_loss_per = remain_loss_per[loss_mask].mean(dim=1)
                fm_loss_remain = remain_loss_per.sum() / (loss_mask.sum() + 1e-8)
            else:
                fm_loss_remain = torch.tensor(0.0, device=device)

            loss_dict = {
                "fm_velocity_loss": fm_loss.item(),
                "fm_velocity_loss_front6": fm_loss_front6.item(),
                "fm_velocity_loss_remain": fm_loss_remain.item(),
                "total_loss": fm_loss.item(),
                "training_stage": 1.0,
            }
            return fm_loss, loss_dict

        # ==================== Stage 2: Confidence on one-step FM MSE ===========
        elif self.training_stage == 2:
            nobs = self.normalizer.normalize(batch["obs"])
            nactions = self.normalizer["action"].normalize(batch["action"])
            B = nactions.shape[0]
            device = nactions.device

            if self.rnn_obs_encoder is not None:
                assert "extended_obs" in batch
                temporal_cond = self.get_temporal_cond(batch["extended_obs"])
            else:
                temporal_cond = None

            # Encode obs (no grad for encoder in stage 2)
            with torch.no_grad():
                this_nobs = dict_apply(
                    nobs, lambda x: x.reshape(-1, *x.shape[2:])
                )
                obs_tokens = self.obs_encoder(this_nobs)
                if len(obs_tokens.shape) == 2:
                    obs_tokens = obs_tokens.reshape(
                        B, -1, *obs_tokens.shape[1:]
                    )

                # temporal cond tokens
                if (
                    self.rnn_obs_encoder is not None
                    and temporal_cond is not None
                ):
                    temporal_cond_tokens = self.rnn_obs_encoder(temporal_cond)
                else:
                    temporal_cond_tokens = temporal_cond

            # One-step FM MSE (FM is frozen via no_grad inside)
            (
                fm_mse,
                fm_mse_vec,
                fm_mse_vec_conf,
                fm_mse_vec_virtual,
            ) = self._compute_fm_one_step_mse(
                nactions, obs_tokens, temporal_cond_tokens, loss_mask
            )

            # Confidence loss
            (
                conf_loss,
                log_sigma,
                sigma_stats,
                fm_loss_normalized,
            ) = self._compute_confidence_loss(
                obs_tokens, fm_mse_vec_conf, loss_mask
            )

            if conf_loss is not None:
                total_loss = conf_loss
            else:
                total_loss = fm_mse

            loss_dict = {
                "fm_one_step_mse": fm_mse.item(),
                "total_loss": total_loss.item(),
                "training_stage": 2.0,
            }

            if log_sigma is not None and sigma_stats is not None:
                loss_dict.update(
                    {
                        "confidence/log_sigma": log_sigma.mean().item(),
                        "confidence/log_sigma_min": log_sigma.min().item(),
                        "confidence/log_sigma_max": log_sigma.max().item(),
                        "confidence/sigma": sigma_stats.mean().item(),
                        "confidence/fm_origin_loss": fm_loss_normalized.mean().item(),
                    }
                )
                if conf_loss is not None:
                    loss_dict["confidence/conf_loss"] = conf_loss.item()

            return total_loss, loss_dict

        else:
            raise ValueError(
                f"Invalid training_stage: {self.training_stage}. Must be 1 or 2."
            )

    def forward(self, batch, loss_mask=None):
        return self.compute_loss(batch, loss_mask)

    # ------------------------------------------------------------------
    # Optimizer: stage-aware
    # ------------------------------------------------------------------
    def get_optimizer(
        self,
        lr: float,
        weight_decay: float,
        obs_encoder_lr: Optional[float] = None,
        obs_encoder_weight_decay: Optional[float] = None,
        betas: Tuple[float, float] = (0.9, 0.95),
        **kwargs,
    ) -> torch.optim.Optimizer:
        if obs_encoder_lr is None:
            obs_encoder_lr = lr
        if obs_encoder_weight_decay is None:
            obs_encoder_weight_decay = weight_decay

        optim_groups = []

        if self.training_stage == 1:
            # FM model
            optim_groups.extend(
                self.model.get_optim_groups(weight_decay=weight_decay)
            )
            # RNN obs encoder
            if self.rnn_obs_encoder is not None:
                optim_groups += self.rnn_obs_encoder.get_optim_groups(
                    weight_decay=weight_decay
                )
            # Obs encoder
            backbone_params = []
            other_obs_params = []
            for key, value in self.obs_encoder.named_parameters():
                if key.startswith("key_model_map"):
                    backbone_params.append(value)
                else:
                    other_obs_params.append(value)
            optim_groups.append(
                {
                    "params": backbone_params,
                    "weight_decay": obs_encoder_weight_decay,
                    "lr": obs_encoder_lr,
                }
            )
            optim_groups.append(
                {
                    "params": other_obs_params,
                    "weight_decay": obs_encoder_weight_decay,
                }
            )

        elif self.training_stage == 2:
            # Only confidence head
            if self.confidence_head is not None:
                if hasattr(self.confidence_head, "get_optim_groups"):
                    optim_groups += self.confidence_head.get_optim_groups(
                        weight_decay=weight_decay
                    )
                else:
                    optim_groups.append(
                        {
                            "params": list(
                                self.confidence_head.parameters()
                            ),
                            "weight_decay": weight_decay,
                        }
                    )

        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        return optimizer
