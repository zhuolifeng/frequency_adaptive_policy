"""
Multi-frequency Flow Matching + Confidence Head Policy.

Extends FMMultiFreqPolicy with a confidence head for:
  Stage 1: FM velocity matching (from base policy)
  Stage 2: 10hz FM + 50hz CD distillation (from base policy)
  Stage 3: Freeze FM, train Confidence Head on one-step FM MSE (10hz part)

Inference:
  confidence > threshold → 50hz consistency sampling (distill=1, fast)
  confidence ≤ threshold → 10hz N-step Euler (distill=0, precise)
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
from ImplicitRDP.policy.fm_multifreq_policy import FMMultiFreqPolicy
from ImplicitRDP.model.diffusion.transformer_for_diffusion_multifreq import (
    TransformerForDiffusionMultiFreq,
)
from ImplicitRDP.model.confidence.transformer import TransformerConfidenceHead
from ImplicitRDP.model.force.rnn import RNN


class FMMultiFreqConfidencePolicy(FMMultiFreqPolicy):
    """
    Multi-frequency Flow Matching policy with Confidence Head.

    Three-stage training:
      Stage 1 - FM velocity matching (10hz + 50hz, both distill=0)
      Stage 2 - 10hz FM + 50hz Consistency Distillation
      Stage 3 - Freeze FM, train confidence head on 10hz one-step FM MSE
    """

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: torch.nn.Module,
        model: TransformerForDiffusionMultiFreq,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: int = 5,
        beta_a: float = 1.5,
        beta_b: float = 1.0,
        n_emb: int = 768,
        use_rnn_obs_encoder: bool = True,
        rnn_obs_encoder_n_layer: int = 1,
        rnn_obs_encoder_hidden_dim: int = 512,
        use_amp_for_inference: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        # Multi-freq
        horizon_10hz: int = 16,
        horizon_50hz: int = 80,
        action_start_10hz: int = 1,
        action_start_50hz: int = 13,
        # Distillation
        use_distillation: bool = True,
        cd_num_scales: int = 40,
        stage2_use_10hz_loss: bool = True,
        stage2_use_delta_loss: bool = False,
        stage2_delta_loss_weight: float = 0.05,
        # Confidence head
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
        # Confidence-based inference
        confidence_threshold: float = 0.0,
        num_inference_steps_precise: int = 10,
        # Inference head selection
        inference_head: str = "10hz",
        use_confidence_based_head_selection: bool = True,
        confidence_threshold_for_head_selection: float = 0.0,
        # Training stage
        training_stage: int = 1,
        **kwargs,
    ):
        super().__init__(
            shape_meta=shape_meta,
            obs_encoder=obs_encoder,
            model=model,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            num_inference_steps=num_inference_steps,
            beta_a=beta_a,
            beta_b=beta_b,
            n_emb=n_emb,
            use_rnn_obs_encoder=use_rnn_obs_encoder,
            rnn_obs_encoder_n_layer=rnn_obs_encoder_n_layer,
            rnn_obs_encoder_hidden_dim=rnn_obs_encoder_hidden_dim,
            use_amp_for_inference=use_amp_for_inference,
            amp_dtype=amp_dtype,
            horizon_10hz=horizon_10hz,
            horizon_50hz=horizon_50hz,
            action_start_10hz=action_start_10hz,
            action_start_50hz=action_start_50hz,
            use_distillation=use_distillation,
            cd_num_scales=cd_num_scales,
            stage2_use_10hz_loss=stage2_use_10hz_loss,
            stage2_use_delta_loss=stage2_use_delta_loss,
            stage2_delta_loss_weight=stage2_delta_loss_weight,
            inference_head=inference_head,
            training_stage=training_stage,
            **kwargs,
        )

        # ---- Confidence head ----
        self.confidence_gamma = confidence_gamma
        self.confidence_alpha = confidence_alpha
        self.use_confidence_head = use_confidence_head
        self.confidence_action_dims = confidence_action_dims
        self.normalize_confidence_loss = normalize_confidence_loss
        self.confidence_threshold = confidence_threshold
        self.num_inference_steps_precise = num_inference_steps_precise

        # ---- Confidence-based head selection ----
        self.use_confidence_based_head_selection = use_confidence_based_head_selection
        self.confidence_threshold_for_head_selection = confidence_threshold_for_head_selection

        hidden_dim = confidence_hidden_dim or n_emb
        self.confidence_head = None
        if self.use_confidence_head:
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

        logger.info(
            f"FMMultiFreqConfidencePolicy initialized "
            f"(stage={training_stage}, confidence_head={use_confidence_head}, "
            f"inference_head={inference_head}, "
            f"use_confidence_based_head_selection={use_confidence_based_head_selection}, "
            f"confidence_threshold_for_head_selection={confidence_threshold_for_head_selection})"
        )

    def set_training_stage(self, stage: int):
        assert stage in (1, 2, 3), f"Invalid stage {stage}, must be 1, 2 or 3"
        self.training_stage = stage

    # ------------------------------------------------------------------
    # Override: confidence-based head selection for reactive runner
    # ------------------------------------------------------------------
    def _get_inference_head(self, obs_tokens=None):
        """
        Select 10hz or 50hz based on confidence or fixed inference_head.

        When use_confidence_based_head_selection=True and confidence_head exists:
          confidence > threshold → "50hz" (fast consistency sampling)
          confidence ≤ threshold → "10hz" (precise multi-step Euler)
        Otherwise: use fixed self.inference_head.
        """
        if not (self.use_confidence_based_head_selection and self.confidence_head is not None):
            return self.inference_head

        if obs_tokens is None:
            return self.inference_head

        # obs_tokens may be flattened (B, N*n_emb) — reshape to (B, N, n_emb)
        if len(obs_tokens.shape) == 2:
            obs_tokens_3d = obs_tokens.reshape(obs_tokens.shape[0], -1, self.n_emb)
        else:
            obs_tokens_3d = obs_tokens

        log_sigma = self._compute_confidence_logits(obs_tokens_3d)
        if log_sigma is None:
            logger.warning("Confidence returned None, falling back to default inference_head")
            return self.inference_head

        sigma = 1.0 + torch.exp(log_sigma)
        confidence_value = torch.log(sigma).mean().item()
        logger.info(
            f"FM confidence={confidence_value:.4f}, "
            f"threshold={self.confidence_threshold_for_head_selection:.4f}"
        )

        return "50hz" if confidence_value > self.confidence_threshold_for_head_selection else "10hz"

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
        log_sigma_term = torch.log(sigma)

        gamma = self.confidence_gamma
        alpha = self.confidence_alpha
        conf_loss = gamma * fm_loss_masked * sigma - alpha * log_sigma_term
        conf_loss = conf_loss.mean()

        return conf_loss, log_sigma, sigma, fm_loss_masked

    # ------------------------------------------------------------------
    # One-step FM MSE (for confidence training, uses 10hz)
    # ------------------------------------------------------------------
    def _compute_fm_one_step_mse(
        self,
        nactions_10: torch.Tensor,
        obs_tokens: torch.Tensor,
        tc_10: Optional[torch.Tensor],
        loss_mask=None,
        n_samples: int = 8,
        num_denoise_steps: int = 2,
    ):
        """
        Multi-step Euler MSE on 10hz actions (distill=0).
        Average over n_samples noise draws for stable confidence target.

        Each noise draw runs `num_denoise_steps`-step Euler ODE from t=1→0,
        matching actual inference behavior. Total forward passes per batch:
        n_samples * num_denoise_steps (default: 8 * 10 = 80).
        """
        B = nactions_10.shape[0]
        device = nactions_10.device

        per_sample_loss_all_accum = torch.zeros_like(nactions_10)

        with torch.no_grad():
            for _ in range(n_samples):
                x1 = torch.randn_like(nactions_10)

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
                        trajectory, t_tensor,
                        cond=obs_tokens, temporal_cond=tc_10,
                        freq_mode="10hz", distill_mode=0,
                    )
                    trajectory = trajectory + dt * pred_v

                x0_pred = trajectory

                per_sample_loss_all_accum += F.mse_loss(
                    x0_pred, nactions_10, reduction="none"
                )

        per_sample_loss_all = per_sample_loss_all_accum / n_samples

        per_sample_loss = reduce(per_sample_loss_all, "b t a -> b", "mean")

        # Confidence dims (first confidence_action_dims)
        conf_dims = min(
            self.confidence_action_dims, per_sample_loss_all.shape[-1]
        )
        sel_loss = per_sample_loss_all[..., :conf_dims]
        if self.normalize_confidence_loss:
            gt_sq = nactions_10[..., :conf_dims] ** 2
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
    # compute_loss: three-stage
    # ------------------------------------------------------------------
    def compute_loss(self, batch, loss_mask=None):
        # Stage 1 & 2: delegate to base policy
        if self.training_stage in (1, 2):
            return super().compute_loss(batch, loss_mask)

        # ==================== Stage 3: Confidence on one-step FM MSE ===========
        elif self.training_stage == 3:
            nobs = self.normalizer.normalize(batch["obs"])
            nactions_10 = self.normalizer["action"].normalize(batch["action"])
            B = nactions_10.shape[0]
            device = nactions_10.device

            # Temporal cond for 10hz
            tc_10 = None
            if self.rnn_lowfreq is not None and "extended_obs" in batch:
                with torch.no_grad():
                    tc_10 = self._get_temporal_cond_lowfreq(
                        batch["extended_obs"]
                    )

            # Encode obs (no grad in stage 3)
            with torch.no_grad():
                from ImplicitRDP.common.pytorch_util import dict_apply

                this_nobs = dict_apply(
                    nobs, lambda x: x.reshape(-1, *x.shape[2:])
                )
                obs_tokens = self.obs_encoder(this_nobs)
                if len(obs_tokens.shape) == 2:
                    obs_tokens = obs_tokens.reshape(
                        B, -1, *obs_tokens.shape[1:]
                    )

            # One-step FM MSE (10hz)
            (
                fm_mse,
                fm_mse_vec,
                fm_mse_vec_conf,
                fm_mse_vec_virtual,
            ) = self._compute_fm_one_step_mse(
                nactions_10, obs_tokens, tc_10, loss_mask
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
                "training_stage": 3.0,
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
                f"Invalid training_stage: {self.training_stage}. Must be 1, 2 or 3."
            )

    # ------------------------------------------------------------------
    # Inference: confidence-based
    # ------------------------------------------------------------------
    def _predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        dataset_obs_temporal_downsample_ratio: int = 1,
        extended_obs_dict: Dict[str, torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict

        obs_tokens = self._encode_obs(obs_dict)
        B = obs_tokens.shape[0]

        # Compute confidence
        confidence_value = None
        selected_freq = "10hz"
        actual_steps = num_inference_steps or self.num_inference_steps

        if self.confidence_head is not None:
            log_sigma = self._compute_confidence_logits(obs_tokens)
            if log_sigma is not None:
                sigma = 1.0 + torch.exp(log_sigma)
                confidence_value = torch.log(sigma).mean().item()

                if confidence_value > self.confidence_threshold:
                    selected_freq = "50hz"
                else:
                    selected_freq = "10hz"

        # Get temporal conds
        tc_10 = self._get_temporal_cond_lowfreq(extended_obs_dict)
        tc_50 = self._get_temporal_cond_highfreq(extended_obs_dict)

        if selected_freq == "50hz":
            # High confidence → 50hz consistency sampling (distill=1)
            nsample = self.consistency_sample(
                cond=obs_tokens,
                temporal_cond=tc_50,
                horizon=self.horizon_50hz,
                freq_mode="50hz",
                num_inference_steps=actual_steps,
            )
            action_pred = self.normalizer["action_stack"].unnormalize(nsample)
            start = self.action_start_50hz
            n_steps = self.horizon_50hz - start
            action = action_pred[:, start:start + n_steps]
        else:
            # Low confidence → 10hz + distill=0 + N-step Euler
            nsample = self.conditional_sample(
                cond=obs_tokens,
                temporal_cond=tc_10,
                horizon=self.horizon_10hz,
                freq_mode="10hz",
                distill_mode=0,
                num_inference_steps=self.num_inference_steps_precise,
            )
            action_pred = self.normalizer["action"].unnormalize(nsample)
            start = self.action_start_10hz
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        # Also compute both for logging
        nsample_10 = self.conditional_sample(
            cond=obs_tokens, temporal_cond=tc_10,
            horizon=self.horizon_10hz, freq_mode="10hz",
            distill_mode=0, num_inference_steps=actual_steps,
        )
        nsample_50 = self.consistency_sample(
            cond=obs_tokens, temporal_cond=tc_50,
            horizon=self.horizon_50hz, freq_mode="50hz",
            num_inference_steps=actual_steps,
        )
        action_pred_10hz = self.normalizer["action"].unnormalize(nsample_10)
        action_pred_50hz = self.normalizer["action_stack"].unnormalize(nsample_50)

        action_10hz = action_pred_10hz[:, self.action_start_10hz:self.action_start_10hz + self.n_action_steps]
        start_50 = self.action_start_50hz
        action_50hz = action_pred_50hz[:, start_50:start_50 + self.horizon_50hz - start_50]

        result = {
            "action": action,
            "action_pred": action_pred,
            "action_10hz": action_10hz,
            "action_50hz": action_50hz,
            "action_pred_10hz": action_pred_10hz,
            "action_pred_50hz": action_pred_50hz,
            "selected_freq": selected_freq,
        }

        if confidence_value is not None:
            result["confidence"] = confidence_value

        return result

    # ------------------------------------------------------------------
    # Optimizer: three-stage
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
        if self.training_stage in (1, 2):
            return super().get_optimizer(
                lr=lr,
                weight_decay=weight_decay,
                obs_encoder_lr=obs_encoder_lr,
                obs_encoder_weight_decay=obs_encoder_weight_decay,
                betas=betas,
                **kwargs,
            )
        elif self.training_stage == 3:
            # Only confidence head
            optim_groups = []
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
        else:
            raise ValueError(f"Invalid training_stage: {self.training_stage}")
