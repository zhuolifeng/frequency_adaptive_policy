"""
Multi-frequency Flow Matching Policy (v3: split forward, optional CD distillation).

Single FM backbone outputs 10hz (T=16) and 50hz (T=80) actions via separate forward calls.

Flag `use_distillation` controls behavior:
  - False: Stage 1 only (standard FM velocity matching on both frequencies)
  - True:  Stage 1 + Stage 2 (Consistency Distillation for 50hz)

Training:
  Stage 1: FM velocity matching, two forward calls (10hz + 50hz), distill=0
  Stage 2: 50hz official CD (distill=1)
           Teacher = frozen Stage1 model, Target = EMA model

Inference:
  10hz: multi-step Euler (distill=0)
  50hz: multi-step consistency sampling (distill=1) if distillation enabled, else multi-step Euler (distill=0)
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
from ImplicitRDP.model.diffusion.transformer_for_diffusion_multifreq import (
    TransformerForDiffusionMultiFreq,
)
from ImplicitRDP.model.force.rnn import RNN


class FMMultiFreqPolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: torch.nn.Module,
        model: TransformerForDiffusionMultiFreq,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: int = 5,
        # Flow matching time sampling
        beta_a: float = 1.5,
        beta_b: float = 1.0,
        # Architecture
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
        # Distillation control
        use_distillation: bool = False,
        cd_num_scales: int = 40,
        # Inference head selection ("10hz" or "50hz")
        inference_head: str = "10hz",
        # Training stage
        training_stage: int = 1,
        **kwargs,
    ):
        super().__init__()

        # ---- Action shape ----
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        # ---- Extended obs keys ----
        all_extended_obs_keys = (
            list(shape_meta["extended_obs"].keys())
            if "extended_obs" in shape_meta
            else []
        )
        self.extended_obs_keys = sorted(all_extended_obs_keys)

        # Determine temporal_cond_dim for each RNN
        temporal_cond_dim_low = shape_meta["extended_obs"].get(
            "right_robot_tcp_wrench", {}
        ).get("shape", [6])[-1]
        temporal_cond_dim_high = shape_meta["extended_obs"].get(
            "right_robot_tcp_wrench_stack", {}
        ).get("shape", [6])[-1]

        # ---- Two RNN encoders ----
        if use_rnn_obs_encoder:
            self.rnn_lowfreq = RNN(
                input_dim=temporal_cond_dim_low,
                hidden_dim=rnn_obs_encoder_hidden_dim,
                layer_num=rnn_obs_encoder_n_layer,
                n_emb=n_emb,
            )
            self.rnn_highfreq = RNN(
                input_dim=temporal_cond_dim_high,
                hidden_dim=rnn_obs_encoder_hidden_dim,
                layer_num=rnn_obs_encoder_n_layer,
                n_emb=n_emb,
            )
        else:
            self.rnn_lowfreq = None
            self.rnn_highfreq = None

        # ---- Core modules ----
        self.shape_meta = shape_meta
        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()

        # ---- Scalars ----
        self.action_dim = action_dim
        self.horizon = horizon
        self.horizon_10hz = horizon_10hz
        self.horizon_50hz = horizon_50hz
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.n_emb = n_emb
        self.num_inference_steps = num_inference_steps
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.action_start_10hz = action_start_10hz
        self.action_start_50hz = action_start_50hz
        self.kwargs = kwargs

        # ---- Distillation ----
        self.use_distillation = use_distillation
        self.cd_num_scales = cd_num_scales
        self.frozen_teacher = None   # set by workspace at stage transition
        self.ema_model_ref = None    # set by workspace at stage transition

        # ---- Training stage ----
        self.training_stage = training_stage

        # ---- Inference head ----
        self.inference_head = inference_head  # "10hz" or "50hz"

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

        self.progress = 0.0

        logger.info(
            f"FMMultiFreqPolicy initialized "
            f"(horizon_10hz={horizon_10hz}, horizon_50hz={horizon_50hz}, "
            f"use_distillation={use_distillation}, stage={training_stage}, "
            f"inference_head={inference_head})"
        )

    # ------------------------------------------------------------------
    # Normalizer & stage control
    # ------------------------------------------------------------------
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def update_progress(self, progress: float):
        self.progress = progress

    def set_training_stage(self, stage: int):
        assert stage in (1, 2), f"Invalid stage {stage}, must be 1 or 2"
        self.training_stage = stage

    def set_teacher_model(self, teacher):
        """Set frozen teacher model for CD (Stage 2). Called by workspace."""
        self.frozen_teacher = teacher

    def set_ema_model_ref(self, ema_model):
        """Set ema_model reference for CD Target. Called by workspace."""
        self.ema_model_ref = ema_model

    def _get_inference_head(self, obs_tokens=None):
        return self.inference_head
    # ------------------------------------------------------------------
    # Temporal condition helpers
    # ------------------------------------------------------------------
    def get_temporal_cond(
        self,
        extended_obs_dict,
        extended_obs_last_step: Optional[int] = None,
        extend_obs_pad_after_n: Optional[int] = None,
        head: str = "10hz",
    ):
        if extended_obs_dict is None:
            return None
        if head == "50hz":
            return self.get_temporal_cond_highfreq(extended_obs_dict, extended_obs_last_step, extend_obs_pad_after_n)
        else:
            return self.get_temporal_cond_lowfreq(extended_obs_dict, extended_obs_last_step, extend_obs_pad_after_n)

    def get_temporal_cond_highfreq(
        self, 
        extended_obs_dict,
        extended_obs_last_step: Optional[int] = None,
        extend_obs_pad_after_n: Optional[int] = None,
    ):
        key = "right_robot_tcp_wrench_stack"
        obs = extended_obs_dict[key]
        if extended_obs_last_step is not None:
            obs = obs[:, -extended_obs_last_step:, ...]
        if extend_obs_pad_after_n is not None:
            padding_obs = obs[:, -1:, ...].repeat(1, extend_obs_pad_after_n, 1)
            obs = torch.cat([padding_obs, obs], dim=-2)
        obs = self.normalizer[f"extended_{key}"].normalize(obs)
        return self.rnn_highfreq(obs)

    def get_temporal_cond_lowfreq(
        self, 
        extended_obs_dict,
        extended_obs_last_step: Optional[int] = None,
        extend_obs_pad_after_n: Optional[int] = None,
    ):
        key = "right_robot_tcp_wrench"
        obs = extended_obs_dict[key]
        if extended_obs_last_step is not None:
            obs = obs[:, -extended_obs_last_step:, ...]
        if extend_obs_pad_after_n is not None:
            padding_obs = obs[:, -1:, ...].repeat(1, extend_obs_pad_after_n, 1)
            obs = torch.cat([padding_obs, obs], dim=-2)
        obs = self.normalizer[f"extended_{key}"].normalize(obs)
        return self.rnn_lowfreq(obs)

    # ------------------------------------------------------------------
    # Obs tokens
    # ------------------------------------------------------------------
    def _predict_obs_tokens(
        self,
        obs_dict: Dict[str, torch.Tensor],
        return_expanded_obs_tokens=False,
    ) -> torch.Tensor:
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])
        # (B, N, n_emb)

        B = obs_tokens.shape[0]
        obs_tokens = obs_tokens.reshape(B, -1)  # (B, N*n_emb)
        if return_expanded_obs_tokens:
            obs_tokens = obs_tokens.unsqueeze(1).expand(-1, self.action_horizon, -1)

        return obs_tokens

    def predict_obs_tokens(self, *args, **kwargs):
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_obs_tokens(*args, **kwargs)

    def sample_noisy_trajectory(
        self,
        B: int = 1,
        generator=None,
        obs_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        head = self._get_inference_head(obs_tokens)
        horizon = self.horizon_50hz if head == "50hz" else self.horizon_10hz
        noisy_trajectory = torch.randn(
            size=(B, horizon, self.action_dim),
            dtype=torch.float32,
            device=self.device,
            generator=generator)

        noisy_trajectory = noisy_trajectory.reshape(B, -1)

        return noisy_trajectory, head


    def _predict_from_obs_tokens_and_noisy_trajectory(
        self,
        obs_tokens_and_noisy_trajectory_list: List[torch.Tensor],
        extended_obs_dict: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
        extended_obs_last_step_list: List[int],
        dataset_obs_temporal_downsample_ratio: int,
        extend_obs_pad_after: bool = False,
        head: str = "10hz"
    ) -> List[Dict[str, torch.Tensor]]:
        obs_tokens_and_noisy_trajectory = torch.cat(obs_tokens_and_noisy_trajectory_list, dim=0)
        B = obs_tokens_and_noisy_trajectory.shape[0]

        head = head
        logger.info(f"FM MultiFreq reactive inference: selected head = {head}")
        target_horizon = self.horizon_50hz if head == "50hz" else self.horizon_10hz

        obs_tokens = obs_tokens_and_noisy_trajectory[:, :- target_horizon * self.action_dim]
        noisy_trajectory = obs_tokens_and_noisy_trajectory[:, - target_horizon * self.action_dim:]
        obs_tokens = obs_tokens.to(self.device).reshape(B, -1, self.n_emb)
        noisy_trajectory = noisy_trajectory.to(self.device).reshape(B, target_horizon, self.action_dim)

        max_extended_obs_last_step = max(extended_obs_last_step_list)
        assert extended_obs_dict is not None, "extended_obs is required for RNNObsEncoder"
        if extend_obs_pad_after:
            extend_obs_pad_after_n = target_horizon - max_extended_obs_last_step
        else:
            extend_obs_pad_after_n = None

        if isinstance(extended_obs_dict, list):
            temporal_cond = []
            for i in range(len(extended_obs_dict)):
                temporal_cond.append(
                    self.get_temporal_cond(extended_obs_dict[i], max_extended_obs_last_step, extend_obs_pad_after_n=extend_obs_pad_after_n, head=head)
                )
        else:
            temporal_cond = self.get_temporal_cond(extended_obs_dict, max_extended_obs_last_step, extend_obs_pad_after_n=extend_obs_pad_after_n, head=head)

        temporal_cond_list = []
        for i in range(B):
            if isinstance(temporal_cond, list):
                if extended_obs_last_step_list[i] < max_extended_obs_last_step:
                    padding_obs = temporal_cond[i][:, -1:, :].repeat(1, max_extended_obs_last_step -
                                                                    extended_obs_last_step_list[i], 1)
                    temporal_cond_list.append(
                        torch.cat([temporal_cond[i][:, -extended_obs_last_step_list[i]:, :], padding_obs], dim=-2))
                else:
                    temporal_cond_list.append(temporal_cond[i])
            else:
                if extended_obs_last_step_list[i] < max_extended_obs_last_step:
                    padding_obs = temporal_cond[:, -1:, :].repeat(1, max_extended_obs_last_step -
                                                                    extended_obs_last_step_list[i], 1)
                    temporal_cond_list.append(
                        torch.cat([temporal_cond[:, -extended_obs_last_step_list[i]:, :], padding_obs], dim=-2))
                else:
                    temporal_cond_list.append(temporal_cond)
        temporal_cond = torch.cat(temporal_cond_list, dim=0)
        noisy_trajectory = noisy_trajectory[:, :temporal_cond.shape[1], :]

        if head == "50hz":
            normalizer_key = "action_stack"
            if self.use_distillation:
                nsample = self.consistency_sample(
                    cond=obs_tokens, temporal_cond=temporal_cond, 
                    freq_mode="50hz",
                    noisy_trajectory=noisy_trajectory,
                )
            else:
                nsample = self.conditional_sample(
                    cond=obs_tokens, temporal_cond=temporal_cond, 
                    freq_mode="50hz", distill_mode=0,
                    noisy_trajectory=noisy_trajectory,
                )
        else:  # "10hz"
            normalizer_key = "action"
            nsample = self.conditional_sample(
                cond=obs_tokens, temporal_cond=temporal_cond,
                freq_mode="10hz", distill_mode=0,
                noisy_trajectory=noisy_trajectory,
            )
        # unnormalize prediction
        assert nsample.shape == (B, temporal_cond.shape[1], self.action_dim)
        action_pred = self.normalizer[normalizer_key].unnormalize(nsample)

        if head == "50hz":
            start = self.action_start_50hz
            end = self.horizon_50hz
        else:
            start = self.action_start_10hz
            end = start + self.n_action_steps
        action = action_pred[:, start:end]

        results = []
        for i in range(B):
            if extend_obs_pad_after:
                results.append({
                    'action': action[i: i + 1],
                    'action_pred': action_pred[i: i + 1],
                })
            else:
                valid_len = max(0, min(extended_obs_last_step_list[i] - start, end - start))
                results.append({
                    'action': action[i: i + 1, :valid_len, :],
                    'action_pred': action_pred[i: i + 1, :extended_obs_last_step_list[i], :],
                })

        return results

    def predict_from_obs_tokens_and_noisy_trajectory(
                                self,
                                *args,
                                **kwargs,
                            ) -> List[Dict[str, torch.Tensor]]:
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_from_obs_tokens_and_noisy_trajectory(
                *args,
                **kwargs
            )

    # ------------------------------------------------------------------
    # FM Euler ODE sampler (split, supports freq_mode & distill_mode)
    # ------------------------------------------------------------------
    def conditional_sample(
        self,
        cond=None,
        temporal_cond=None,
        freq_mode: str = "10hz",
        distill_mode: int = 0,
        generator=None,
        num_inference_steps: Optional[int] = None,
        noisy_trajectory=None,
        **kwargs,
    ):
        """Euler ODE solver for Rectified Flow: t=1 (noise) -> t=0 (data)."""
        model = self.model
        steps = num_inference_steps if num_inference_steps is not None else self.num_inference_steps

        if noisy_trajectory is None:
            trajectory = torch.randn(
                size=(cond.shape[0], temporal_cond.shape[1], self.action_dim),
                dtype=cond.dtype,
                device=cond.device,
                generator=generator,
            )
        else:
            trajectory = noisy_trajectory

        B = trajectory.shape[0]

        t_schedule = torch.linspace(
            1.0, 0.0, steps + 1, device=trajectory.device
        )
        for i in range(steps):
            t_cur = t_schedule[i]
            dt = t_schedule[i + 1] - t_schedule[i]

            t_tensor = torch.full(
                (B,), t_cur * 1000, device=trajectory.device
            ).long()

            model_output = model(
                trajectory,
                t_tensor,
                cond=cond,
                temporal_cond=temporal_cond,
                freq_mode=freq_mode,
                distill_mode=distill_mode,
            )

            trajectory = trajectory + dt * model_output

        return trajectory

    # ------------------------------------------------------------------
    # CD consistency mapping: x_0 = x_t - t * v(x_t, t)
    # ------------------------------------------------------------------
    def consistency_sample(
        self,
        cond=None,
        temporal_cond=None,
        freq_mode: str = "50hz",
        generator=None,
        num_inference_steps: Optional[int] = None,
        noisy_trajectory=None,
        **kwargs,
    ):
        """
        Multi-step consistency sampling for CD-trained model.

        Single step (num_inference_steps=1):
            x_0 = x_1 - 1.0 * v(x_1, 1.0)

        Multi-step (num_inference_steps=N, N>1):
            For each step i with time t_i (from t=1 toward t=0):
              1. Denoise: x_0 = x_{t_i} - t_i * v(x_{t_i}, t_i)
              2. If not last step, re-noise to next time t_{i+1}:
                 x_{t_{i+1}} = t_{i+1} * eps + (1 - t_{i+1}) * x_0
            This iterative denoise-then-renoise procedure progressively
            refines the sample, improving quality over single-step.
        """
        model = self.model
        steps = num_inference_steps if num_inference_steps is not None else 1

        # Start from pure noise (t=1)
        if noisy_trajectory is None:
            x_t = torch.randn(
                size=(cond.shape[0], temporal_cond.shape[1], self.action_dim),
                dtype=cond.dtype,
                device=cond.device,
                generator=generator,
            )
        else:
            x_t = noisy_trajectory

        B = x_t.shape[0]

        # Time schedule: evenly spaced from 1.0 toward 0.0
        # e.g. steps=1: [1.0], steps=2: [1.0, 0.5], steps=3: [1.0, 0.667, 0.333]
        t_schedule = torch.linspace(1.0, 1.0 / steps, steps, device=x_t.device)

        for i in range(steps):
            t = t_schedule[i]

            t_tensor = torch.full(
                (B,), t.item() * 1000, device=x_t.device
            ).long()

            v = model(
                x_t,
                t_tensor,
                cond=cond,
                temporal_cond=temporal_cond,
                freq_mode=freq_mode,
                distill_mode=1,
            )

            # Consistency mapping: x_0 = x_t - t * v
            x_0 = x_t - t * v

            # If not the last step, re-noise to the next time level
            if i < steps - 1:
                t_next = t_schedule[i + 1]
                eps = torch.randn_like(x_0)
                x_t = t_next * eps + (1 - t_next) * x_0
            else:
                x_t = x_0

        return x_t

    # ------------------------------------------------------------------
    # Training: model_forward (Stage 1 - standard FM)
    # ------------------------------------------------------------------
    def model_forward(self, batch):
        """Rectified Flow velocity matching, two separate forward calls."""
        assert "valid_mask" not in batch

        nobs = self.normalizer.normalize(batch["obs"])
        nactions_10 = self.normalizer["action"].normalize(batch["action"])
        nactions_50 = self.normalizer["action_stack"].normalize(batch["action_stack"])
        B = nactions_10.shape[0]

        if self.rnn_lowfreq is not None and self.rnn_highfreq is not None:
            assert 'extended_obs' in batch, "extended_obs is required for RNNObsEncoder"
            temporal_cond_10hz = self.get_temporal_cond(batch["extended_obs"], head="10hz")
            temporal_cond_50hz = self.get_temporal_cond(batch["extended_obs"], head="50hz")
        else:
            temporal_cond_10hz = None
            temporal_cond_50hz = None

        # process input
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])

        # Sample t ~ Beta (shared for both frequencies)
        t = torch.distributions.Beta(self.beta_a, self.beta_b).sample(
            (B,)
        ).to(nactions_10.device)
        t = t * 0.999 + 0.001
        timesteps = (t * 1000).long()
        t_exp = t.view(B, 1, 1)

        # Forward 1: 10hz
        noise_10 = torch.randn(nactions_10.shape, device=nactions_10.device)
        x_t_10 = t_exp * noise_10 + (1 - t_exp) * nactions_10
        pred_v_10 = self.model(
            x_t_10, timesteps, cond=obs_tokens, temporal_cond=temporal_cond_10hz,
            freq_mode="10hz", distill_mode=0,
        )
        target_10 = noise_10 - nactions_10

        # Forward 2: 50hz
        noise_50 = torch.randn(nactions_50.shape, device=nactions_50.device)
        x_t_50 = t_exp * noise_50 + (1 - t_exp) * nactions_50
        pred_v_50 = self.model(
            x_t_50, timesteps, cond=obs_tokens, temporal_cond=temporal_cond_50hz,
            freq_mode="50hz", distill_mode=0,
        )
        target_50 = noise_50 - nactions_50

        return pred_v_10, target_10, pred_v_50, target_50

    # ------------------------------------------------------------------
    # Training: CD loss (Stage 2 - 50hz consistency distillation)
    # ------------------------------------------------------------------
    def _compute_cd_loss(self, batch):
        """
        Three model roles:
          frozen_teacher (distill=0): one ODE step on frozen Stage1 model
          model/Student (distill=1): consistency function, has gradient
          ema_model/Target (distill=1): stable consistency target, no gradient
        """
        assert self.frozen_teacher is not None, (
            "frozen_teacher must be set before Stage 2. "
            "Call set_teacher_model() in workspace."
        )
        assert self.ema_model_ref is not None, (
            "ema_model_ref must be set before Stage 2. "
            "Call set_ema_model_ref() in workspace."
        )
        nobs = self.normalizer.normalize(batch["obs"])
        B = next(iter(nobs.values())).shape[0]

        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])

        nactions_50 = self.normalizer["action_stack"].normalize(batch["action_stack"])
        device = nactions_50.device

        if self.rnn_highfreq is not None:
            assert 'extended_obs' in batch, "extended_obs is required for RNNObsEncoder"
            temporal_cond_50hz = self.get_temporal_cond(batch["extended_obs"], head="50hz")

        noise = torch.randn(nactions_50.shape, device=nactions_50.device)

        # Sample adjacent time pair (t_n, t_{n+1}), t_n > t_{n+1}
        N = self.cd_num_scales
        indices = torch.randint(0, N - 1, (B,), device=device)
        t_n = 1.0 - indices.float() / N         # closer to noise
        t_n1 = 1.0 - (indices + 1).float() / N  # closer to data
        t_n_exp = t_n.view(B, 1, 1)
        t_n1_exp = t_n1.view(B, 1, 1)

        # Create x_{t_n} from real data
        x_tn = t_n_exp * noise + (1 - t_n_exp) * nactions_50
        timesteps_n = (t_n * 1000).long()
        timesteps_n1 = (t_n1 * 1000).long()

        # Teacher: one ODE step (frozen, distill=0)
        with torch.no_grad():
            v_teacher = self.frozen_teacher(
                x_tn, timesteps_n, cond=obs_tokens, temporal_cond=temporal_cond_50hz,
                freq_mode="50hz", distill_mode=0,
            )
            dt = (t_n1 - t_n).view(B, 1, 1)  # negative
            x_tn1 = x_tn + dt * v_teacher

        # Student: consistency function (model, distill=1, has gradient)
        v_student = self.model(
            x_tn, timesteps_n, cond=obs_tokens, temporal_cond=temporal_cond_50hz,
            freq_mode="50hz", distill_mode=1,
        )
        f_student = x_tn - t_n_exp * v_student  # f(x_t, t) = x_t - t*v

        # Target: consistency function (ema_model, distill=1, no gradient)
        with torch.no_grad():
            v_target = self.ema_model_ref(
                x_tn1, timesteps_n1, cond=obs_tokens, temporal_cond=temporal_cond_50hz,
                freq_mode="50hz", distill_mode=1,
            )
            f_target = x_tn1 - t_n1_exp * v_target

        loss_cd = F.mse_loss(f_student, f_target.detach())

        return loss_cd

    # ------------------------------------------------------------------
    # compute_loss
    # ------------------------------------------------------------------
    def compute_loss(self, batch, loss_mask=None):
        # ==================== Stage 1: FM velocity matching ====================
        if self.training_stage == 1:
            pred_v_10, target_10, pred_v_50, target_50 = (
                self.model_forward(batch)
            )
            B = pred_v_10.shape[0]
            device = pred_v_10.device

            if loss_mask is None:
                loss_mask = torch.ones(B, dtype=torch.bool, device=device)

            # 10hz loss
            loss_10_all = F.mse_loss(pred_v_10, target_10, reduction="none")
            loss_10_per = reduce(loss_10_all, "b ... -> b (...)", "mean")
            loss_10_per = loss_10_per[loss_mask].mean(dim=1)
            loss_10 = loss_10_per.sum() / (loss_mask.sum() + 1e-8)

            # 50hz loss
            loss_50_all = F.mse_loss(pred_v_50, target_50, reduction="none")
            loss_50_per = reduce(loss_50_all, "b ... -> b (...)", "mean")
            loss_50_per = loss_50_per[loss_mask].mean(dim=1)
            loss_50 = loss_50_per.sum() / (loss_mask.sum() + 1e-8)

            fm_loss = loss_10 + loss_50

            # Front 6 dims (for logging)
            front6_10 = reduce(
                loss_10_all[..., :6], "b ... -> b (...)", "mean"
            )
            front6_10 = front6_10[loss_mask].mean(dim=1).sum() / (
                loss_mask.sum() + 1e-8
            )

            loss_dict = {
                "fm_velocity_loss": fm_loss.item(),
                "fm_velocity_loss_10hz": loss_10.item(),
                "fm_velocity_loss_50hz": loss_50.item(),
                "fm_velocity_loss_front6_10hz": front6_10.item(),
                "total_loss": fm_loss.item(),
                "training_stage": 1.0,
            }

            return fm_loss, loss_dict

        # ==================== Stage 2: 50hz CD ====================
        elif self.training_stage == 2:
            total_loss = self._compute_cd_loss(batch)

            loss_dict = {
                "cd_loss_50hz": total_loss.item(),
                "total_loss": total_loss.item(),
                "training_stage": 2.0,
            }

            return total_loss, loss_dict

        else:
            raise ValueError(
                f"Invalid training_stage: {self.training_stage}. Must be 1 or 2."
            )

    def forward(self, batch, loss_mask=None):
        return self.compute_loss(batch, loss_mask)

    # ------------------------------------------------------------------
    # Inference
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

        if self.rnn_lowfreq is not None or self.rnn_highfreq is not None:
            assert (
                extended_obs_dict is not None
            ), "extended_obs is required for RNN temporal condition in FMMultiFreqPolicy"

        temporal_cond_10hz = self.get_temporal_cond(extended_obs_dict, head="10hz")
        temporal_cond_50hz = self.get_temporal_cond(extended_obs_dict, head="50hz")

        steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps

        # 10hz: multi-step Euler (distill=0)
        nsample_10 = self.conditional_sample(
            cond=obs_tokens,
            temporal_cond=temporal_cond_10hz,
            freq_mode="10hz",
            distill_mode=0,
            num_inference_steps=steps,
        )

        # 50hz: depends on use_distillation
        if self.use_distillation:
            # Multi-step consistency sampling (distill=1)
            nsample_50 = self.consistency_sample(
                cond=obs_tokens,
                temporal_cond=temporal_cond_50hz,
                freq_mode="50hz",
                num_inference_steps=steps,
            )
        else:
            # multi-step Euler (distill=0)
            nsample_50 = self.conditional_sample(
                cond=obs_tokens,
                temporal_cond=temporal_cond_50hz,
                freq_mode="50hz",
                distill_mode=0,
                num_inference_steps=steps,
            )

        # Unnormalize
        assert nsample_10.shape == (B, temporal_cond_10hz.shape[1], self.action_dim)
        assert nsample_50.shape == (B, temporal_cond_50hz.shape[1], self.action_dim)
        action_pred_10hz = self.normalizer["action"].unnormalize(nsample_10)
        action_pred_50hz = self.normalizer["action_stack"].unnormalize(nsample_50)

        # Extract action windows
        start_10 = self.action_start_10hz
        end_10 = start_10 + self.n_action_steps
        action_10hz = action_pred_10hz[:, start_10:end_10]

        start_50 = self.action_start_50hz
        end_50 = self.horizon_50hz
        action_50hz = action_pred_50hz[:, start_50:end_50]

        return {
            "action": action_10hz,           # default to 10hz
            "action_pred": action_pred_10hz,
            "action_10hz": action_10hz,
            "action_50hz": action_50hz,
            "action_pred_10hz": action_pred_10hz,
            "action_pred_50hz": action_pred_50hz,
        }

    def predict_action(self,
                       *args,
                       **kwargs,
                       ) -> Dict[str, torch.Tensor]:
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_action(*args, **kwargs)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def get_optimizer(
        self,
        lr: float,
        weight_decay: float,
        obs_encoder_lr: Optional[float] = None,
        obs_encoder_weight_decay: Optional[float] = None,
        betas: Tuple[float, float] = (0.9, 0.95)
    ) -> torch.optim.Optimizer:
        optim_groups = []

        if self.training_stage in (1, 2):
            # FM model
            optim_groups.extend(
                self.model.get_optim_groups(weight_decay=weight_decay)
            )
            # RNN encoders
            if self.rnn_lowfreq is not None:
                optim_groups += self.rnn_lowfreq.get_optim_groups(
                    weight_decay=weight_decay
                )
            if self.rnn_highfreq is not None:
                optim_groups += self.rnn_highfreq.get_optim_groups(
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

        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        return optimizer
