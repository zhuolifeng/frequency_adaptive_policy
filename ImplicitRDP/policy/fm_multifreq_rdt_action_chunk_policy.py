"""
RDT-style multi-frequency policy with action-chunk parameterization.

The backbone predicts x0/action chunks. Velocity is derived from x0_pred for
flow-matching losses and ODE sampling:
  x_t = (1 - t) * noise + t * x0, t=0 is full noise, t=1 is clean
  v_pred = (x0_pred - x_t) / (1 - t)
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import reduce
from loguru import logger

from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.model.common.normalizer import LinearNormalizer
from ImplicitRDP.model.force.rnn import RNN
from ImplicitRDP.model.diffusion.transformer_for_rdt_action_chunk_multifreq import (
    TransformerForRDTActionChunkMultiFreq,
)
from ImplicitRDP.policy.base_image_policy import BaseImagePolicy


class FMMultiFreqRDTActionChunkPolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: torch.nn.Module,
        model: TransformerForRDTActionChunkMultiFreq,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: int = 5,
        beta_a: float = 1.5,
        beta_b: float = 1.0,
        time_sampler: str = "logit_normal",
        P_mean: float = -0.8,
        P_std: float = 0.8,
        t_eps: float = 0.05,
        noise_scale: float = 1.0,
        loss_type: str = "velocity",
        n_emb: int = 768,
        use_rnn_obs_encoder: bool = True,
        rnn_obs_encoder_n_layer: int = 1,
        rnn_obs_encoder_hidden_dim: int = 512,
        use_amp_for_inference: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        horizon_10hz: int = 16,
        horizon_50hz: int = 80,
        action_start_10hz: int = 1,
        action_start_50hz: int = 13,
        teacher_num_inference_steps: int = 5,
        distill_gt_weight: float = 0.1,
        training_stage: int = 1,
        inference_head: str = "50hz",
        inference_t_schedule: str = "linear",
        inference_noise_min: float = 0.2,
        inference_t_schedule_power: float = 2.0,
        inference_t_min: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        all_extended_obs_keys = (
            list(shape_meta["extended_obs"].keys())
            if "extended_obs" in shape_meta
            else []
        )
        self.extended_obs_keys = sorted(all_extended_obs_keys)

        temporal_cond_dim_low = shape_meta["extended_obs"].get(
            "right_robot_tcp_wrench", {}
        ).get("shape", [6])[-1]
        temporal_cond_dim_high = shape_meta["extended_obs"].get(
            "right_robot_tcp_wrench_stack", {}
        ).get("shape", [6])[-1]

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

        self.shape_meta = shape_meta
        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()

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
        if time_sampler not in ("logit_normal", "beta"):
            raise ValueError(
                f"Unsupported time_sampler={time_sampler}. "
                "Expected 'logit_normal' or 'beta'."
            )
        self.time_sampler = time_sampler
        self.P_mean = P_mean
        self.P_std = P_std
        self.t_eps = t_eps
        self.noise_scale = noise_scale
        if loss_type not in ("velocity", "x0"):
            raise ValueError(
                f"Unsupported loss_type={loss_type}. Expected 'velocity' or 'x0'."
            )
        self.loss_type = loss_type
        self.action_start_10hz = action_start_10hz
        self.action_start_50hz = action_start_50hz
        self.teacher_num_inference_steps = teacher_num_inference_steps
        self.distill_gt_weight = distill_gt_weight
        self.training_stage = training_stage
        self.inference_head = inference_head
        if inference_t_min is not None:
            inference_noise_min = inference_t_min
        if inference_t_schedule not in (
            "linear",
            "trailing",
            "power",
            "nonlinear",
            "geometric",
        ):
            raise ValueError(
                f"Unsupported inference_t_schedule={inference_t_schedule}. "
                "Expected 'linear', 'trailing', 'power', 'nonlinear', or 'geometric'."
            )
        if not 0.0 < inference_noise_min < 1.0:
            raise ValueError(
                f"inference_noise_min must be in (0, 1), got {inference_noise_min}."
            )
        if inference_t_schedule_power <= 0.0:
            raise ValueError(
                "inference_t_schedule_power must be positive, "
                f"got {inference_t_schedule_power}."
            )
        self.inference_t_schedule = inference_t_schedule
        self.inference_noise_min = inference_noise_min
        self.inference_t_schedule_power = inference_t_schedule_power
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

        self.teacher_model = None
        self.progress = 0.0

        logger.info(
            "FMMultiFreqRDTActionChunkPolicy initialized "
            f"(stage={training_stage}, inference_head={inference_head}, "
            f"time_sampler={time_sampler}, t_eps={t_eps}, "
            f"noise_scale={noise_scale}, loss_type={loss_type}, "
            f"teacher_num_inference_steps={teacher_num_inference_steps}, "
            f"inference_t_schedule={inference_t_schedule}, "
            f"inference_noise_min={inference_noise_min}, "
            f"inference_t_schedule_power={inference_t_schedule_power}, "
        )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def reset(self):
        pass

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def update_progress(self, progress: float):
        self.progress = progress

    def set_training_stage(self, stage: int):
        assert stage in (1, 2), f"Invalid stage {stage}, must be 1 or 2"
        self.training_stage = stage

    def set_teacher_model(self, teacher: nn.Module):
        self.__dict__["teacher_model"] = teacher
        if self.teacher_model is not None:
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()

    def freeze_condition_modules(self):
        if self.obs_encoder is not None:
            self.obs_encoder.requires_grad_(False)
            self.obs_encoder.eval()
        if self.rnn_lowfreq is not None:
            self.rnn_lowfreq.requires_grad_(False)
            self.rnn_lowfreq.eval()
        if self.rnn_highfreq is not None:
            self.rnn_highfreq.requires_grad_(False)
            self.rnn_highfreq.eval()

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
            return self.get_temporal_cond_highfreq(
                extended_obs_dict, extended_obs_last_step, extend_obs_pad_after_n
            )
        return self.get_temporal_cond_lowfreq(
            extended_obs_dict, extended_obs_last_step, extend_obs_pad_after_n
        )

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
            padding_obs = obs[:, -1:, :].repeat(1, extend_obs_pad_after_n, 1)
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
            padding_obs = obs[:, -1:, :].repeat(1, extend_obs_pad_after_n, 1)
            obs = torch.cat([padding_obs, obs], dim=-2)
        obs = self.normalizer[f"extended_{key}"].normalize(obs)
        return self.rnn_lowfreq(obs)

    # ------------------------------------------------------------------
    # Obs tokens
    # ------------------------------------------------------------------
    def _predict_obs_tokens(
        self,
        obs_dict: Dict[str, torch.Tensor],
        return_expanded_obs_tokens: bool = False,
    ) -> torch.Tensor:
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])
        obs_tokens = obs_tokens.reshape(B, -1)
        if return_expanded_obs_tokens:
            obs_tokens = obs_tokens.unsqueeze(1).expand(-1, self.horizon_50hz, -1)
        return obs_tokens

    def predict_obs_tokens(self, *args, **kwargs):
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_obs_tokens(*args, **kwargs)

    def _get_inference_head(self, obs_tokens=None):
        return self.inference_head

    def _timestep_to_t(self, timestep: Union[torch.Tensor, float, int], device: torch.device):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=device)
        timestep = timestep.to(device=device, dtype=torch.float32)
        if timestep.ndim == 0:
            timestep = timestep[None]
        return torch.clamp(timestep / 1000.0, min=0.0, max=1.0)

    def _velocity_from_x0(
        self,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
    ):
        t = self._timestep_to_t(timestep, x_t.device).view(-1, 1, 1)
        one_minus_t = torch.clamp(1.0 - t, min=self.t_eps)
        return (x0_pred - x_t) / one_minus_t

    def _velocity_loss_from_x0(
        self,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        x0_target: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
    ):
        v_pred = self._velocity_from_x0(x_t, x0_pred, timestep)
        v_target = self._velocity_from_x0(x_t, x0_target, timestep)
        return F.mse_loss(v_pred, v_target)

    def _action_loss_from_x0(
        self,
        x0_pred: torch.Tensor,
        x0_target: torch.Tensor,
    ):
        return F.mse_loss(x0_pred, x0_target)

    def _loss_from_x0(
        self,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        x0_target: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
    ):
        if self.loss_type == "velocity":
            return self._velocity_loss_from_x0(x_t, x0_pred, x0_target, timestep)
        return self._action_loss_from_x0(x0_pred, x0_target)

    def _predict_x0(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        cond: torch.Tensor,
        temporal_cond: Optional[torch.Tensor],
        freq_mode: str,
    ):
        return model(
            x_t,
            timestep,
            cond=cond,
            temporal_cond=temporal_cond,
            freq_mode=freq_mode,
        )

    def _make_inference_t_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        if num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps must be >= 1, got {num_inference_steps}."
            )

        if self.inference_t_schedule == "linear":
            return torch.linspace(
                0.0, 1.0, num_inference_steps + 1, device=device
            )

        if self.inference_t_schedule == "trailing":
            if num_inference_steps == 1:
                return torch.tensor([0.0, 1.0], device=device)
            remaining_noise = torch.linspace(
                1.0,
                self.inference_noise_min,
                num_inference_steps,
                device=device,
            )
            clean_t = 1.0 - remaining_noise
            return torch.cat([clean_t, clean_t.new_ones(1)], dim=0)

        if self.inference_t_schedule == "geometric":
            if num_inference_steps == 1:
                return torch.tensor([0.0, 1.0], device=device)
            step_idx = torch.arange(
                num_inference_steps,
                device=device,
                dtype=torch.float32,
            )
            ratio = self.inference_noise_min ** (1.0 / (num_inference_steps - 1))
            remaining_noise = ratio ** step_idx
            clean_t = 1.0 - remaining_noise
            return torch.cat([clean_t, clean_t.new_ones(1)], dim=0)

        ramp = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device)
        t_schedule = ramp.pow(self.inference_t_schedule_power)
        t_schedule[0] = 0.0
        t_schedule[-1] = 1.0
        return t_schedule

    def sample_noisy_trajectory(
        self,
        B: int = 1,
        generator=None,
        obs_tokens: Optional[torch.Tensor] = None,
    ):
        head = self._get_inference_head(obs_tokens)
        horizon = self.horizon_50hz if head == "50hz" else self.horizon_10hz
        noisy_trajectory = torch.randn(
            size=(B, horizon, self.action_dim),
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        ) * self.noise_scale
        noisy_trajectory = noisy_trajectory.reshape(B, -1)
        
        return noisy_trajectory

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    def _prepare_condition_inputs(
        self,
        lang_cond: Optional[torch.Tensor] = None,
        lang_attn_mask: Optional[torch.Tensor] = None,
        img_cond: Optional[torch.Tensor] = None,
        state_cond: Optional[torch.Tensor] = None,
    ):
        return {
            "lang_c": lang_cond,
            "lang_mask": lang_attn_mask,
            "img_c": img_cond,
            "state_c": state_cond,
        }

    def _solve_ode(
        self,
        model: nn.Module,
        noisy_action: torch.Tensor,
        cond: torch.Tensor,
        temporal_cond: Optional[torch.Tensor],
        freq_mode: str,
        num_inference_steps: int,
    ):
        if noisy_action is None:
            raise ValueError("noisy_action must be provided for ODE solve")
        trajectory = noisy_action
        B = trajectory.shape[0]
        t_schedule = self._make_inference_t_schedule(
            num_inference_steps,
            trajectory.device,
        )
        for i in range(num_inference_steps):
            t_cur = t_schedule[i]
            t_next = t_schedule[i + 1]
            dt = t_schedule[i + 1] - t_schedule[i]
            t_tensor = torch.full(
                (B,),
                float(t_cur.item()) * 1000.0,
                device=trajectory.device,
                dtype=torch.float32,
            )
            x0_pred = self._predict_x0(
                model, trajectory, t_tensor, cond, temporal_cond, freq_mode
            )
            velocity = self._velocity_from_x0(trajectory, x0_pred, t_tensor)
            trajectory = trajectory + dt * velocity
        return trajectory

    def _one_step_action_chunk(
        self,
        model: nn.Module,
        noisy_action: torch.Tensor,
        cond: torch.Tensor,
        temporal_cond: Optional[torch.Tensor],
    ):
        B = noisy_action.shape[0]
        t_tensor = torch.full(
            (B,), 0.0, device=noisy_action.device, dtype=torch.float32
        )
        x0_pred = self._predict_x0(
            model, noisy_action, t_tensor, cond, temporal_cond, "50hz"
        )
        return x0_pred, noisy_action - x0_pred

    def _predict_from_obs_tokens_and_noisy_trajectory(
        self,
        obs_tokens_and_noisy_trajectory_list: List[torch.Tensor],
        extended_obs_dict: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
        extended_obs_last_step_list: List[int],
        dataset_obs_temporal_downsample_ratio: int,
        extend_obs_pad_after: bool = False,
        head: Optional[str] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        head = self._get_inference_head() if head is None else head
        obs_tokens_and_noisy_trajectory = torch.cat(obs_tokens_and_noisy_trajectory_list, dim=0)
        B = obs_tokens_and_noisy_trajectory.shape[0]
        target_horizon = self.horizon_50hz if head == "50hz" else self.horizon_10hz
        obs_tokens = obs_tokens_and_noisy_trajectory[:, :-target_horizon * self.action_dim]
        noisy_trajectory = obs_tokens_and_noisy_trajectory[:, -target_horizon * self.action_dim:]
        obs_tokens = obs_tokens.to(self.device).reshape(B, -1, self.n_emb)
        noisy_trajectory = noisy_trajectory.to(self.device).reshape(B, target_horizon, self.action_dim)

        max_extended_obs_last_step = max(extended_obs_last_step_list)
        assert extended_obs_dict is not None, "extended_obs is required for RNNObsEncoder"
        if extend_obs_pad_after:
            extend_obs_pad_after_n = target_horizon - max_extended_obs_last_step
        else:
            extend_obs_pad_after_n = None

        if isinstance(extended_obs_dict, list):
            temporal_cond = [
                self.get_temporal_cond(
                    extended_obs_dict[i],
                    max_extended_obs_last_step,
                    extend_obs_pad_after_n=extend_obs_pad_after_n,
                    head=head,
                )
                for i in range(len(extended_obs_dict))
            ]
        else:
            temporal_cond = self.get_temporal_cond(
                extended_obs_dict,
                max_extended_obs_last_step,
                extend_obs_pad_after_n=extend_obs_pad_after_n,
                head=head,
            )

        temporal_cond_list = []
        for i in range(B):
            if isinstance(temporal_cond, list):
                this_temporal = temporal_cond[i]
            else:
                this_temporal = temporal_cond
            if extended_obs_last_step_list[i] < max_extended_obs_last_step:
                padding_obs = this_temporal[:, -1:, :].repeat(
                    1, max_extended_obs_last_step - extended_obs_last_step_list[i], 1
                )
                this_temporal = torch.cat(
                    [this_temporal[:, -extended_obs_last_step_list[i]:, :], padding_obs],
                    dim=-2,
                )
            temporal_cond_list.append(this_temporal)
        temporal_cond = torch.cat(temporal_cond_list, dim=0)
        noisy_trajectory = noisy_trajectory[:, : temporal_cond.shape[1], :]

        if head == "50hz":
            normalizer_key = "action_stack"
            nsample = self._predict_x0(
                self.model,
                noisy_trajectory,
                torch.full(
                    (B,),
                    0.0,
                    device=noisy_trajectory.device,
                    dtype=torch.float32,
                ),
                obs_tokens,
                temporal_cond,
                "50hz",
            )
        else:
            normalizer_key = "action"
            nsample = self._solve_ode(
                self.model,
                noisy_trajectory,
                obs_tokens,
                temporal_cond,
                "10hz",
                self.num_inference_steps,
            )

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
                results.append({"action": action[i: i + 1], "action_pred": action_pred[i: i + 1]})
            else:
                valid_len = max(0, min(extended_obs_last_step_list[i] - start, end - start))
                results.append(
                    {
                        "action": action[i: i + 1, :valid_len, :],
                        "action_pred": action_pred[i: i + 1, :extended_obs_last_step_list[i], :],
                    }
                )
        return results

    def predict_from_obs_tokens_and_noisy_trajectory(
        self,
        *args,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_from_obs_tokens_and_noisy_trajectory(*args, **kwargs)

    # ------------------------------------------------------------------
    # Training losses
    # ------------------------------------------------------------------
    def _sample_timesteps(self, batch_size: int, device: torch.device):
        if self.time_sampler == "logit_normal":
            t = torch.sigmoid(
                torch.randn(batch_size, device=device) * self.P_std + self.P_mean
            )
        else:
            noise_t = torch.distributions.Beta(self.beta_a, self.beta_b).sample((batch_size,)).to(device)
            t = 1.0 - noise_t
        return t.clamp(min=0.0, max=1.0 - self.t_eps)

    def model_forward(self, batch):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions_10 = self.normalizer["action"].normalize(batch["action"])
        nactions_50 = self.normalizer["action_stack"].normalize(batch["action_stack"])
        B = nactions_10.shape[0]

        if self.rnn_lowfreq is not None and self.rnn_highfreq is not None:
            assert "extended_obs" in batch, "extended_obs is required for RNNObsEncoder"
            temporal_cond_10hz = self.get_temporal_cond(batch["extended_obs"], head="10hz")
            temporal_cond_50hz = self.get_temporal_cond(batch["extended_obs"], head="50hz")
        else:
            temporal_cond_10hz = None
            temporal_cond_50hz = None

        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])

        t = self._sample_timesteps(B, nactions_10.device)
        timesteps = t * 1000.0
        t_exp = t.view(B, 1, 1)

        noise_10 = torch.randn_like(nactions_10) * self.noise_scale
        x_t_10 = (1 - t_exp) * noise_10 + t_exp * nactions_10
        x0_pred_10 = self._predict_x0(
            self.model,
            x_t_10,
            timesteps,
            obs_tokens,
            temporal_cond_10hz,
            "10hz",
        )

        noise_50 = torch.randn_like(nactions_50) * self.noise_scale
        x_t_50 = (1 - t_exp) * noise_50 + t_exp * nactions_50
        x0_pred_50 = self._predict_x0(
            self.model,
            x_t_50,
            timesteps,
            obs_tokens,
            temporal_cond_50hz,
            "50hz",
        )
        return (
            x0_pred_10,
            nactions_10,
            x_t_10,
            x0_pred_50,
            nactions_50,
            x_t_50,
            timesteps,
            obs_tokens,
            temporal_cond_10hz,
            temporal_cond_50hz,
        )

    def _compute_stage2_loss(self, batch):
        assert self.teacher_model is not None, "teacher_model must be set for Stage 2"
        nobs = self.normalizer.normalize(batch["obs"])
        nactions_10 = self.normalizer["action"].normalize(batch["action"])
        nactions_50 = self.normalizer["action_stack"].normalize(batch["action_stack"])
        B = nactions_10.shape[0]

        if self.rnn_lowfreq is not None and self.rnn_highfreq is not None:
            assert "extended_obs" in batch, "extended_obs is required for RNNObsEncoder"
            temporal_cond_10hz = self.get_temporal_cond(batch["extended_obs"], head="10hz")
            temporal_cond_50hz = self.get_temporal_cond(batch["extended_obs"], head="50hz")
        else:
            temporal_cond_10hz = None
            temporal_cond_50hz = None

        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])

        t = self._sample_timesteps(B, nactions_10.device)
        timesteps = t * 1000.0
        t_exp = t.view(B, 1, 1)

        noise_10 = torch.randn_like(nactions_10) * self.noise_scale
        x_t_10 = (1 - t_exp) * noise_10 + t_exp * nactions_10
        x0_pred_10 = self._predict_x0(
            self.model,
            x_t_10,
            timesteps,
            obs_tokens,
            temporal_cond_10hz,
            "10hz",
        )
        loss_10 = self._loss_from_x0(
            x_t_10,
            x0_pred_10,
            nactions_10,
            timesteps,
        )

        noise_50 = torch.randn_like(nactions_50) * self.noise_scale
        with torch.no_grad():
            teacher_x0 = self._solve_ode(
                self.teacher_model,
                noise_50,
                obs_tokens,
                temporal_cond_50hz,
                "50hz",
                self.teacher_num_inference_steps,
            )

        x0_pred_50 = self._predict_x0(
            self.model,
            noise_50,
            torch.full((B,), 0.0, device=noise_50.device, dtype=torch.float32),
            obs_tokens,
            temporal_cond_50hz,
            "50hz",
        )
        t_full_noise = torch.full(
            (B,),
            0.0,
            device=noise_50.device,
            dtype=torch.float32,
        )
        loss_50_distill = self._loss_from_x0(
            noise_50,
            x0_pred_50,
            teacher_x0,
            t_full_noise,
        )
        loss_50_gt = self._loss_from_x0(
            noise_50,
            x0_pred_50,
            nactions_50,
            t_full_noise,
        )
        loss_50 = loss_50_distill + self.distill_gt_weight * loss_50_gt

        total_loss = loss_10 + loss_50
        loss_dict = {
            f"{self.loss_type}_loss_10hz": float(loss_10.item()),
            f"distill_{self.loss_type}_loss_50hz": float(loss_50_distill.item()),
            f"gt_{self.loss_type}_loss_50hz": float(loss_50_gt.item()),
            "total_loss": float(total_loss.item()),
            "training_stage": float(self.training_stage),
        }
        return total_loss, loss_dict

    def compute_loss(self, batch, loss_mask=None):
        if self.training_stage == 1:
            (
                x0_pred_10,
                x0_gt_10,
                x_t_10,
                x0_pred_50,
                x0_gt_50,
                x_t_50,
                timesteps,
                *_,
            ) = self.model_forward(batch)
            loss_10 = self._loss_from_x0(
                x_t_10,
                x0_pred_10,
                x0_gt_10,
                timesteps,
            )
            loss_50 = self._loss_from_x0(
                x_t_50,
                x0_pred_50,
                x0_gt_50,
                timesteps,
            )
            fm_loss = loss_10 + loss_50
            return fm_loss, {
                f"{self.loss_type}_loss": float(fm_loss.item()),
                f"{self.loss_type}_loss_10hz": float(loss_10.item()),
                f"{self.loss_type}_loss_50hz": float(loss_50.item()),
                "total_loss": float(fm_loss.item()),
                "training_stage": 1.0,
            }

        if self.training_stage == 2:
            return self._compute_stage2_loss(batch)

        raise ValueError(f"Invalid training_stage: {self.training_stage}. Must be 1 or 2.")

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
            assert extended_obs_dict is not None, "extended_obs is required for RNN temporal condition"

        temporal_cond_10hz = self.get_temporal_cond(extended_obs_dict, head="10hz")
        temporal_cond_50hz = self.get_temporal_cond(extended_obs_dict, head="50hz")

        steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps

        nsample_10 = self._solve_ode(
            self.model,
            torch.randn(
                B,
                self.horizon_10hz,
                self.action_dim,
                device=obs_tokens.device,
                dtype=obs_tokens.dtype,
            )
            * self.noise_scale,
            obs_tokens,
            temporal_cond_10hz,
            "10hz",
            steps,
        )

        # 50Hz inference
        noise_50 = (
            torch.randn(
                B,
                self.horizon_50hz,
                self.action_dim,
                device=obs_tokens.device,
                dtype=obs_tokens.dtype,
            )
            * self.noise_scale
        )
        if self.training_stage == 1:
            # Stage 1: 流匹配模型，用多步 ODE 求解（和 10Hz 一样）
            nsample_50 = self._solve_ode(
                self.model,
                noise_50,
                obs_tokens,
                temporal_cond_50hz,
                "50hz",
                steps,
            )
        else:
            # Stage 2: 蒸馏后的模型，单步 action chunk 预测
            nsample_50, _ = self._one_step_action_chunk(
                self.model,
                noise_50,
                obs_tokens,
                temporal_cond_50hz,
            )

        expected_10 = temporal_cond_10hz.shape[1] if temporal_cond_10hz is not None else self.horizon_10hz
        expected_50 = temporal_cond_50hz.shape[1] if temporal_cond_50hz is not None else self.horizon_50hz
        assert nsample_10.shape == (B, expected_10, self.action_dim)
        assert nsample_50.shape == (B, expected_50, self.action_dim)
        action_pred_10hz = self.normalizer["action"].unnormalize(nsample_10)
        action_pred_50hz = self.normalizer["action_stack"].unnormalize(nsample_50)

        start_10 = self.action_start_10hz
        end_10 = start_10 + self.n_action_steps
        action_10hz = action_pred_10hz[:, start_10:end_10]

        start_50 = self.action_start_50hz
        end_50 = self.horizon_50hz
        action_50hz = action_pred_50hz[:, start_50:end_50]

        if self.inference_head == "50hz":
            action_default = action_50hz
        else:
            action_default = action_10hz

        return {
            "action": action_default,
            "action_pred": action_pred_50hz if self.inference_head == "50hz" else action_pred_10hz,
            "action_10hz": action_10hz,
            "action_50hz": action_50hz,
            "action_pred_10hz": action_pred_10hz,
            "action_pred_50hz": action_pred_50hz,
        }

    def predict_action(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
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
        betas: Tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.Optimizer:
        optim_groups = []

        def _add_groups(module, group_weight_decay, group_lr=None):
            if module is None:
                return
            trainable = [p for p in module.parameters() if p.requires_grad]
            if not trainable:
                return
            if hasattr(module, "get_optim_groups"):
                module_groups = module.get_optim_groups(weight_decay=group_weight_decay)
                for g in module_groups:
                    g["params"] = [p for p in g["params"] if p.requires_grad]
                module_groups = [g for g in module_groups if len(g["params"]) > 0]
                if group_lr is not None:
                    for g in module_groups:
                        g["lr"] = group_lr
                optim_groups.extend(module_groups)
            else:
                optim_groups.append(
                    {
                        "params": trainable,
                        "weight_decay": group_weight_decay,
                        **({} if group_lr is None else {"lr": group_lr}),
                    }
                )

        _add_groups(self.model, weight_decay)
        _add_groups(self.rnn_lowfreq, weight_decay)
        _add_groups(self.rnn_highfreq, weight_decay)

        if self.obs_encoder is not None:
            backbone_params = []
            other_obs_params = []
            for key, value in self.obs_encoder.named_parameters():
                if not value.requires_grad:
                    continue
                if key.startswith("key_model_map"):
                    backbone_params.append(value)
                else:
                    other_obs_params.append(value)
            if len(backbone_params) > 0:
                optim_groups.append(
                    {
                        "params": backbone_params,
                        "weight_decay": obs_encoder_weight_decay,
                        "lr": obs_encoder_lr,
                    }
                )
            if len(other_obs_params) > 0:
                optim_groups.append(
                    {
                        "params": other_obs_params,
                        "weight_decay": obs_encoder_weight_decay,
                    }
                )

        if not optim_groups:
            raise ValueError("No trainable parameters found for optimizer.")

        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
