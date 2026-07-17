import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import reduce
from loguru import logger

from ImplicitRDP.policy.dp_30hz_imeanflow_policy import DP30HziMeanFlowPolicy


class DP30HzConsistencyPolicy(DP30HziMeanFlowPolicy):
    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: torch.nn.Module,
        model: torch.nn.Module,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        n_emb: int = 768,
        use_rnn_obs_encoder: bool = True,
        rnn_obs_encoder_n_layer: int = 1,
        rnn_obs_encoder_hidden_dim: int = 512,
        use_amp_for_inference: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        highfreq_extended_obs_key: str = "right_robot_tcp_wrench_stack",
        action_start_30hz: Optional[int] = None,
        action_normalizer_key: str = "action_stack",
        action_train_key: str = "action_stack",
        num_inference_steps: int = 1,
        sigma_data: float = 0.5,
        sigma_min: float = 0.02,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        num_scales: int = 40,
        ctm_ode_steps_max: int = 1,
        weight_schedule: str = "karras",
        consistency_weight: float = 1.0,
        dsm_weight: float = 1.0,
        sigma_sample_p_mean: float = -1.2,
        sigma_sample_p_std: float = 1.2,
        dsm_karras_proportion: float = 0.5,
        dsm_karras_ramp_max: float = 0.7,
        clamp_log_normal_sigmas: bool = False,
        delta: float = -1.0,
        initial_noise_scale: float = 1.0,
        clip_sample: bool = True,
        clip_sample_min: float = -1.0,
        clip_sample_max: float = 1.0,
        chaining_indices: Optional[Tuple[int, ...]] = None,
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
            n_emb=n_emb,
            use_rnn_obs_encoder=use_rnn_obs_encoder,
            rnn_obs_encoder_n_layer=rnn_obs_encoder_n_layer,
            rnn_obs_encoder_hidden_dim=rnn_obs_encoder_hidden_dim,
            use_amp_for_inference=use_amp_for_inference,
            amp_dtype=amp_dtype,
            highfreq_extended_obs_key=highfreq_extended_obs_key,
            action_start_30hz=action_start_30hz,
            action_normalizer_key=action_normalizer_key,
            action_train_key=action_train_key,
            num_inference_steps=num_inference_steps,
            **kwargs,
        )
        self.sigma_data = float(sigma_data)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.num_scales = int(num_scales)
        self.ctm_ode_steps_max = int(ctm_ode_steps_max)
        self.weight_schedule = weight_schedule
        self.consistency_weight = float(consistency_weight)
        self.dsm_weight = float(dsm_weight)
        self.sigma_sample_p_mean = float(sigma_sample_p_mean)
        self.sigma_sample_p_std = float(sigma_sample_p_std)
        self.dsm_karras_proportion = float(dsm_karras_proportion)
        self.dsm_karras_ramp_max = float(dsm_karras_ramp_max)
        self.clamp_log_normal_sigmas = bool(clamp_log_normal_sigmas)
        self.delta = float(delta)
        self.initial_noise_scale = float(initial_noise_scale)
        self.clip_sample = bool(clip_sample)
        self.clip_sample_min = float(clip_sample_min)
        self.clip_sample_max = float(clip_sample_max)
        self.chaining_indices = (
            None if chaining_indices is None else tuple(int(index) for index in chaining_indices)
        )
        if self.num_scales < 2:
            raise ValueError(f"num_scales must be at least 2, got {self.num_scales}.")
        if not 0.0 <= self.dsm_karras_proportion <= 1.0:
            raise ValueError(
                "dsm_karras_proportion must be in [0, 1], "
                f"got {self.dsm_karras_proportion}."
            )
        if not 0.0 < self.dsm_karras_ramp_max <= 1.0:
            raise ValueError(
                f"dsm_karras_ramp_max must be in (0, 1], got {self.dsm_karras_ramp_max}."
            )
        if self.initial_noise_scale < 0.0:
            raise ValueError(f"initial_noise_scale must be non-negative, got {self.initial_noise_scale}.")
        if self.clip_sample_min >= self.clip_sample_max:
            raise ValueError(
                "clip_sample_min must be smaller than clip_sample_max, "
                f"got {self.clip_sample_min} and {self.clip_sample_max}."
            )
        self.training_stage = int(training_stage)
        self.set_training_stage(self.training_stage)
        logger.info(
            "DP30HzConsistencyPolicy initialized "
            f"(stage={self.training_stage}, num_scales={self.num_scales}, "
            f"sigma_min={self.sigma_min}, sigma_max={self.sigma_max}, "
            f"dsm_weight={self.dsm_weight}, "
            f"dsm_karras_proportion={self.dsm_karras_proportion}, "
            f"initial_noise_scale={self.initial_noise_scale})"
        )

    def set_training_stage(self, stage: int):
        if stage not in (1, 2):
            raise ValueError(f"Invalid training_stage={stage}; expected 1 or 2.")
        self.training_stage = int(stage)

    def _set_module_trainable(self, module, trainable: bool):
        if module is None:
            return
        module.requires_grad_(trainable)
        module.train(trainable)

    def apply_stage_freeze(self):
        if self.training_stage == 1:
            self._set_module_trainable(self.obs_encoder, True)
            self._set_module_trainable(self.rnn_obs_encoder, True)
            self._set_module_trainable(self.model, True)
        elif self.training_stage == 2:
            self._set_module_trainable(self.obs_encoder, False)
            self._set_module_trainable(self.rnn_obs_encoder, False)
            self._set_module_trainable(self.model, True)
        else:
            raise ValueError(f"Invalid training_stage={self.training_stage}")

    def compute_stage1_loss(self, batch, loss_mask=None):
        obs_tokens, trajectory, temporal_cond = self._prepare_inputs(batch)
        batch_size = trajectory.shape[0]
        sigmas = self._sample_log_normal_sigmas(batch_size, trajectory.device, trajectory.dtype)
        loss, raw_loss, denoised = self._compute_dsm_loss(
            trajectory,
            obs_tokens,
            temporal_cond,
            sigmas,
            loss_mask=loss_mask,
        )
        return loss, {
            "edm_dsm_loss": loss.item(),
            "edm_raw_huber": raw_loss.item(),
            "edm_sigma_mean": sigmas.mean().item(),
            "edm_denoised_abs_mean": denoised.detach().abs().mean().item(),
            "training_stage": 1.0,
            "total_loss": loss.item(),
        }

    def _append_dims(self, x: torch.Tensor, target_ndim: int):
        return x.reshape(x.shape + (1,) * (target_ndim - x.ndim))

    def _get_weightings(self, sigmas: torch.Tensor):
        snrs = sigmas.clamp_min(1e-12) ** -2
        if self.weight_schedule == "snr":
            return snrs
        if self.weight_schedule == "snr+1":
            return snrs + 1
        if self.weight_schedule == "karras":
            return snrs + 1.0 / self.sigma_data ** 2
        if self.weight_schedule == "truncated-snr":
            return torch.clamp(snrs, min=1.0)
        if self.weight_schedule == "uniform":
            return torch.ones_like(snrs)
        raise NotImplementedError(f"Unsupported weight_schedule={self.weight_schedule}")

    def _pseudo_huber_per_sample(self, pred: torch.Tensor, target: torch.Tensor):
        squared_error = F.mse_loss(pred, target, reduction="none")
        if self.delta == -1:
            delta = torch.as_tensor(
                math.sqrt(math.prod(pred.shape[1:])) * 0.00054,
                device=pred.device,
                dtype=pred.dtype,
            )
        else:
            delta = torch.as_tensor(self.delta, device=pred.device, dtype=pred.dtype)
        if self.delta == 0:
            loss = squared_error
        else:
            loss = torch.sqrt(squared_error ** 2 + delta ** 2) - delta
        return reduce(loss, "b ... -> b", "mean")

    def _sample_log_normal_sigmas(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        clamp: Optional[bool] = None,
    ):
        sigmas = (
            torch.randn(batch_size, device=device, dtype=dtype) * self.sigma_sample_p_std
            + self.sigma_sample_p_mean
        ).exp()
        if clamp is None:
            clamp = self.clamp_log_normal_sigmas
        if clamp:
            sigmas = sigmas.clamp(self.sigma_min, self.sigma_max)
        return sigmas

    def _sample_ctm_dsm_sigmas(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Match the reference CTM DSM mixture.

        The high-noise component samples a uniform Karras ramp in [0, 0.7]
        by default, while the remaining samples use the EDM log-normal law.
        """
        karras_size = int(math.ceil(batch_size * self.dsm_karras_proportion))
        log_normal_size = batch_size - karras_size
        parts = []
        if karras_size > 0:
            ramp = torch.rand(karras_size, device=device, dtype=dtype)
            ramp = ramp * self.dsm_karras_ramp_max
            max_inv_rho = self.sigma_max ** (1.0 / self.rho)
            min_inv_rho = self.sigma_min ** (1.0 / self.rho)
            karras_sigmas = (
                max_inv_rho + ramp * (min_inv_rho - max_inv_rho)
            ) ** self.rho
            parts.append(karras_sigmas.clamp(self.sigma_min, self.sigma_max))
        if log_normal_size > 0:
            parts.append(
                self._sample_log_normal_sigmas(
                    log_normal_size,
                    device,
                    dtype,
                    clamp=self.clamp_log_normal_sigmas,
                )
            )
        return torch.cat(parts, dim=0)

    def _karras_index_to_sigma(self, indices: torch.Tensor, dtype: torch.dtype):
        idx = indices.to(dtype=dtype).clamp(0, self.num_scales)
        ramp = idx / max(1, self.num_scales - 1)
        max_inv_rho = self.sigma_max ** (1.0 / self.rho)
        min_inv_rho = self.sigma_min ** (1.0 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigmas.clamp(self.sigma_min, self.sigma_max)

    def _get_scalings(self, sigma: torch.Tensor, boundary: bool):
        sigma_data = torch.as_tensor(self.sigma_data, device=sigma.device, dtype=sigma.dtype)
        sigma_min = torch.as_tensor(self.sigma_min, device=sigma.device, dtype=sigma.dtype)
        if boundary:
            c_skip = sigma_data ** 2 / ((sigma - sigma_min) ** 2 + sigma_data ** 2)
            c_out = (sigma - sigma_min) * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
        else:
            c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
            c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
        c_in = 1 / (sigma ** 2 + sigma_data ** 2).sqrt()
        return c_skip, c_out, c_in

    def _cm_denoise(
        self,
        policy,
        sample,
        sigmas,
        obs_tokens,
        temporal_cond,
        boundary=True,
        stop_sigmas: Optional[torch.Tensor] = None,
    ):
        c_skip, c_out, c_in = self._get_scalings(sigmas, boundary=boundary)
        model_input = self._append_dims(c_in, sample.ndim) * sample
        rescaled_t = 250.0 * torch.log(sigmas.clamp_min(1e-44))
        kwargs = {}
        if stop_sigmas is not None:
            kwargs["time_cond"] = 250.0 * torch.log(stop_sigmas.clamp_min(1e-44))
        model_output = policy.model(
            model_input,
            rescaled_t,
            cond=obs_tokens,
            temporal_cond=temporal_cond,
            **kwargs,
        )
        denoised = self._append_dims(c_out, sample.ndim) * model_output + self._append_dims(c_skip, sample.ndim) * sample
        if stop_sigmas is not None:
            ratio = self._append_dims(stop_sigmas / sigmas.clamp_min(1e-12), sample.ndim)
            denoised = sample * ratio + denoised * (1.0 - ratio)
        return model_output, denoised

    def _compute_dsm_loss(
        self,
        x_start: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond: Optional[torch.Tensor],
        sigmas: torch.Tensor,
        loss_mask=None,
        stop_sigmas: Optional[torch.Tensor] = None,
    ):
        noise = torch.randn_like(x_start)
        x_t = x_start + noise * self._append_dims(sigmas, x_start.ndim)
        _, denoised = self._cm_denoise(
            self,
            x_t,
            sigmas,
            obs_tokens,
            temporal_cond,
            boundary=True,
            stop_sigmas=stop_sigmas,
        )
        loss_per = self._pseudo_huber_per_sample(denoised, x_start)
        raw_loss = loss_per.mean()
        loss_per = loss_per * self._get_weightings(sigmas)
        if loss_mask is not None:
            loss_per = loss_per[loss_mask]
        return loss_per.mean(), raw_loss, denoised

    def _sample_ctm_indices(self, batch_size: int, device: torch.device):
        t_idx = torch.randint(0, self.num_scales, (batch_size,), device=device).long()
        s_idx = torch.empty_like(t_idx)
        u_idx = torch.empty_like(t_idx)
        for i, t_i in enumerate(t_idx):
            s_i = torch.randint(int(t_i.item()), self.num_scales + 1, (1,), device=device).long()
            u_i = torch.randint(int(t_i.item()), int(s_i.item()) + 1, (1,), device=device).long()
            u_i = torch.minimum(u_i, t_i + self.ctm_ode_steps_max)
            s_idx[i] = s_i[0]
            u_idx[i] = u_i[0]
        return t_idx, s_idx, u_idx

    def _sample_karras_pairs(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        indices = torch.randint(0, self.num_scales - 1, (batch_size,), device=device)
        ramp = indices.to(dtype) / (self.num_scales - 1)
        ramp_next = (indices + 1).to(dtype) / (self.num_scales - 1)
        max_inv_rho = self.sigma_max ** (1.0 / self.rho)
        min_inv_rho = self.sigma_min ** (1.0 / self.rho)
        t = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        t2 = (max_inv_rho + ramp_next * (min_inv_rho - max_inv_rho)) ** self.rho
        return t, t2

    @torch.no_grad()
    def _teacher_denoise_from_sigma(self, teacher_policy, sample, sigmas, obs_tokens, temporal_cond):
        _, denoised = teacher_policy._cm_denoise(
            teacher_policy,
            sample,
            sigmas,
            obs_tokens,
            temporal_cond,
            boundary=True,
        )
        return denoised

    @torch.no_grad()
    def _heun_solver(self, samples, t, next_t, obs_tokens, temporal_cond, teacher_policy):
        x = samples
        denoiser = self._teacher_denoise_from_sigma(teacher_policy, x, t, obs_tokens, temporal_cond)
        step = self._append_dims(next_t - t, x.ndim)
        mask = (step == 0).to(dtype=x.dtype)
        d = (x - denoiser) / (self._append_dims(t.clamp_min(1e-12), x.ndim) + mask)
        samples = x + d * self._append_dims(next_t - t, x.ndim)
        denoiser = self._teacher_denoise_from_sigma(teacher_policy, samples, next_t, obs_tokens, temporal_cond)
        next_d = (samples - denoiser) / (self._append_dims(next_t.clamp_min(1e-12), x.ndim) + mask)
        out = x + (d + next_d) * self._append_dims((next_t - t) / 2, x.ndim)
        return out * (1.0 - mask) + x * mask

    @torch.no_grad()
    def _teacher_rollout(self, samples, t_idx, u_idx, obs_tokens, temporal_cond, teacher_policy, dtype):
        x = samples
        for d in range(self.ctm_ode_steps_max):
            current_idx = torch.minimum(t_idx + d, u_idx)
            next_idx = torch.minimum(t_idx + d + 1, u_idx)
            current_t = self._karras_index_to_sigma(current_idx, dtype=dtype)
            next_t = self._karras_index_to_sigma(next_idx, dtype=dtype)
            x = self._heun_solver(x, current_t, next_t, obs_tokens, temporal_cond, teacher_policy)
        return x

    def compute_stage2_loss(self, batch, target_policy=None, teacher_policy=None, loss_mask=None):
        if target_policy is None:
            raise ValueError("Stage 2 consistency distillation requires target_policy.")
        if teacher_policy is None:
            raise ValueError("Stage 2 CTM distillation requires teacher_policy.")
        target_policy.requires_grad_(False)
        teacher_policy.requires_grad_(False)
        obs_tokens, x_start, temporal_cond = self._prepare_inputs(batch)
        batch_size = x_start.shape[0]
        noise = torch.randn_like(x_start)
        t_idx, s_idx, u_idx = self._sample_ctm_indices(batch_size, x_start.device)
        t = self._karras_index_to_sigma(t_idx, x_start.dtype)
        s = self._karras_index_to_sigma(s_idx, x_start.dtype)
        u = self._karras_index_to_sigma(u_idx, x_start.dtype)
        x_t = x_start + noise * self._append_dims(t, x_start.ndim)

        _, pred = self._cm_denoise(
            self,
            x_t,
            t,
            obs_tokens,
            temporal_cond,
            boundary=True,
            stop_sigmas=s,
        )
        with torch.no_grad():
            x_u = self._teacher_rollout(
                x_t, t_idx, u_idx, obs_tokens, temporal_cond, teacher_policy, x_start.dtype
            ).detach()
            _, target = self._cm_denoise(
                target_policy,
                x_u,
                u,
                obs_tokens,
                temporal_cond,
                boundary=True,
                stop_sigmas=s,
            )
            start = torch.full_like(s, self.sigma_min)
            _, target = self._cm_denoise(
                target_policy,
                target,
                s,
                obs_tokens,
                temporal_cond,
                boundary=True,
                stop_sigmas=start,
            )
            target = target.detach()

        start = torch.full_like(s, self.sigma_min)
        _, pred = self._cm_denoise(
            target_policy,
            pred,
            s,
            obs_tokens,
            temporal_cond,
            boundary=True,
            stop_sigmas=start,
        )

        loss_per = self._pseudo_huber_per_sample(pred, target)
        if loss_mask is not None:
            loss_per = loss_per[loss_mask]
        consistency_loss = loss_per.mean()
        dsm_sigmas = self._sample_ctm_dsm_sigmas(
            batch_size,
            x_start.device,
            x_start.dtype,
        )
        dsm_stop = torch.full_like(dsm_sigmas, self.sigma_min)
        dsm_loss, dsm_raw_loss, denoised = self._compute_dsm_loss(
            x_start,
            obs_tokens,
            temporal_cond,
            dsm_sigmas,
            loss_mask=loss_mask,
            stop_sigmas=dsm_stop,
        )
        total = self.consistency_weight * consistency_loss + self.dsm_weight * dsm_loss
        return total, {
            "cm_loss": total.item(),
            "cm_consistency_loss": consistency_loss.item(),
            "cm_dsm_loss": dsm_loss.item(),
            "cm_dsm_raw_huber": dsm_raw_loss.item(),
            "cm_t_mean": t.mean().item(),
            "cm_s_mean": s.mean().item(),
            "cm_u_mean": u.mean().item(),
            "cm_dsm_sigma_mean": dsm_sigmas.mean().item(),
            "cm_denoised_abs_mean": denoised.detach().abs().mean().item(),
            "training_stage": 2.0,
            "total_loss": total.item(),
        }

    def compute_loss(self, batch, target_policy=None, teacher_policy=None, loss_mask=None):
        if self.training_stage == 1:
            return self.compute_stage1_loss(batch, loss_mask)
        if self.training_stage == 2:
            return self.compute_stage2_loss(
                batch,
                target_policy=target_policy,
                teacher_policy=teacher_policy,
                loss_mask=loss_mask,
            )
        raise ValueError(f"Invalid training_stage={self.training_stage}")

    def forward(self, batch, target_policy=None, teacher_policy=None, loss_mask=None):
        return self.compute_loss(
            batch,
            target_policy=target_policy,
            teacher_policy=teacher_policy,
            loss_mask=loss_mask,
        )

    def _get_karras_sigmas(self, n: int, device: torch.device, dtype: torch.dtype):
        ramp = torch.linspace(0, 1, max(2, n), device=device, dtype=dtype)
        min_inv_rho = self.sigma_min ** (1.0 / self.rho)
        max_inv_rho = self.sigma_max ** (1.0 / self.rho)
        return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho

    def _resolve_chaining_indices(self, steps: int):
        if steps <= 1:
            return []
        if steps > self.num_scales:
            raise ValueError(
                f"num_inference_steps={steps} exceeds num_scales={self.num_scales}."
            )
        if self.chaining_indices is not None:
            required = steps - 1
            if len(self.chaining_indices) < required:
                raise ValueError(
                    f"Need at least {required} chaining_indices for {steps} inference steps, "
                    f"got {len(self.chaining_indices)}."
                )
            indices = list(self.chaining_indices[:required])
        else:
            indices = [round(i * self.num_scales / steps) for i in range(1, steps)]
        if any(index <= 0 or index >= self.num_scales for index in indices):
            raise ValueError(
                f"chaining_indices must lie in [1, {self.num_scales - 1}], got {indices}."
            )
        if any(left >= right for left, right in zip(indices, indices[1:])):
            raise ValueError(f"chaining_indices must be strictly increasing, got {indices}.")
        return indices

    def _clip_inference_sample(self, sample: torch.Tensor):
        if not self.clip_sample:
            return sample
        return sample.clamp(self.clip_sample_min, self.clip_sample_max)

    def conditional_sample(self, cond, temporal_cond=None, generator=None, num_inference_steps=None, noisy_trajectory=None):
        batch_size = cond.shape[0]
        steps = self.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        steps = max(1, steps)
        if noisy_trajectory is None:
            trajectory = torch.randn(
                (batch_size, self.action_horizon, self.action_dim),
                dtype=cond.dtype,
                device=cond.device,
                generator=generator,
            ) * self.initial_noise_scale
        else:
            trajectory = noisy_trajectory

        sigma = torch.full(
            (batch_size,),
            self.sigma_max,
            device=trajectory.device,
            dtype=trajectory.dtype,
        )
        stop_sigma = torch.full_like(sigma, self.sigma_min)
        _, trajectory = self._cm_denoise(
            self,
            trajectory,
            sigma,
            cond,
            temporal_cond,
            boundary=True,
            stop_sigmas=stop_sigma,
        )
        trajectory = self._clip_inference_sample(trajectory)

        for index in self._resolve_chaining_indices(steps):
            sigma = self._karras_index_to_sigma(
                torch.full(
                    (batch_size,),
                    index,
                    device=trajectory.device,
                    dtype=torch.long,
                ),
                dtype=trajectory.dtype,
            )
            noise = torch.randn(
                trajectory.shape,
                dtype=trajectory.dtype,
                device=trajectory.device,
                generator=generator,
            )
            trajectory = trajectory + noise * self._append_dims(sigma, trajectory.ndim)
            _, trajectory = self._cm_denoise(
                self,
                trajectory,
                sigma,
                cond,
                temporal_cond,
                boundary=True,
                stop_sigmas=stop_sigma,
            )
            trajectory = self._clip_inference_sample(trajectory)
        return trajectory

    def sample_noisy_trajectory(self, B: int = 1, generator=None, return_expanded_noisy_trajectory: bool = False):
        noisy = torch.randn(
            (B, self.action_horizon, self.action_dim),
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        ) * self.initial_noise_scale
        if return_expanded_noisy_trajectory:
            return noisy
        return noisy.reshape(B, -1)

    def get_optimizer(
        self,
        lr: float,
        weight_decay: float,
        obs_encoder_lr: Optional[float] = None,
        obs_encoder_weight_decay: Optional[float] = None,
        betas: Tuple[float, float] = (0.9, 0.95),
        **kwargs,
    ) -> torch.optim.Optimizer:
        self.apply_stage_freeze()
        if obs_encoder_lr is None:
            obs_encoder_lr = lr
        if obs_encoder_weight_decay is None:
            obs_encoder_weight_decay = weight_decay

        optim_groups = []
        if self.training_stage == 1:
            if self.rnn_obs_encoder is not None:
                optim_groups += self.rnn_obs_encoder.get_optim_groups(weight_decay=weight_decay)
            backbone_params = []
            other_obs_params = []
            for key, value in self.obs_encoder.named_parameters():
                if key.startswith("key_model_map"):
                    backbone_params.append(value)
                else:
                    other_obs_params.append(value)
            optim_groups.append({
                "params": backbone_params,
                "weight_decay": obs_encoder_weight_decay,
                "lr": obs_encoder_lr,
            })
            optim_groups.append({
                "params": other_obs_params,
                "weight_decay": obs_encoder_weight_decay,
            })
            optim_groups.extend(self.model.get_optim_groups(weight_decay=weight_decay))
        elif self.training_stage == 2:
            optim_groups.extend(self.model.get_optim_groups(weight_decay=weight_decay))
        else:
            raise ValueError(f"Invalid training_stage={self.training_stage}")
        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
