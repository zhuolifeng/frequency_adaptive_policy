from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from einops import reduce
from loguru import logger

from ImplicitRDP.policy.dp_30hz_imeanflow_policy import DP30HziMeanFlowPolicy


class DP30HzConsistencyPolicy(DP30HziMeanFlowPolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDIMScheduler,
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
        loss_type: str = "eps",
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        num_scales: int = 40,
        weight_schedule: str = "karras",
        consistency_weight: float = 1.0,
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
        self.noise_scheduler = noise_scheduler
        self.loss_type = (loss_type or "eps").lower()
        if self.loss_type == "epsilon":
            self.loss_type = "eps"
        if self.loss_type in ("v", "v_prediction"):
            self.loss_type = "velocity"
        if self.loss_type not in ("sample", "eps", "velocity"):
            raise ValueError(f"Unsupported loss_type={loss_type}")

        self.sigma_data = float(sigma_data)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.num_scales = int(num_scales)
        self.weight_schedule = weight_schedule
        self.consistency_weight = float(consistency_weight)
        self.training_stage = int(training_stage)
        self.set_training_stage(self.training_stage)
        logger.info(
            "DP30HzConsistencyPolicy initialized "
            f"(stage={self.training_stage}, num_scales={self.num_scales}, "
            f"sigma_min={self.sigma_min}, sigma_max={self.sigma_max})"
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

    def _get_variance_terms(self, timesteps: torch.Tensor, sample: torch.Tensor):
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod = alphas_cumprod[timesteps].view(-1, 1, 1)
        beta_prod = 1.0 - alpha_prod
        return alpha_prod, beta_prod

    def _prediction_to_x0_eps(self, sample, model_output, timesteps):
        alpha_prod, beta_prod = self._get_variance_terms(timesteps, sample)
        sqrt_alpha = alpha_prod.sqrt()
        sqrt_beta = beta_prod.sqrt()
        if self.loss_type == "velocity":
            pred_velocity = self._sample_prediction_to_velocity(sample, model_output, timesteps)
            pred_x0 = sqrt_alpha * sample - sqrt_beta * pred_velocity
            pred_eps = sqrt_alpha * pred_velocity + sqrt_beta * sample
        else:
            pred_x0 = model_output
            pred_eps = (sample - sqrt_alpha * pred_x0) / sqrt_beta.clamp_min(1e-12)
        return pred_x0, pred_eps

    def _sample_prediction_to_velocity(self, sample, model_output, timesteps):
        alpha_prod, beta_prod = self._get_variance_terms(timesteps, sample)
        sqrt_alpha = alpha_prod.sqrt()
        sqrt_beta = beta_prod.sqrt().clamp_min(1e-12)
        return (sqrt_alpha * sample - model_output) / sqrt_beta

    def compute_stage1_loss(self, batch, loss_mask=None):
        obs_tokens, trajectory, temporal_cond = self._prepare_inputs(batch)
        batch_size = trajectory.shape[0]
        device = trajectory.device
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
        ).long()
        noisy = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        pred = self.model(noisy, timesteps, cond=obs_tokens, temporal_cond=temporal_cond)

        if self.loss_type == "sample":
            loss_all = F.mse_loss(pred, trajectory, reduction="none")
        elif self.loss_type == "eps":
            _, pred_eps = self._prediction_to_x0_eps(noisy, pred, timesteps)
            loss_all = F.mse_loss(pred_eps, noise, reduction="none")
        elif self.loss_type == "velocity":
            pred_velocity = self._sample_prediction_to_velocity(noisy, pred, timesteps)
            target_velocity = self.noise_scheduler.get_velocity(trajectory, noise, timesteps)
            loss_all = F.mse_loss(pred_velocity, target_velocity, reduction="none")
        else:
            raise ValueError(f"Unsupported loss_type={self.loss_type}")

        loss_per = reduce(loss_all, "b ... -> b", "mean")
        if loss_mask is not None:
            loss_per = loss_per[loss_mask]
        loss = loss_per.mean()
        return loss, {
            "base_dp_loss": loss.item(),
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

    def _cm_denoise(self, policy, sample, sigmas, obs_tokens, temporal_cond, boundary=True):
        c_skip, c_out, c_in = self._get_scalings(sigmas, boundary=boundary)
        model_input = self._append_dims(c_in, sample.ndim) * sample
        rescaled_t = 250.0 * torch.log(sigmas.clamp_min(1e-44))
        model_output = policy.model(
            model_input,
            rescaled_t,
            cond=obs_tokens,
            temporal_cond=temporal_cond,
        )
        denoised = self._append_dims(c_out, sample.ndim) * model_output + self._append_dims(c_skip, sample.ndim) * sample
        return model_output, denoised

    def _sample_karras_pairs(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        indices = torch.randint(0, self.num_scales - 1, (batch_size,), device=device)
        ramp = indices.to(dtype) / (self.num_scales - 1)
        ramp_next = (indices + 1).to(dtype) / (self.num_scales - 1)
        max_inv_rho = self.sigma_max ** (1.0 / self.rho)
        min_inv_rho = self.sigma_min ** (1.0 / self.rho)
        t = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        t2 = (max_inv_rho + ramp_next * (min_inv_rho - max_inv_rho)) ** self.rho
        return t, t2

    def _sigma_to_ddim_timestep(self, sigmas: torch.Tensor, dtype: torch.dtype):
        alpha = 1.0 / (1.0 + sigmas ** 2)
        alphas = self.noise_scheduler.alphas_cumprod.to(device=sigmas.device, dtype=dtype)
        return torch.argmin(torch.abs(alphas[None, :] - alpha[:, None].to(dtype)), dim=1).long()

    @torch.no_grad()
    def _teacher_denoise_from_sigma(self, teacher_policy, sample, sigmas, obs_tokens, temporal_cond):
        timesteps = teacher_policy._sigma_to_ddim_timestep(sigmas, sample.dtype)
        teacher_input = sample / self._append_dims(torch.sqrt(1.0 + sigmas ** 2).to(sample.dtype), sample.ndim)
        return teacher_policy.model(
            teacher_input,
            timesteps,
            cond=obs_tokens,
            temporal_cond=temporal_cond,
        )

    @torch.no_grad()
    def _heun_solver(self, samples, t, next_t, x0, obs_tokens, temporal_cond, teacher_policy):
        x = samples
        denoiser = x0 if teacher_policy is None else self._teacher_denoise_from_sigma(
            teacher_policy, x, t, obs_tokens, temporal_cond
        )
        d = (x - denoiser) / self._append_dims(t.clamp_min(1e-12), x.ndim)
        samples = x + d * self._append_dims(next_t - t, x.ndim)
        denoiser = x0 if teacher_policy is None else self._teacher_denoise_from_sigma(
            teacher_policy, samples, next_t, obs_tokens, temporal_cond
        )
        next_d = (samples - denoiser) / self._append_dims(next_t.clamp_min(1e-12), x.ndim)
        return x + (d + next_d) * self._append_dims((next_t - t) / 2, x.ndim)

    def compute_stage2_loss(self, batch, target_policy=None, teacher_policy=None, loss_mask=None):
        if target_policy is None:
            raise ValueError("Stage 2 consistency distillation requires target_policy.")
        obs_tokens, x_start, temporal_cond = self._prepare_inputs(batch)
        batch_size = x_start.shape[0]
        noise = torch.randn_like(x_start)
        t, t2 = self._sample_karras_pairs(batch_size, x_start.device, x_start.dtype)
        x_t = x_start + noise * self._append_dims(t, x_start.ndim)

        _, distiller = self._cm_denoise(self, x_t, t, obs_tokens, temporal_cond, boundary=True)
        with torch.no_grad():
            x_t2 = self._heun_solver(
                x_t, t, t2, x_start, obs_tokens, temporal_cond, teacher_policy
            ).detach()
            _, distiller_target = self._cm_denoise(
                target_policy, x_t2, t2, obs_tokens, temporal_cond, boundary=True
            )
            distiller_target = distiller_target.detach()

        loss_per = reduce((distiller - distiller_target) ** 2, "b ... -> b", "mean")
        loss_per = loss_per * self._get_weightings(t)
        if loss_mask is not None:
            loss_per = loss_per[loss_mask]
        consistency_loss = loss_per.mean()
        total = self.consistency_weight * consistency_loss
        return total, {
            "cm_loss": total.item(),
            "cm_consistency_loss": consistency_loss.item(),
            "cm_t_mean": t.mean().item(),
            "cm_t2_mean": t2.mean().item(),
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
        ramp = torch.linspace(0, 1, n, device=device, dtype=dtype)
        min_inv_rho = self.sigma_min ** (1.0 / self.rho)
        max_inv_rho = self.sigma_max ** (1.0 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return torch.cat([sigmas, torch.zeros(1, device=device, dtype=dtype)])

    def conditional_sample(self, cond, temporal_cond=None, generator=None, num_inference_steps=None, noisy_trajectory=None):
        batch_size = cond.shape[0]
        steps = self.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        sigmas = self._get_karras_sigmas(max(1, steps), cond.device, cond.dtype)
        if noisy_trajectory is None:
            trajectory = torch.randn(
                (batch_size, self.action_horizon, self.action_dim),
                dtype=cond.dtype,
                device=cond.device,
                generator=generator,
            ) * sigmas[0]
        else:
            trajectory = noisy_trajectory

        for i in range(len(sigmas) - 1):
            sigma = torch.full((batch_size,), sigmas[i].item(), device=trajectory.device, dtype=trajectory.dtype)
            next_sigma = torch.full((batch_size,), sigmas[i + 1].item(), device=trajectory.device, dtype=trajectory.dtype)
            _, denoised = self._cm_denoise(self, trajectory, sigma, cond, temporal_cond, boundary=True)
            if sigmas[i + 1].item() == 0:
                trajectory = denoised
            else:
                d = (trajectory - denoised) / self._append_dims(sigma.clamp_min(1e-12), trajectory.ndim)
                trajectory = trajectory + d * self._append_dims(next_sigma - sigma, trajectory.ndim)
        return trajectory

    def sample_noisy_trajectory(self, B: int = 1, generator=None, return_expanded_noisy_trajectory: bool = False):
        noisy = torch.randn(
            (B, self.action_horizon, self.action_dim),
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        ) * self.sigma_max
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
