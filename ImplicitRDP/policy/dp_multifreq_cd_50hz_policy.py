"""
50Hz consistency distillation policy for the multi-frequency DP model.

Stage 2 keeps the same diffusion transformer backbone as the supervised
multi-frequency DP policy. The 50Hz branch is trained with consistency
distillation, while the 10Hz branch continues standard diffusion training.
The EMA model used as the CD target is managed by the workspace.
"""
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from ImplicitRDP.policy.dp_multifreq_policy import DPMultiFreqPolicy


class DPMultiFreqCD50HzPolicy(DPMultiFreqPolicy):
    def __init__(
        self,
        *args,
        cd_timestep_grid: List[int] = (99, 79, 59, 39, 19, 0),
        cd_teacher_rollout_steps: int = 2,
        cd_weight: float = 1.0,
        dsm_weight: float = 0.0,
        loss_10hz_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, training_stage=1, **kwargs)
        self.cd_timestep_grid = [int(x) for x in cd_timestep_grid]
        if len(self.cd_timestep_grid) < 2:
            raise ValueError("cd_timestep_grid must contain at least two timesteps.")
        if any(
            self.cd_timestep_grid[i] <= self.cd_timestep_grid[i + 1]
            for i in range(len(self.cd_timestep_grid) - 1)
        ):
            raise ValueError(
                "cd_timestep_grid must be strictly descending, e.g. [99, 79, ..., 0]."
            )
        if cd_teacher_rollout_steps < 1:
            raise ValueError("cd_teacher_rollout_steps must be >= 1.")
        self.cd_teacher_rollout_steps = int(cd_teacher_rollout_steps)
        self.cd_weight = float(cd_weight)
        self.dsm_weight = float(dsm_weight)
        self.loss_10hz_weight = float(loss_10hz_weight)
        self.training_stage = 2

    def set_training_stage(self, stage: int):
        if stage != 2:
            raise ValueError(
                "DPMultiFreqCD50HzPolicy is for stage-2 50Hz CD; only stage=2 is supported."
            )
        self.training_stage = 2

    def _sample_cd_timestep_pair(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid = torch.tensor(self.cd_timestep_grid, dtype=torch.long, device=device)
        pair_idx = torch.randint(0, len(self.cd_timestep_grid) - 1, (batch_size,), device=device)
        return grid[pair_idx], grid[pair_idx + 1]

    def _teacher_rollout_to_timestep(
        self,
        teacher_policy: DPMultiFreqPolicy,
        sample: torch.Tensor,
        start_timestep: torch.Tensor,
        target_timestep: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond: torch.Tensor,
    ) -> torch.Tensor:
        x = sample
        cur_t = start_timestep
        for step_idx in range(1, self.cd_teacher_rollout_steps + 1):
            next_t = (
                start_timestep.float()
                + (target_timestep.float() - start_timestep.float())
                * (step_idx / self.cd_teacher_rollout_steps)
            ).round().long()
            next_t = torch.maximum(next_t, target_timestep)
            next_t = torch.minimum(next_t, cur_t - 1)
            if step_idx == self.cd_teacher_rollout_steps:
                next_t = target_timestep

            teacher_out = teacher_policy.model(
                x,
                cur_t,
                cond=obs_tokens,
                temporal_cond=temporal_cond,
                freq_mode="50hz",
            )
            x = teacher_policy._ddim_jump_to(x, teacher_out, cur_t, next_t)
            cur_t = next_t
        return x

    def _prepare_stage2_inputs(self, batch: Dict[str, torch.Tensor]):
        nobs = self.normalizer.normalize(batch["obs"])
        nactions_10 = self.normalizer["action"].normalize(batch["action"])
        nactions_50 = self.normalizer["action_stack"].normalize(batch["action_stack"])
        batch_size = nactions_50.shape[0]

        assert "extended_obs" in batch, "extended_obs is required for RNN temporal condition"
        temporal_cond_10hz = self.get_temporal_cond(batch["extended_obs"], head="10hz")
        temporal_cond_50hz = self.get_temporal_cond(batch["extended_obs"], head="50hz")

        from ImplicitRDP.common.pytorch_util import dict_apply

        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(batch_size, -1, *obs_tokens.shape[1:])
        return obs_tokens, nactions_10, nactions_50, temporal_cond_10hz, temporal_cond_50hz

    def compute_cd_loss(
        self,
        batch: Dict[str, torch.Tensor],
        teacher_policy: DPMultiFreqPolicy,
        target_policy: DPMultiFreqPolicy,
    ):
        (
            obs_tokens,
            x0_10hz,
            x0_50hz,
            temporal_cond_10hz,
            temporal_cond_50hz,
        ) = self._prepare_stage2_inputs(batch)
        batch_size = x0_50hz.shape[0]
        device = x0_50hz.device

        t, u = self._sample_cd_timestep_pair(batch_size, device)
        noise = torch.randn_like(x0_50hz)
        x_t = self.noise_scheduler.add_noise(x0_50hz, noise, t)

        with torch.no_grad():
            x_u = self._teacher_rollout_to_timestep(
                teacher_policy=teacher_policy,
                sample=x_t,
                start_timestep=t,
                target_timestep=u,
                obs_tokens=obs_tokens,
                temporal_cond=temporal_cond_50hz,
            )

            target_out_u = target_policy.model(
                x_u,
                u,
                cond=obs_tokens,
                temporal_cond=temporal_cond_50hz,
                freq_mode="50hz",
            )
            target_x0_u, _ = target_policy._prediction_to_x0_eps(x_u, target_out_u, u)

        student_out_t = self.model(
            x_t,
            t,
            cond=obs_tokens,
            temporal_cond=temporal_cond_50hz,
            freq_mode="50hz",
        )
        student_x0_t, _ = self._prediction_to_x0_eps(x_t, student_out_t, t)

        cd_consistency_loss = F.mse_loss(student_x0_t, target_x0_u.detach())
        train_cd_loss = self.cd_weight * cd_consistency_loss

        loss_dict = {
            "cd_loss": train_cd_loss.item(),
            "cd_consistency_loss": cd_consistency_loss.item(),
            "cd_t_mean": t.float().mean().item(),
            "cd_u_mean": u.float().mean().item(),
            "cd_teacher_rollout_steps": float(self.cd_teacher_rollout_steps),
            "training_stage": 2.0,
        }

        if self.dsm_weight > 0:
            dsm_loss = F.mse_loss(student_x0_t, x0_50hz)
            train_cd_loss = train_cd_loss + self.dsm_weight * dsm_loss
            loss_dict["cd_loss"] = train_cd_loss.item()
            loss_dict["dsm_loss"] = dsm_loss.item()
            loss_dict["weighted_dsm_loss"] = (self.dsm_weight * dsm_loss).item()

        loss_10hz = self._compute_dp_loss_one_freq(
            self.model,
            x0_10hz,
            obs_tokens,
            temporal_cond_10hz,
            "10hz",
        )
        total_loss = train_cd_loss + self.loss_10hz_weight * loss_10hz
        loss_dict["dp_loss_10hz"] = loss_10hz.item()
        loss_dict["weighted_dp_loss_10hz"] = (
            self.loss_10hz_weight * loss_10hz
        ).item()
        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict

    def compute_loss(self, batch, loss_mask=None):
        raise RuntimeError(
            "Use compute_cd_loss(batch, teacher_policy, target_policy) for stage-2 CD."
        )

    def forward(self, batch, teacher_policy, target_policy):
        return self.compute_cd_loss(batch, teacher_policy, target_policy)

    def _ddim_sample(
        self,
        model,
        cond=None,
        temporal_cond=None,
        freq_mode: str = "10hz",
        horizon: int = 16,
        generator=None,
        num_inference_steps: Optional[int] = None,
        noisy_trajectory=None,
        **kwargs,
    ):
        steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        if freq_mode != "50hz" or steps != 1:
            return super()._ddim_sample(
                model,
                cond=cond,
                temporal_cond=temporal_cond,
                freq_mode=freq_mode,
                horizon=horizon,
                generator=generator,
                num_inference_steps=num_inference_steps,
                noisy_trajectory=noisy_trajectory,
                **kwargs,
            )

        if noisy_trajectory is None:
            trajectory = torch.randn(
                size=(cond.shape[0], horizon, self.action_dim),
                dtype=cond.dtype,
                device=cond.device,
                generator=generator,
            )
        else:
            trajectory = noisy_trajectory

        # Diffusers DDIM uses timestep 0 for one-step leading spacing, but CD
        # trains the student at the first grid timestep, e.g. 99.
        t_value = int(self.cd_timestep_grid[0])
        self.noise_scheduler.set_timesteps(1, device=trajectory.device)
        t = torch.tensor(t_value, dtype=torch.long, device=trajectory.device)
        t_batch = torch.full(
            (trajectory.shape[0],),
            t_value,
            dtype=torch.long,
            device=trajectory.device,
        )
        model_output = model(
            trajectory,
            t_batch,
            cond=cond,
            temporal_cond=temporal_cond,
            freq_mode=freq_mode,
        )
        trajectory = self.noise_scheduler.step(
            model_output, t, trajectory, generator=generator, **self.kwargs
        ).prev_sample
        return trajectory

    def _predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        dataset_obs_temporal_downsample_ratio: int = 1,
        extended_obs_dict: Optional[Dict[str, torch.Tensor]] = None,
        num_inference_steps: Optional[int] = None,
        num_inference_steps_10hz: Optional[int] = None,
        num_inference_steps_50hz: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        assert extended_obs_dict is not None, "extended_obs is required for RNN temporal condition"

        obs_tokens = self._encode_obs(obs_dict)
        temporal_cond_10hz = self.get_temporal_cond(extended_obs_dict, head="10hz")
        temporal_cond_50hz = self.get_temporal_cond(extended_obs_dict, head="50hz")
        steps_10hz = (
            num_inference_steps if num_inference_steps_10hz is None else num_inference_steps_10hz
        )
        steps_50hz = (
            num_inference_steps if num_inference_steps_50hz is None else num_inference_steps_50hz
        )

        nsample_10hz = self.conditional_sample(
            cond=obs_tokens,
            temporal_cond=temporal_cond_10hz,
            freq_mode="10hz",
            num_inference_steps=steps_10hz,
        )
        nsample_50hz = self.conditional_sample(
            cond=obs_tokens,
            temporal_cond=temporal_cond_50hz,
            freq_mode="50hz",
            num_inference_steps=steps_50hz,
        )

        action_pred_10hz = self.normalizer["action"].unnormalize(nsample_10hz)
        action_pred_50hz = self.normalizer["action_stack"].unnormalize(nsample_50hz)
        action_10hz = action_pred_10hz[
            :, self.action_start_10hz:self.action_start_10hz + self.n_action_steps
        ]
        action_50hz = action_pred_50hz[:, self.action_start_50hz:self.horizon_50hz]

        if self.inference_head == "50hz":
            action = action_50hz
            action_pred = action_pred_50hz
        else:
            action = action_10hz
            action_pred = action_pred_10hz

        return {
            "action": action,
            "action_pred": action_pred,
            "action_10hz": action_10hz,
            "action_50hz": action_50hz,
            "action_pred_10hz": action_pred_10hz,
            "action_pred_50hz": action_pred_50hz,
        }
