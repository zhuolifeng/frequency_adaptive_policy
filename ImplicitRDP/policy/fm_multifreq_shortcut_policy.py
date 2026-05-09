"""
One-stage multi-frequency flow matching policy with 50Hz shortcut training.

10Hz is trained with standard flow matching for multi-step Euler inference.
50Hz is trained with both flow-matching anchors and shortcut bootstrap targets
for one-step or few-step inference.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import reduce
from torch.cuda.amp import autocast

from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.model.diffusion.transformer_for_shortcut_multifreq import (
    TransformerForShortcutMultiFreq,
)
from ImplicitRDP.policy.fm_multifreq_policy import FMMultiFreqPolicy


class FMMultiFreqShortcutPolicy(FMMultiFreqPolicy):
    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: torch.nn.Module,
        model: TransformerForShortcutMultiFreq,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: int = 1,
        beta_a: float = 1.5,
        beta_b: float = 1.0,
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
        inference_head: str = "50hz",
        shortcut_denoise_steps: int = 128,
        shortcut_bootstrap_every: int = 8,
        fm_10hz_weight: float = 1.0,
        fm_50hz_weight: float = 1.0,
        shortcut_50hz_weight: float = 1.0,
        shortcut_one_step_50hz_weight: float = 1.0,
        shortcut_use_ema_target: bool = True,
        num_inference_steps_10hz: int = 10,
        shortcut_version: str = "new",
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
            use_distillation=False,
            inference_head=inference_head,
            training_stage=1,
            **kwargs,
        )
        if shortcut_version not in ("old", "new"):
            raise ValueError(
                f"Unsupported shortcut_version={shortcut_version}. Expected 'old' or 'new'."
            )
        assert shortcut_denoise_steps >= 2
        assert shortcut_denoise_steps & (shortcut_denoise_steps - 1) == 0, (
            "shortcut_denoise_steps must be a power of two"
        )
        assert shortcut_bootstrap_every >= 1

        self.shortcut_denoise_steps = shortcut_denoise_steps
        self.shortcut_dt_flow = int(torch.log2(torch.tensor(shortcut_denoise_steps)).item())
        self.shortcut_bootstrap_every = shortcut_bootstrap_every
        self.fm_10hz_weight = fm_10hz_weight
        self.fm_50hz_weight = fm_50hz_weight
        self.shortcut_50hz_weight = shortcut_50hz_weight
        self.shortcut_one_step_50hz_weight = shortcut_one_step_50hz_weight
        self.shortcut_use_ema_target = shortcut_use_ema_target
        self.num_inference_steps_10hz = num_inference_steps_10hz
        self.shortcut_version = shortcut_version
        self.ema_model_ref = None

    def set_ema_model_ref(self, ema_model):
        self.ema_model_ref = ema_model

    def _encode_batch(self, batch):
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
        return obs_tokens, nactions_10, nactions_50, temporal_cond_10hz, temporal_cond_50hz

    def _sample_flow(self, x0: torch.Tensor):
        B = x0.shape[0]
        t = torch.distributions.Beta(self.beta_a, self.beta_b).sample((B,)).to(x0.device)
        t = t * 0.999 + 0.001
        noise = torch.randn_like(x0)
        x_t = t.view(B, 1, 1) * noise + (1.0 - t.view(B, 1, 1)) * x0
        target = noise - x0
        timestep = (t * 1000).long()
        dt_base = torch.full(
            (B,), self.shortcut_dt_flow, dtype=torch.float32, device=x0.device
        )
        return x_t, timestep, dt_base, target

    def _sample_origin_flow(self, x1: torch.Tensor):
        B = x1.shape[0]
        t = torch.randint(
            0, self.shortcut_denoise_steps, (B,), device=x1.device
        ).float()
        t = t / float(self.shortcut_denoise_steps)
        noise = torch.randn_like(x1)
        x_t = (1.0 - (1.0 - 1e-5) * t.view(B, 1, 1)) * noise + t.view(B, 1, 1) * x1
        target = x1 - (1.0 - 1e-5) * noise
        timestep = t
        dt_base = torch.full(
            (B,), self.shortcut_dt_flow, dtype=torch.float32, device=x1.device
        )
        return x_t, timestep, dt_base, target

    def _model_call(
        self,
        model,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        dt_base: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond: torch.Tensor,
        freq_mode: str,
    ):
        return model(
            sample,
            timestep,
            cond=obs_tokens,
            temporal_cond=temporal_cond,
            freq_mode=freq_mode,
            shortcut_dt_base=dt_base,
        )

    def _compute_fm_loss_one_freq(
        self,
        trajectory: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond: torch.Tensor,
        freq_mode: str,
        loss_mask=None,
    ):
        if self.shortcut_version == "new":
            x_t, timestep, dt_base, target = self._sample_origin_flow(trajectory)
        else:
            x_t, timestep, dt_base, target = self._sample_flow(trajectory)
        pred = self._model_call(
            self.model, x_t, timestep, dt_base, obs_tokens, temporal_cond, freq_mode
        )
        loss_all = F.mse_loss(pred, target, reduction="none")
        loss_per = reduce(loss_all, "b ... -> b (...)", "mean")
        return self._mean_loss(loss_per, loss_mask)

    def _sample_shortcut_dt_base(self, B: int, device: torch.device) -> torch.Tensor:
        max_base = max(1, self.shortcut_dt_flow)
        return torch.randint(0, max_base, (B,), device=device).float()

    def _sample_shortcut_dt_base_balanced(self, B: int, device: torch.device) -> torch.Tensor:
        log2_sections = max(1, self.shortcut_dt_flow)
        dt_values = (log2_sections - 1 - torch.arange(log2_sections, device=device)).float()
        repeats = B // log2_sections
        dt_base = dt_values.repeat_interleave(repeats)
        if dt_base.shape[0] < B:
            pad = torch.zeros(B - dt_base.shape[0], dtype=torch.float32, device=device)
            dt_base = torch.cat([dt_base, pad], dim=0)
        return dt_base

    @staticmethod
    def _mean_loss(loss_per: torch.Tensor, loss_mask=None):
        if loss_mask is None:
            return loss_per.mean()
        if not torch.is_tensor(loss_mask):
            loss_mask = torch.as_tensor(loss_mask, device=loss_per.device)
        else:
            loss_mask = loss_mask.to(loss_per.device)
        if loss_mask.dtype == torch.bool:
            return loss_per[loss_mask].mean()
        return loss_per[loss_mask.long()].mean()

    def _sample_shortcut_bootstrap(
        self,
        x1: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond: torch.Tensor,
    ):
        B = x1.shape[0]
        device = x1.device

        dt_base = self._sample_shortcut_dt_base_balanced(B, device)
        delta = torch.pow(2.0, -dt_base)
        dt_sections = torch.pow(2.0, dt_base)
        t = torch.floor(torch.rand(B, device=device) * dt_sections) / dt_sections
        noise = torch.randn_like(x1)
        x_t = (1.0 - (1.0 - 1e-5) * t.view(B, 1, 1)) * noise + t.view(B, 1, 1) * x1

        t_step = t
        half_dt_base = dt_base + 1.0
        half_delta = delta * 0.5
        t_mid = (t + half_delta).clamp_max(1.0)
        t_mid_step = t_mid

        target_model = self.model
        if self.shortcut_use_ema_target and self.ema_model_ref is not None:
            target_model = self.ema_model_ref

        target_was_training = target_model.training
        target_model.eval()
        try:
            with torch.no_grad():
                v1 = self._model_call(
                    target_model,
                    x_t,
                    t_step,
                    half_dt_base,
                    obs_tokens,
                    temporal_cond,
                    "50hz",
                )
                x_mid = x_t + half_delta.view(B, 1, 1) * v1
                v2 = self._model_call(
                    target_model,
                    x_mid,
                    t_mid_step,
                    half_dt_base,
                    obs_tokens,
                    temporal_cond,
                    "50hz",
                )
                target = 0.5 * (v1 + v2)
        finally:
            target_model.train(target_was_training)

        return x_t, t_step, dt_base, target

    def _compute_shortcut_loss_50hz(
        self,
        trajectory: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond: torch.Tensor,
        loss_mask=None,
    ):
        B = trajectory.shape[0]
        device = trajectory.device
        bootstrap_size = max(1, B // self.shortcut_bootstrap_every)
        x0 = trajectory[:bootstrap_size]
        obs_b = obs_tokens[:bootstrap_size]
        temporal_b = None if temporal_cond is None else temporal_cond[:bootstrap_size]

        dt_base = self._sample_shortcut_dt_base(bootstrap_size, device)
        delta = torch.pow(2.0, -dt_base)
        t_min = delta
        t = t_min + (1.0 - t_min) * torch.rand(bootstrap_size, device=device)
        noise = torch.randn_like(x0)
        x_t = t.view(bootstrap_size, 1, 1) * noise + (1.0 - t.view(bootstrap_size, 1, 1)) * x0

        t_step = (t * 1000).long()
        half_dt_base = dt_base + 1.0
        half_delta = delta * 0.5
        t_mid = (t - half_delta).clamp_min(0.0)
        t_mid_step = (t_mid * 1000).long()

        target_model = self.model
        if self.shortcut_use_ema_target and self.ema_model_ref is not None:
            target_model = self.ema_model_ref

        target_was_training = target_model.training
        target_model.eval()
        try:
            with torch.no_grad():
                v1 = self._model_call(
                    target_model,
                    x_t,
                    t_step,
                    half_dt_base,
                    obs_b,
                    temporal_b,
                    "50hz",
                )
                x_mid = x_t - half_delta.view(bootstrap_size, 1, 1) * v1
                v2 = self._model_call(
                    target_model,
                    x_mid,
                    t_mid_step,
                    half_dt_base,
                    obs_b,
                    temporal_b,
                    "50hz",
                )
                target = 0.5 * (v1 + v2)
        finally:
            target_model.train(target_was_training)

        pred = self._model_call(
            self.model,
            x_t,
            t_step,
            dt_base,
            obs_b,
            temporal_b,
            "50hz",
        )
        loss_all = F.mse_loss(pred, target.detach(), reduction="none")
        loss_per = reduce(loss_all, "b ... -> b (...)", "mean")
        return self._mean_loss(
            loss_per,
            None if loss_mask is None else loss_mask[:bootstrap_size],
        )

    def _compute_shortcut_one_step_loss_50hz(
        self,
        trajectory: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond: torch.Tensor,
        loss_mask=None,
    ):
        B = trajectory.shape[0]
        device = trajectory.device
        bootstrap_size = max(1, B // self.shortcut_bootstrap_every)
        bootstrap_size = min(bootstrap_size, B)

        x0 = trajectory[:bootstrap_size]
        obs_b = obs_tokens[:bootstrap_size]
        temporal_b = None if temporal_cond is None else temporal_cond[:bootstrap_size]

        dt_base = torch.zeros(bootstrap_size, dtype=torch.float32, device=device)
        delta = torch.ones_like(dt_base)
        t = torch.ones(bootstrap_size, dtype=torch.float32, device=device)
        noise = torch.randn_like(x0)
        x_t = t.view(bootstrap_size, 1, 1) * noise + (1.0 - t.view(bootstrap_size, 1, 1)) * x0

        t_step = (t * 1000).long()
        half_dt_base = dt_base + 1.0
        half_delta = delta * 0.5
        t_mid = (t - half_delta).clamp_min(0.0)
        t_mid_step = (t_mid * 1000).long()

        target_model = self.model
        if self.shortcut_use_ema_target and self.ema_model_ref is not None:
            target_model = self.ema_model_ref

        target_was_training = target_model.training
        target_model.eval()
        try:
            with torch.no_grad():
                v1 = self._model_call(
                    target_model,
                    x_t,
                    t_step,
                    half_dt_base,
                    obs_b,
                    temporal_b,
                    "50hz",
                )
                x_mid = x_t - half_delta.view(bootstrap_size, 1, 1) * v1
                v2 = self._model_call(
                    target_model,
                    x_mid,
                    t_mid_step,
                    half_dt_base,
                    obs_b,
                    temporal_b,
                    "50hz",
                )
                target = 0.5 * (v1 + v2)
        finally:
            target_model.train(target_was_training)

        pred = self._model_call(
            self.model,
            x_t,
            t_step,
            dt_base,
            obs_b,
            temporal_b,
            "50hz",
        )
        loss_all = F.mse_loss(pred, target.detach(), reduction="none")
        loss_per = reduce(loss_all, "b ... -> b (...)", "mean")
        return self._mean_loss(
            loss_per,
            None if loss_mask is None else loss_mask[:bootstrap_size],
        )

    def _compute_shortcut_merged_loss_50hz(
        self,
        trajectory: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond: torch.Tensor,
        loss_mask=None,
    ):
        B = trajectory.shape[0]
        bootstrap_size = max(1, B // self.shortcut_bootstrap_every)
        bootstrap_size = min(bootstrap_size, B)
        flow_size = B - bootstrap_size

        boot_x1 = trajectory[:bootstrap_size]
        boot_obs = obs_tokens[:bootstrap_size]
        boot_temporal = None if temporal_cond is None else temporal_cond[:bootstrap_size]
        boot_x_t, boot_timestep, boot_dt_base, boot_target = self._sample_shortcut_bootstrap(
            boot_x1,
            boot_obs,
            boot_temporal,
        )

        if flow_size > 0:
            flow_x1 = trajectory[bootstrap_size:]
            flow_obs = obs_tokens[bootstrap_size:]
            flow_temporal = None if temporal_cond is None else temporal_cond[bootstrap_size:]
            flow_x_t, flow_timestep, flow_dt_base, flow_target = self._sample_origin_flow(flow_x1)

            x_t = torch.cat([boot_x_t, flow_x_t], dim=0)
            timestep = torch.cat([boot_timestep, flow_timestep], dim=0)
            dt_base = torch.cat([boot_dt_base, flow_dt_base], dim=0)
            target = torch.cat([boot_target, flow_target], dim=0)
            obs_all = torch.cat([boot_obs, flow_obs], dim=0)
            if boot_temporal is None:
                temporal_all = None
            else:
                temporal_all = torch.cat([boot_temporal, flow_temporal], dim=0)
        else:
            x_t = boot_x_t
            timestep = boot_timestep
            dt_base = boot_dt_base
            target = boot_target
            obs_all = boot_obs
            temporal_all = boot_temporal

        pred = self._model_call(
            self.model,
            x_t,
            timestep,
            dt_base,
            obs_all,
            temporal_all,
            "50hz",
        )
        loss_all = F.mse_loss(pred, target.detach(), reduction="none")
        loss_per = reduce(loss_all, "b ... -> b (...)", "mean")

        merged_loss = self._mean_loss(loss_per, loss_mask)
        boot_loss = self._mean_loss(
            loss_per[:bootstrap_size],
            None if loss_mask is None else loss_mask[:bootstrap_size],
        )
        if flow_size > 0:
            flow_loss = self._mean_loss(
                loss_per[bootstrap_size:],
                None if loss_mask is None else loss_mask[bootstrap_size:],
            )
        else:
            flow_loss = merged_loss

        return merged_loss, boot_loss, flow_loss

    def compute_loss(self, batch, loss_mask=None):
        obs_tokens, nactions_10, nactions_50, tc_10, tc_50 = self._encode_batch(batch)
        loss_10 = self._compute_fm_loss_one_freq(
            nactions_10, obs_tokens, tc_10, "10hz", loss_mask
        )
        if self.shortcut_version == "old":
            loss_50 = self._compute_fm_loss_one_freq(
                nactions_50, obs_tokens, tc_50, "50hz", loss_mask
            )
            shortcut_50 = self._compute_shortcut_loss_50hz(
                nactions_50, obs_tokens, tc_50, loss_mask
            )
            shortcut_50_one_step = self._compute_shortcut_one_step_loss_50hz(
                nactions_50, obs_tokens, tc_50, loss_mask
            )
            total = (
                self.fm_10hz_weight * loss_10
                + self.fm_50hz_weight * loss_50
                + self.shortcut_50hz_weight * shortcut_50
                + self.shortcut_one_step_50hz_weight * shortcut_50_one_step
            )
            return total, {
                "fm_loss_10hz": loss_10.item(),
                "fm_loss_50hz": loss_50.item(),
                "shortcut_loss_50hz": shortcut_50.item(),
                "shortcut_loss_50hz_one_step": shortcut_50_one_step.item(),
                "total_loss": total.item(),
                "training_stage": 1.0,
            }

        loss_50, shortcut_50, flow_50 = self._compute_shortcut_merged_loss_50hz(
            nactions_50, obs_tokens, tc_50, loss_mask
        )
        total = self.fm_10hz_weight * loss_10 + self.fm_50hz_weight * loss_50
        return total, {
            "fm_loss_10hz": loss_10.item(),
            "fm_loss_50hz": loss_50.item(),
            "shortcut_loss_50hz": shortcut_50.item(),
            "shortcut_loss_50hz_bootstrap": shortcut_50.item(),
            "fm_loss_50hz_flow": flow_50.item(),
            "total_loss": total.item(),
            "training_stage": 1.0,
        }

    def conditional_sample(
        self,
        cond=None,
        temporal_cond=None,
        freq_mode: str = "10hz",
        generator=None,
        num_inference_steps: Optional[int] = None,
        noisy_trajectory=None,
        **kwargs,
    ):
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
        if freq_mode == "50hz":
            if steps < 1 or steps & (steps - 1) != 0:
                raise ValueError("50hz shortcut num_inference_steps must be a power of two")
            dt_base_value = int(torch.log2(torch.tensor(steps)).item())
            delta = 1.0 / float(steps)
            for i in range(steps):
                if self.shortcut_version == "new":
                    t_cur = i * delta
                else:
                    t_cur = 1.0 - i * delta
                t_tensor = torch.full(
                    (B,), t_cur, device=trajectory.device
                )
                if self.shortcut_version == "old":
                    t_tensor = (t_tensor * 1000).long()
                dt_base = torch.full(
                    (B,), dt_base_value, dtype=torch.float32, device=trajectory.device
                )
                v = self._model_call(
                    self.model, trajectory, t_tensor, dt_base, cond, temporal_cond, freq_mode
                )
                if self.shortcut_version == "new":
                    trajectory = trajectory + delta * v
                else:
                    trajectory = trajectory - delta * v
            return trajectory

        dt_base = torch.full(
            (B,), self.shortcut_dt_flow, dtype=torch.float32, device=trajectory.device
        )
        if self.shortcut_version == "new":
            delta = 1.0 / float(steps)
            for i in range(steps):
                t_cur = i * delta
                t_tensor = torch.full((B,), t_cur, device=trajectory.device)
                v = self._model_call(
                    self.model, trajectory, t_tensor, dt_base, cond, temporal_cond, freq_mode
                )
                trajectory = trajectory + delta * v
        else:
            t_schedule = torch.linspace(1.0, 0.0, steps + 1, device=trajectory.device)
            for i in range(steps):
                t_cur = t_schedule[i]
                dt = t_schedule[i + 1] - t_schedule[i]
                t_tensor = torch.full(
                    (B,), t_cur * 1000, device=trajectory.device
                ).long()
                v = self._model_call(
                    self.model, trajectory, t_tensor, dt_base, cond, temporal_cond, freq_mode
                )
                trajectory = trajectory + dt * v
        return trajectory

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

        assert extended_obs_dict is not None, (
            "extended_obs is required for RNN temporal condition in "
            "FMMultiFreqShortcutPolicy"
        )
        temporal_cond_10hz = self.get_temporal_cond(extended_obs_dict, head="10hz")
        temporal_cond_50hz = self.get_temporal_cond(extended_obs_dict, head="50hz")

        nsample_10 = self.conditional_sample(
            cond=obs_tokens,
            temporal_cond=temporal_cond_10hz,
            freq_mode="10hz",
            num_inference_steps=self.num_inference_steps_10hz,
        )
        steps_50 = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        nsample_50 = self.conditional_sample(
            cond=obs_tokens,
            temporal_cond=temporal_cond_50hz,
            freq_mode="50hz",
            num_inference_steps=steps_50,
        )

        assert nsample_10.shape == (B, temporal_cond_10hz.shape[1], self.action_dim)
        assert nsample_50.shape == (B, temporal_cond_50hz.shape[1], self.action_dim)
        action_pred_10hz = self.normalizer["action"].unnormalize(nsample_10)
        action_pred_50hz = self.normalizer["action_stack"].unnormalize(nsample_50)

        start_10 = self.action_start_10hz
        end_10 = start_10 + self.n_action_steps
        action_10hz = action_pred_10hz[:, start_10:end_10]

        start_50 = self.action_start_50hz
        end_50 = self.horizon_50hz
        action_50hz = action_pred_50hz[:, start_50:end_50]

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

    def predict_action(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_action(*args, **kwargs)

    def forward(self, batch, loss_mask=None):
        return self.compute_loss(batch, loss_mask)
