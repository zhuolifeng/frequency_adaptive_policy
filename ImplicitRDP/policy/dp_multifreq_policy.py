"""
Multi-frequency diffusion policy.

This policy trains one standard diffusion model on both 10Hz and 50Hz action
chunks. The model output follows ``noise_scheduler.config.prediction_type``:
epsilon, sample/x0, or v_prediction.
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from einops import reduce
from torch.cuda.amp import autocast

from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.model.common.normalizer import LinearNormalizer
from ImplicitRDP.model.diffusion.transformer_for_dp_multifreq import (
    TransformerForDPMultiFreq,
)
from ImplicitRDP.model.force.rnn import RNN
from ImplicitRDP.policy.base_image_policy import BaseImagePolicy


class DPMultiFreqPolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDIMScheduler,
        obs_encoder: torch.nn.Module,
        model: TransformerForDPMultiFreq,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: int = 5,
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
        training_stage: int = 1,
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
        self.noise_scheduler = noise_scheduler
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
        self.action_start_10hz = action_start_10hz
        self.action_start_50hz = action_start_50hz
        self.inference_head = inference_head
        self.training_stage = 1
        self.kwargs = kwargs

        assert self.inference_head in ("10hz", "50hz")
        if training_stage != 1:
            raise ValueError("DPMultiFreqPolicy is single-stage; use training_stage=1.")

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

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def set_training_stage(self, stage: int):
        if stage != 1:
            raise ValueError("DPMultiFreqPolicy is single-stage; only stage=1 is supported.")
        self.training_stage = 1

    def update_progress(self, progress: float):
        self.progress = progress

    def get_temporal_cond(
        self,
        extended_obs_dict,
        head: str,
        extended_obs_last_step: Optional[int] = None,
        extend_obs_pad_after_n: Optional[int] = None,
    ):
        if extended_obs_dict is None:
            return None
        if head == "10hz":
            return self.get_temporal_cond_lowfreq(
                extended_obs_dict,
                extended_obs_last_step=extended_obs_last_step,
                extend_obs_pad_after_n=extend_obs_pad_after_n,
            )
        if head == "50hz":
            return self.get_temporal_cond_highfreq(
                extended_obs_dict,
                extended_obs_last_step=extended_obs_last_step,
                extend_obs_pad_after_n=extend_obs_pad_after_n,
            )
        raise ValueError(f"Unsupported head={head}")

    def get_temporal_cond_highfreq(
        self,
        extended_obs_dict,
        extended_obs_last_step: Optional[int] = None,
        extend_obs_pad_after_n: Optional[int] = None,
    ):
        key = "right_robot_tcp_wrench_stack"
        if extended_obs_last_step is not None:
            extended_obs = extended_obs_dict[key][..., -extended_obs_last_step:, :]
        else:
            extended_obs = extended_obs_dict[key]
        if extend_obs_pad_after_n is not None:
            padding_obs = extended_obs_dict[key][..., -1:, :].repeat(
                1, extend_obs_pad_after_n, 1
            )
            extended_obs = torch.cat([padding_obs, extended_obs], dim=-2)
        extended_obs = self.normalizer[f"extended_{key}"].normalize(extended_obs)
        return self.rnn_highfreq(extended_obs)

    def get_temporal_cond_lowfreq(
        self,
        extended_obs_dict,
        extended_obs_last_step: Optional[int] = None,
        extend_obs_pad_after_n: Optional[int] = None,
    ):
        key = "right_robot_tcp_wrench"
        if extended_obs_last_step is not None:
            extended_obs = extended_obs_dict[key][..., -extended_obs_last_step:, :]
        else:
            extended_obs = extended_obs_dict[key]
        if extend_obs_pad_after_n is not None:
            padding_obs = extended_obs_dict[key][..., -1:, :].repeat(
                1, extend_obs_pad_after_n, 1
            )
            extended_obs = torch.cat([padding_obs, extended_obs], dim=-2)
        extended_obs = self.normalizer[f"extended_{key}"].normalize(extended_obs)
        return self.rnn_lowfreq(extended_obs)

    def _encode_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])
        return obs_tokens

    def predict_obs_tokens(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            obs_tokens = self._encode_obs(obs_dict)
            return obs_tokens.reshape(obs_tokens.shape[0], -1)

    def _get_variance_terms(self, timesteps: torch.Tensor, sample: torch.Tensor):
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=sample.device, dtype=sample.dtype
        )
        alpha_prod = alphas_cumprod[timesteps].view(-1, 1, 1)
        beta_prod = 1.0 - alpha_prod
        return alpha_prod, beta_prod

    def _prediction_to_x0_eps(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        alpha_prod, beta_prod = self._get_variance_terms(timesteps, sample)
        sqrt_alpha = alpha_prod.sqrt()
        sqrt_beta = beta_prod.sqrt()
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            pred_eps = model_output
            pred_x0 = (sample - sqrt_beta * pred_eps) / sqrt_alpha.clamp_min(1e-12)
        elif pred_type == "sample":
            pred_x0 = model_output
            pred_eps = (sample - sqrt_alpha * pred_x0) / sqrt_beta.clamp_min(1e-12)
        elif pred_type == "v_prediction":
            pred_x0 = sqrt_alpha * sample - sqrt_beta * model_output
            pred_eps = sqrt_alpha * model_output + sqrt_beta * sample
        else:
            raise ValueError(f"Unsupported prediction_type={pred_type}")
        return pred_x0, pred_eps

    def _ddim_jump_to(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        target_timestep: torch.Tensor,
    ):
        pred_x0, pred_eps = self._prediction_to_x0_eps(sample, model_output, timestep)
        if self.noise_scheduler.config.clip_sample:
            clip_range = getattr(self.noise_scheduler.config, "clip_sample_range", 1.0)
            pred_x0 = pred_x0.clamp(
                -clip_range,
                clip_range,
            )
            _, beta_prod = self._get_variance_terms(timestep, sample)
            alpha_prod = 1.0 - beta_prod
            pred_eps = (sample - alpha_prod.sqrt() * pred_x0) / beta_prod.sqrt().clamp_min(1e-12)
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=sample.device, dtype=sample.dtype
        )
        target_timestep = target_timestep.clamp(
            0, self.noise_scheduler.config.num_train_timesteps - 1
        ).long()
        alpha_target_flat = alphas_cumprod[target_timestep]
        alpha_target_flat = torch.where(
            target_timestep == 0,
            torch.ones_like(alpha_target_flat),
            alpha_target_flat,
        )
        alpha_target = alpha_target_flat.view(-1, 1, 1)
        beta_target = 1.0 - alpha_target
        return alpha_target.sqrt() * pred_x0 + beta_target.sqrt() * pred_eps

    def _prediction_target(
        self,
        trajectory: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            return noise
        if pred_type == "sample":
            return trajectory
        if pred_type == "v_prediction":
            return self.noise_scheduler.get_velocity(trajectory, noise, timesteps)
        raise ValueError(f"Unsupported prediction_type={pred_type}")

    def _sample_diffusion_batch(self, trajectory: torch.Tensor):
        B = trajectory.shape[0]
        device = trajectory.device
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=device,
        ).long()
        noisy = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        target = self._prediction_target(trajectory, noise, timesteps)
        return noisy, timesteps, target

    def _compute_dp_loss_one_freq(
        self,
        model,
        trajectory: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond: torch.Tensor,
        freq_mode: str,
        loss_mask=None,
    ):
        noisy, timesteps, target = self._sample_diffusion_batch(trajectory)
        pred = model(
            noisy,
            timesteps,
            cond=obs_tokens,
            temporal_cond=temporal_cond,
            freq_mode=freq_mode,
        )
        loss_all = F.mse_loss(pred, target, reduction="none")
        loss_per = reduce(loss_all, "b ... -> b (...)", "mean")
        if loss_mask is None:
            return loss_per.mean()
        loss_per = loss_per[loss_mask]
        return loss_per.mean()

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
        return obs_tokens, nactions_10, nactions_50, temporal_cond_10hz, temporal_cond_50hz

    def compute_loss(self, batch, loss_mask=None):
        obs_tokens, nactions_10, nactions_50, tc_10, tc_50 = self.model_forward(batch)
        loss_10 = self._compute_dp_loss_one_freq(
            self.model, nactions_10, obs_tokens, tc_10, "10hz", loss_mask
        )
        loss_50 = self._compute_dp_loss_one_freq(
            self.model, nactions_50, obs_tokens, tc_50, "50hz", loss_mask
        )
        total = loss_10 + loss_50
        return total, {
            "dp_loss_10hz": loss_10.item(),
            "dp_loss_50hz": loss_50.item(),
            "total_loss": total.item(),
            "training_stage": 1.0,
        }

    def forward(self, batch, loss_mask=None):
        return self.compute_loss(batch, loss_mask)

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
        if noisy_trajectory is None:
            trajectory = torch.randn(
                size=(cond.shape[0], horizon, self.action_dim),
                dtype=cond.dtype,
                device=cond.device,
                generator=generator,
            )
        else:
            trajectory = noisy_trajectory

        self.noise_scheduler.set_timesteps(steps, device=trajectory.device)
        timesteps = self.noise_scheduler.timesteps
        for i, t in enumerate(timesteps):
            t_batch = torch.full(
                (trajectory.shape[0],),
                int(t.item()),
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
        horizon = self.horizon_50hz if freq_mode == "50hz" else self.horizon_10hz
        return self._ddim_sample(
            self.model,
            cond=cond,
            temporal_cond=temporal_cond,
            freq_mode=freq_mode,
            horizon=horizon,
            generator=generator,
            num_inference_steps=num_inference_steps,
            noisy_trajectory=noisy_trajectory,
        )

    def _predict_from_obs_tokens_and_noisy_trajectory(
        self,
        obs_tokens_and_noisy_trajectory_list: List[torch.Tensor],
        extended_obs_dict: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
        extended_obs_last_step_list: List[int],
        dataset_obs_temporal_downsample_ratio: int,
        extend_obs_pad_after: bool = False,
        head: str = "10hz",
    ) -> List[Dict[str, torch.Tensor]]:
        obs_tokens_and_noisy_trajectory = torch.cat(
            obs_tokens_and_noisy_trajectory_list, dim=0
        )
        B = obs_tokens_and_noisy_trajectory.shape[0]
        target_horizon = self.horizon_50hz if head == "50hz" else self.horizon_10hz
        obs_tokens = obs_tokens_and_noisy_trajectory[:, :-target_horizon * self.action_dim]
        noisy_trajectory = obs_tokens_and_noisy_trajectory[:, -target_horizon * self.action_dim:]
        obs_tokens = obs_tokens.to(self.device).reshape(B, -1, self.n_emb)
        noisy_trajectory = noisy_trajectory.to(self.device).reshape(
            B, target_horizon, self.action_dim
        )

        max_extended_obs_last_step = max(extended_obs_last_step_list)
        assert extended_obs_dict is not None, "extended_obs is required for RNNObsEncoder"
        extend_obs_pad_after_n = (
            target_horizon - max_extended_obs_last_step if extend_obs_pad_after else None
        )

        if isinstance(extended_obs_dict, list):
            temporal_cond = [
                self.get_temporal_cond(
                    item,
                    head=head,
                    extended_obs_last_step=max_extended_obs_last_step,
                    extend_obs_pad_after_n=extend_obs_pad_after_n,
                )
                for item in extended_obs_dict
            ]
        else:
            temporal_cond = self.get_temporal_cond(
                extended_obs_dict,
                head=head,
                extended_obs_last_step=max_extended_obs_last_step,
                extend_obs_pad_after_n=extend_obs_pad_after_n,
            )

        temporal_cond_list = []
        for i in range(B):
            source = temporal_cond[i] if isinstance(temporal_cond, list) else temporal_cond
            if extended_obs_last_step_list[i] < max_extended_obs_last_step:
                padding_obs = source[:, -1:, :].repeat(
                    1, max_extended_obs_last_step - extended_obs_last_step_list[i], 1
                )
                temporal_cond_list.append(
                    torch.cat(
                        [source[:, -extended_obs_last_step_list[i]:, :], padding_obs],
                        dim=-2,
                    )
                )
            else:
                temporal_cond_list.append(source)
        temporal_cond = torch.cat(temporal_cond_list, dim=0)
        noisy_trajectory = noisy_trajectory[:, :temporal_cond.shape[1], :]

        normalizer_key = "action_stack" if head == "50hz" else "action"
        nsample = self.conditional_sample(
            cond=obs_tokens,
            temporal_cond=temporal_cond,
            freq_mode=head,
            noisy_trajectory=noisy_trajectory,
        )
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
                    "action": action[i: i + 1],
                    "action_pred": action_pred[i: i + 1],
                })
            else:
                valid_len = max(
                    0, min(extended_obs_last_step_list[i] - start, end - start)
                )
                results.append({
                    "action": action[i: i + 1, :valid_len, :],
                    "action_pred": action_pred[
                        i: i + 1, :extended_obs_last_step_list[i], :
                    ],
                })
        return results

    def predict_from_obs_tokens_and_noisy_trajectory(self, *args, **kwargs):
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_from_obs_tokens_and_noisy_trajectory(*args, **kwargs)

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
        assert extended_obs_dict is not None, "extended_obs is required for RNN temporal condition"

        tc_10 = self.get_temporal_cond(extended_obs_dict, head="10hz")
        tc_50 = self.get_temporal_cond(extended_obs_dict, head="50hz")

        nsample_10 = self.conditional_sample(
            cond=obs_tokens,
            temporal_cond=tc_10,
            freq_mode="10hz",
            num_inference_steps=num_inference_steps,
        )
        nsample_50 = self.conditional_sample(
            cond=obs_tokens,
            temporal_cond=tc_50,
            freq_mode="50hz",
            num_inference_steps=num_inference_steps,
        )

        assert nsample_10.shape == (B, tc_10.shape[1], self.action_dim)
        assert nsample_50.shape == (B, tc_50.shape[1], self.action_dim)

        action_pred_10hz = self.normalizer["action"].unnormalize(nsample_10)
        action_pred_50hz = self.normalizer["action_stack"].unnormalize(nsample_50)

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

    def predict_action(self, *args, **kwargs):
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_action(*args, **kwargs)

    def get_optimizer(
        self,
        lr: float,
        weight_decay: float,
        obs_encoder_lr: Optional[float] = None,
        obs_encoder_weight_decay: Optional[float] = None,
        betas: Tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.Optimizer:
        optim_groups = []
        optim_groups.extend(self.model.get_optim_groups(weight_decay=weight_decay))

        if self.rnn_lowfreq is not None:
            optim_groups += self.rnn_lowfreq.get_optim_groups(weight_decay=weight_decay)
        if self.rnn_highfreq is not None:
            optim_groups += self.rnn_highfreq.get_optim_groups(weight_decay=weight_decay)

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

        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
