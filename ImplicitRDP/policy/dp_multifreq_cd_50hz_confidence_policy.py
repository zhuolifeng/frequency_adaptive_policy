"""
Three-stage multi-frequency DP policy with confidence and 50Hz CD.

Stage 1 trains the standard 10Hz + 50Hz DP objective.
Stage 2 freezes the policy and trains a confidence head from 10Hz sample MSE.
Stage 3 freezes observation/temporal/confidence modules and trains the DP
backbone with 10Hz DP loss plus 50Hz consistency distillation.
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from einops import reduce
from loguru import logger
from torch.cuda.amp import autocast

from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.model.common.normalizer import LinearNormalizer
from ImplicitRDP.model.confidence.transformer import TransformerConfidenceHead
from ImplicitRDP.model.diffusion.transformer_for_dp_multifreq import TransformerForDPMultiFreq
from ImplicitRDP.model.force.rnn import RNN
from ImplicitRDP.policy.base_image_policy import BaseImagePolicy


class DPMultiFreqCD50HzConfidencePolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDIMScheduler,
        obs_encoder: torch.nn.Module,
        model: TransformerForDPMultiFreq,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps_10hz: Optional[int] = None,
        num_inference_steps_50hz: Optional[int] = None,
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
        cd_timestep_grid: List[int] = (99, 79, 59, 39, 19, 0),
        cd_teacher_rollout_steps: int = 1,
        cd_weight: float = 1.0,
        dsm_weight: float = 1.0,
        loss_10hz_weight: float = 1.0,
        loss_type: str = "eps",
        use_confidence_head: bool = True,
        confidence_hidden_dim: Optional[int] = None,
        confidence_gamma: float = 1.0,
        confidence_alpha: float = 0.2,
        confidence_obs_encoder_n_layers: int = 2,
        confidence_obs_encoder_hidden_dim: Optional[int] = None,
        confidence_transformer_n_heads: int = 8,
        confidence_transformer_dropout: float = 0.1,
        confidence_action_dims: int = 6,
        confidence_n_samples: int = 8,
        confidence_num_inference_steps: int = 5,
        use_confidence_based_head_selection: bool = False,
        confidence_threshold_for_head_selection: float = 0.0,
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
        self.num_inference_steps_10hz = int(num_inference_steps_10hz)
        self.num_inference_steps_50hz = int(num_inference_steps_50hz)
        self.action_start_10hz = action_start_10hz
        self.action_start_50hz = action_start_50hz
        self.inference_head = inference_head
        self.kwargs = kwargs
        assert self.inference_head in ("10hz", "50hz")

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
        self.loss_type = (loss_type or "eps").lower()
        if self.loss_type == "epsilon":
            self.loss_type = "eps"
        if self.loss_type in ("v", "v_prediction"):
            self.loss_type = "velocity"
        if self.loss_type not in ("sample", "eps", "velocity"):
            raise ValueError(
                f"Unsupported loss_type={loss_type}; expected sample, eps, or velocity."
            )

        self.confidence_gamma = confidence_gamma
        self.confidence_alpha = confidence_alpha
        self.use_confidence_head = use_confidence_head
        self.confidence_action_dims = confidence_action_dims
        self.confidence_n_samples = int(confidence_n_samples)
        self.confidence_num_inference_steps = int(confidence_num_inference_steps)
        self.use_confidence_based_head_selection = use_confidence_based_head_selection
        self.confidence_threshold_for_head_selection = float(confidence_threshold_for_head_selection)

        self.confidence_head = None
        if self.use_confidence_head:
            hidden_dim = confidence_hidden_dim or n_emb
            self.confidence_head = TransformerConfidenceHead(
                n_emb=n_emb,
                hidden_dim=hidden_dim,
                obs_encoder_n_layers=confidence_obs_encoder_n_layers,
                obs_encoder_hidden_dim=confidence_obs_encoder_hidden_dim,
                n_heads=confidence_transformer_n_heads,
                dropout=confidence_transformer_dropout
            )

        self.training_stage = int(training_stage)
        self.set_training_stage(self.training_stage)
        self.progress = 0.0

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
        if stage not in (1, 2, 3):
            raise ValueError(f"Invalid training_stage={stage}; expected 1, 2, or 3.")
        self.training_stage = int(stage)

    def update_progress(self, progress: float):
        self.progress = progress

    def _set_module_trainable(self, module, trainable: bool):
        if module is None:
            return
        module.requires_grad_(trainable)
        module.train(trainable)

    def apply_stage_freeze(self):
        if self.training_stage == 1:
            self._set_module_trainable(self.obs_encoder, True)
            self._set_module_trainable(self.rnn_lowfreq, True)
            self._set_module_trainable(self.rnn_highfreq, True)
            self._set_module_trainable(self.model, True)
            self._set_module_trainable(self.confidence_head, False)
        elif self.training_stage == 2:
            self._set_module_trainable(self.obs_encoder, False)
            self._set_module_trainable(self.rnn_lowfreq, False)
            self._set_module_trainable(self.rnn_highfreq, False)
            self._set_module_trainable(self.model, False)
            self._set_module_trainable(self.confidence_head, True)
        else:
            self._set_module_trainable(self.obs_encoder, False)
            self._set_module_trainable(self.rnn_lowfreq, False)
            self._set_module_trainable(self.rnn_highfreq, False)
            self._set_module_trainable(self.confidence_head, False)
            self._set_module_trainable(self.model, True)

    def get_temporal_cond(
        self,
        extended_obs_dict,
        head: str,
        extended_obs_last_step: Optional[int] = None,
        extend_obs_pad_after_n: Optional[int] = None,
        allow_partial: bool = False,
        post_process: bool = False,
    ):
        if extended_obs_dict is None:
            return None
        if head == "10hz":
            return self.get_temporal_cond_lowfreq(
                extended_obs_dict,
                extended_obs_last_step=extended_obs_last_step,
                extend_obs_pad_after_n=extend_obs_pad_after_n,
                allow_partial=allow_partial,
                post_process=post_process,
            )
        if head == "50hz":
            return self.get_temporal_cond_highfreq(
                extended_obs_dict,
                extended_obs_last_step=extended_obs_last_step,
                extend_obs_pad_after_n=extend_obs_pad_after_n,
                allow_partial=allow_partial,
                post_process=post_process,
            )
        raise ValueError(f"Unsupported head={head}")

    def get_temporal_cond_highfreq(
        self,
        extended_obs_dict,
        extended_obs_last_step: Optional[int] = None,
        extend_obs_pad_after_n: Optional[int] = None,
        allow_partial: bool = False,
        post_process: bool = False,
    ):
        if extended_obs_dict is None or len(self.extended_obs_keys) == 0:
            return None
        extended_obs_keys = ["right_robot_tcp_wrench_stack"]

        temporal_cond = []
        for extended_obs_key in extended_obs_keys:
            if extended_obs_key not in extended_obs_dict:
                if allow_partial:
                    continue
                raise KeyError(f"Missing extended obs key: {extended_obs_key}")
            
            extended_obs = extended_obs_dict[extended_obs_key]
            obs_dim = len(self.shape_meta['extended_obs'][extended_obs_key]['shape'])
            
            if extended_obs_last_step is not None:
                if obs_dim == 1:
                    extended_obs = extended_obs[..., -extended_obs_last_step:, :]
                elif obs_dim == 2:
                    extended_obs = extended_obs[..., -extended_obs_last_step:, :, :]
                else:
                    raise ValueError("Only support 1d or 2d extended obs")
            
            if extend_obs_pad_after_n is not None:
                if obs_dim == 1:
                    padding_obs = extended_obs_dict[extended_obs_key][..., -1:, :].repeat(1, extend_obs_pad_after_n, 1)
                elif obs_dim == 2:
                    padding_obs = extended_obs_dict[extended_obs_key][..., -1:, :, :].repeat(1, extend_obs_pad_after_n, 1, 1)
                else:
                    raise ValueError("Only support 1d or 2d extended obs")
                extended_obs = torch.cat([padding_obs, extended_obs], dim=-2)
            
            extended_obs = self.normalizer[f"extended_{extended_obs_key}"].normalize(extended_obs)
            temporal_cond.append(extended_obs)
        
        if len(temporal_cond) == 0:
            return None
        
        temporal_cond = torch.cat(temporal_cond, dim=-1)

        if post_process and self.rnn_highfreq is not None:
            temporal_cond = self.rnn_highfreq(temporal_cond)

        return temporal_cond

    def get_temporal_cond_lowfreq(
        self,
        extended_obs_dict,
        extended_obs_last_step: Optional[int] = None,
        extend_obs_pad_after_n: Optional[int] = None,
        allow_partial: bool = False,
        post_process: bool = False,
    ):
        if extended_obs_dict is None or len(self.extended_obs_keys) == 0:
            return None
        extended_obs_keys = ["right_robot_tcp_wrench"]

        temporal_cond = []
        for extended_obs_key in extended_obs_keys:
            if extended_obs_key not in extended_obs_dict:
                if allow_partial:
                    continue
                raise KeyError(f"Missing extended obs key: {extended_obs_key}")

            extended_obs = extended_obs_dict[extended_obs_key]
            obs_dim = len(self.shape_meta['extended_obs'][extended_obs_key]['shape'])

            if extended_obs_last_step is not None:
                if obs_dim == 1:
                    extended_obs = extended_obs[..., -extended_obs_last_step:, :]
                elif obs_dim == 2:
                    extended_obs = extended_obs[..., -extended_obs_last_step:, :, :]
                else:
                    raise ValueError("Only support 1d or 2d extended obs")

            if extend_obs_pad_after_n is not None:
                if obs_dim == 1:
                    padding_obs = extended_obs_dict[extended_obs_key][..., -1:, :].repeat(1, extend_obs_pad_after_n, 1)
                elif obs_dim == 2:
                    padding_obs = extended_obs_dict[extended_obs_key][..., -1:, :, :].repeat(1, extend_obs_pad_after_n, 1, 1)
                else:
                    raise ValueError("Only support 1d or 2d extended obs")
                extended_obs = torch.cat([padding_obs, extended_obs], dim=-2)

            extended_obs = self.normalizer[f"extended_{extended_obs_key}"].normalize(extended_obs)
            temporal_cond.append(extended_obs)

        if len(temporal_cond) == 0:
            return None

        temporal_cond = torch.cat(temporal_cond, dim=-1)

        if post_process and self.rnn_lowfreq is not None:
            temporal_cond = self.rnn_lowfreq(temporal_cond)

        return temporal_cond

    def _predict_obs_tokens(self,
                            obs_dict: Dict[str, torch.Tensor],
                            extended_obs_dict: Optional[Dict[str, torch.Tensor]] = None,
                            return_expanded_obs_tokens: bool = False,
                            return_temporal_cond: bool = False,
                            head: str = "10hz"
                            ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))

        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])

        temporal_cond = None
        if extended_obs_dict is not None:
            temporal_cond = self.get_temporal_cond(extended_obs_dict, head=head, allow_partial=True, post_process=True)

        if return_temporal_cond:
            return obs_tokens, temporal_cond

        if return_expanded_obs_tokens:
            obs_flat = obs_tokens.reshape(B, -1)
            obs_flat = obs_flat.unsqueeze(1).expand(-1, self.action_horizon, -1)
            return obs_flat
        return obs_tokens

    def predict_obs_tokens(self, *args, **kwargs) -> torch.Tensor:
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_obs_tokens(*args, **kwargs)

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
        if self.loss_type == "velocity":
            pred_velocity = self._sample_prediction_to_velocity(
                sample, model_output, timesteps
            )
            pred_x0 = sqrt_alpha * sample - sqrt_beta * pred_velocity
            pred_eps = sqrt_alpha * pred_velocity + sqrt_beta * sample
        else:
            pred_x0 = model_output
            pred_eps = (sample - sqrt_alpha * pred_x0) / sqrt_beta.clamp_min(1e-12)
        return pred_x0, pred_eps

    def _sample_prediction_to_velocity(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        alpha_prod, beta_prod = self._get_variance_terms(timesteps, sample)
        sqrt_alpha = alpha_prod.sqrt()
        sqrt_beta = beta_prod.sqrt().clamp_min(1e-12)
        return (sqrt_alpha * sample - model_output) / sqrt_beta

    def _velocity_loss_from_sample_prediction(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        trajectory: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        pred_velocity = self._sample_prediction_to_velocity(
            sample, model_output, timesteps
        )
        target_velocity = self.noise_scheduler.get_velocity(
            trajectory, noise, timesteps
        )
        return F.mse_loss(pred_velocity, target_velocity, reduction="none")

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
            pred_x0 = pred_x0.clamp(-clip_range, clip_range)
            _, beta_prod = self._get_variance_terms(timestep, sample)
            alpha_prod = 1.0 - beta_prod
            pred_eps = (
                sample - alpha_prod.sqrt() * pred_x0
            ) / beta_prod.sqrt().clamp_min(1e-12)

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

    def _compute_dp_loss_one_freq(
        self,
        trajectory: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond: Optional[torch.Tensor],
        freq_mode: str,
        loss_mask=None,
    ):
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
        pred = self.model(
            noisy,
            timesteps,
            cond=obs_tokens,
            temporal_cond=temporal_cond,
            freq_mode=freq_mode,
        )
        if self.loss_type == "sample":
            loss_all = F.mse_loss(pred, trajectory, reduction="none")
        elif self.loss_type == "eps":
            _, pred_eps = self._prediction_to_x0_eps(noisy, pred, timesteps)
            loss_all = F.mse_loss(pred_eps, noise, reduction="none")
        elif self.loss_type == "velocity":
            loss_all = self._velocity_loss_from_sample_prediction(
                noisy,
                pred,
                trajectory,
                noise,
                timesteps,
            )
        else:
            raise ValueError(f"Unsupported loss_type={self.loss_type}")
        loss_per = reduce(loss_all, "b ... -> b (...)", "mean")
        if loss_mask is not None:
            loss_per = loss_per[loss_mask]
        return loss_per.mean()

    def _prepare_multifreq_inputs(self, batch):
        nactions_10 = self.normalizer["action"].normalize(batch["action"])
        nactions_50 = self.normalizer["action_stack"].normalize(batch["action_stack"])

        if self.rnn_lowfreq is not None and self.rnn_highfreq is not None:
            assert "extended_obs" in batch, "extended_obs is required for RNNObsEncoder"
            tc_10 = self.get_temporal_cond(batch["extended_obs"], head="10hz", allow_partial=True, post_process=True)
            tc_50 = self.get_temporal_cond(batch["extended_obs"], head="50hz", allow_partial=True, post_process=True)
        else:
            tc_10 = None
            tc_50 = None

        obs_tokens = self._predict_obs_tokens(batch["obs"])

        return obs_tokens, nactions_10, nactions_50, tc_10, tc_50

    def compute_stage1_loss(self, batch, loss_mask=None):
        obs_tokens, nactions_10, nactions_50, tc_10, tc_50 = self._prepare_multifreq_inputs(batch)
        loss_10 = self._compute_dp_loss_one_freq(
            nactions_10, obs_tokens, tc_10, "10hz", loss_mask
        )
        loss_50 = self._compute_dp_loss_one_freq(
            nactions_50, obs_tokens, tc_50, "50hz", loss_mask
        )
        total = loss_10 + loss_50
        return total, {
            "dp_loss_10hz": loss_10.item(),
            "dp_loss_50hz": loss_50.item(),
            "total_loss": total.item(),
            "training_stage": 1.0,
        }

    def _compute_confidence_logits(
        self, obs_tokens: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self.confidence_head is None:
            return None
        return self.confidence_head(obs_tokens)

    def _compute_confidence_loss(
        self,
        obs_tokens: torch.Tensor,
        mse_vec_conf: torch.Tensor,
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
        mse_masked = (
            mse_vec_conf[loss_mask]
            if loss_mask is not None
            else mse_vec_conf
        )

        sigma = 1.0 + torch.exp(log_sigma)
        log_sigma_term = torch.log(sigma)

        gamma = self.confidence_gamma
        alpha = self.confidence_alpha
        conf_loss = gamma * mse_masked * sigma - alpha * log_sigma_term
        conf_loss = conf_loss.mean()

        return conf_loss, log_sigma, sigma, mse_masked

    def _sample_cd_timestep_pair(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid = torch.tensor(self.cd_timestep_grid, dtype=torch.long, device=device)
        pair_idx = torch.randint(
            0,
            len(self.cd_timestep_grid) - 1,
            (batch_size,),
            device=device,
        )
        return grid[pair_idx], grid[pair_idx + 1]

    def _teacher_rollout_to_timestep(
        self,
        teacher_policy: "DPMultiFreqCD50HzConfidencePolicy",
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

    def _compute_10hz_sample_mse(
        self,
        nactions_10: torch.Tensor,
        obs_tokens: torch.Tensor,
        tc_10: Optional[torch.Tensor],
        loss_mask=None,
    ):
        batch_size = nactions_10.shape[0]
        device = nactions_10.device
        n_samples = max(1, self.confidence_n_samples)
        accum = torch.zeros_like(nactions_10)
        with torch.no_grad():
            for _ in range(n_samples):
                nsample = self.conditional_sample(
                    cond=obs_tokens,
                    temporal_cond=tc_10,
                    freq_mode="10hz",
                    num_inference_steps=self.confidence_num_inference_steps,
                )
                accum += F.mse_loss(nsample, nactions_10, reduction="none")
        per_dim = accum / n_samples
        full_vec = reduce(per_dim, "b t a -> b", "mean")
        conf_dims = min(self.confidence_action_dims, per_dim.shape[-1])
        conf_loss_all = per_dim[..., :conf_dims]
        conf_vec = reduce(conf_loss_all, "b t a -> b", "mean")
        if per_dim.shape[-1] > conf_dims:
            virtual_vec = reduce(per_dim[..., conf_dims:], "b t a -> b", "mean")
        else:
            virtual_vec = torch.zeros(batch_size, device=device, dtype=per_dim.dtype)
        if loss_mask is None:
            loss_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        masked = full_vec[loss_mask]
        return masked.mean(), full_vec, conf_vec, virtual_vec

    def compute_confidence_loss(self, batch, loss_mask=None):
        obs_tokens = self._predict_obs_tokens(batch["obs"])
        nactions_10 = self.normalizer["action"].normalize(batch["action"])
        tc_10 = self.get_temporal_cond(batch["extended_obs"], head="10hz", allow_partial=True, post_process=True)

        mse, mse_vec, mse_vec_conf, mse_vec_virtual = self._compute_10hz_sample_mse(
            nactions_10, obs_tokens, tc_10, loss_mask
        )
        conf_loss, log_sigma, sigma, mse_masked = self._compute_confidence_loss(
            obs_tokens, mse_vec_conf, loss_mask
        )
        total = conf_loss if conf_loss is not None else mse
        loss_dict = {
            "dp_10hz_sample_mse": mse.item(),
            "dp_10hz_sample_mse_front6": mse_vec_conf.mean().item(),
            "dp_10hz_sample_mse_virtual": mse_vec_virtual.mean().item(),
            "total_loss": total.item(),
            "training_stage": 2.0,
        }
        if log_sigma is not None and sigma is not None:
            loss_dict.update({
                "confidence/log_sigma": log_sigma.mean().item(),
                "confidence/log_sigma_min": log_sigma.min().item(),
                "confidence/log_sigma_max": log_sigma.max().item(),
                "confidence/sigma": sigma.mean().item(),
                "confidence/dp_origin_loss": mse_masked.mean().item(),
            })
            if conf_loss is not None:
                loss_dict["confidence/conf_loss"] = conf_loss.item()
        return total, loss_dict

    def compute_stage3_loss(
        self,
        batch,
        target_policy: Optional["DPMultiFreqCD50HzConfidencePolicy"] = None,
        teacher_policy: Optional["DPMultiFreqCD50HzConfidencePolicy"] = None,
        loss_mask=None,
    ):
        if target_policy is None:
            target_policy = self
        if teacher_policy is None:
            teacher_policy = self
        obs_tokens, nactions_10, nactions_50, tc_10, tc_50 = self._prepare_multifreq_inputs(batch)
        loss_10 = self._compute_dp_loss_one_freq(
            nactions_10, obs_tokens, tc_10, "10hz", loss_mask
        )

        batch_size = nactions_50.shape[0]
        device = nactions_50.device
        t, u = self._sample_cd_timestep_pair(batch_size, device)
        noise = torch.randn_like(nactions_50)
        x_t = self.noise_scheduler.add_noise(nactions_50, noise, t)

        with torch.no_grad():
            x_u = self._teacher_rollout_to_timestep(
                teacher_policy=teacher_policy,
                sample=x_t,
                start_timestep=t,
                target_timestep=u,
                obs_tokens=obs_tokens,
                temporal_cond=tc_50,
            )
            target_out_u = target_policy.model(
                x_u,
                u,
                cond=obs_tokens,
                temporal_cond=tc_50,
                freq_mode="50hz",
            )
            target_x0_u, _ = target_policy._prediction_to_x0_eps(
                x_u, target_out_u, u
            )

        student_out_t = self.model(
            x_t,
            t,
            cond=obs_tokens,
            temporal_cond=tc_50,
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
            "dp_loss_10hz": loss_10.item(),
            "weighted_dp_loss_10hz": (self.loss_10hz_weight * loss_10).item(),
            "training_stage": 3.0,
        }
        if self.dsm_weight > 0:
            dsm_loss = F.mse_loss(student_x0_t, nactions_50)
            train_cd_loss = train_cd_loss + self.dsm_weight * dsm_loss
            loss_dict["cd_loss"] = train_cd_loss.item()
            loss_dict["dsm_loss_50hz"] = dsm_loss.item()
            loss_dict["weighted_dsm_loss_50hz"] = (
                self.dsm_weight * dsm_loss
            ).item()

        total = train_cd_loss + self.loss_10hz_weight * loss_10
        loss_dict["total_loss"] = total.item()
        return total, loss_dict

    def compute_loss(
        self,
        batch,
        target_policy: Optional["DPMultiFreqCD50HzConfidencePolicy"] = None,
        teacher_policy: Optional["DPMultiFreqCD50HzConfidencePolicy"] = None,
        loss_mask=None,
    ):
        if self.training_stage == 1:
            return self.compute_stage1_loss(batch, loss_mask)
        if self.training_stage == 2:
            return self.compute_confidence_loss(batch, loss_mask)
        if self.training_stage == 3:
            return self.compute_stage3_loss(
                batch,
                target_policy=target_policy,
                teacher_policy=teacher_policy,
                loss_mask=loss_mask,
            )
        raise ValueError(f"Invalid training_stage={self.training_stage}")

    def forward(
        self,
        batch,
        target_policy: Optional["DPMultiFreqCD50HzConfidencePolicy"] = None,
        teacher_policy: Optional["DPMultiFreqCD50HzConfidencePolicy"] = None,
        loss_mask=None,
    ):
        return self.compute_loss(
            batch,
            target_policy=target_policy,
            teacher_policy=teacher_policy,
            loss_mask=loss_mask,
        )

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
        default_steps = (
            self.num_inference_steps_50hz
            if freq_mode == "50hz"
            else self.num_inference_steps_10hz
        )
        steps = default_steps if num_inference_steps is None else int(num_inference_steps)
        if noisy_trajectory is None:
            trajectory = torch.randn(
                size=(cond.shape[0], horizon, self.action_dim),
                dtype=cond.dtype,
                device=cond.device,
                generator=generator,
            )
        else:
            trajectory = noisy_trajectory

        if freq_mode == "50hz" and steps == 1 and self.training_stage == 3:
            t_value = int(self.cd_timestep_grid[0])
            self.noise_scheduler.set_timesteps(1, device=trajectory.device)
            t = torch.tensor(t_value, dtype=torch.long, device=trajectory.device)
            t_batch = torch.full(
                (trajectory.shape[0],),
                t_value,
                dtype=torch.long,
                device=trajectory.device,
            )
            model_output = self.model(
                trajectory,
                t_batch,
                cond=cond,
                temporal_cond=temporal_cond,
                freq_mode=freq_mode,
            )
            target_t = torch.zeros_like(t_batch)
            return self._ddim_jump_to(trajectory, model_output, t_batch, target_t)

        self.noise_scheduler.set_timesteps(steps, device=trajectory.device)
        timesteps = self.noise_scheduler.timesteps
        for step_idx, t in enumerate(timesteps):
            t_batch = torch.full(
                (trajectory.shape[0],),
                int(t.item()),
                dtype=torch.long,
                device=trajectory.device,
            )
            model_output = self.model(
                trajectory,
                t_batch,
                cond=cond,
                temporal_cond=temporal_cond,
                freq_mode=freq_mode,
            )
            if step_idx + 1 < len(timesteps):
                target_value = int(timesteps[step_idx + 1].item())
            else:
                target_value = 0
            target_t = torch.full(
                (trajectory.shape[0],),
                target_value,
                dtype=torch.long,
                device=trajectory.device,
            )
            trajectory = self._ddim_jump_to(trajectory, model_output, t_batch, target_t)
        return trajectory

    def _get_inference_head(self, obs_tokens: Optional[torch.Tensor] = None):
        if not (self.use_confidence_based_head_selection and self.confidence_head is not None):
            return self.inference_head, None

        if obs_tokens is None:
            return self.inference_head, None

        confidence_value = self._compute_confidence_logits(obs_tokens)

        selected = "50hz" if confidence_value > self.confidence_threshold_for_head_selection else "10hz"
        return selected, confidence_value.mean().item()

    def _predict_one_head(
        self,
        obs_tokens: torch.Tensor,
        temporal_cond: Optional[torch.Tensor],
        head: str,
        num_inference_steps: Optional[int] = None,
        noisy_trajectory: Optional[torch.Tensor] = None,
    ):
        nsample = self.conditional_sample(
            cond=obs_tokens,
            temporal_cond=temporal_cond,
            freq_mode=head,
            num_inference_steps=num_inference_steps,
            noisy_trajectory=noisy_trajectory,
        )
        normalizer_key = "action_stack" if head == "50hz" else "action"
        action_pred = self.normalizer[normalizer_key].unnormalize(nsample)
        if head == "50hz":
            action = action_pred[:, self.action_start_50hz:self.horizon_50hz]
        else:
            action = action_pred[
                :, self.action_start_10hz:self.action_start_10hz + self.n_action_steps
            ]
        return action, action_pred

    def _predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        dataset_obs_temporal_downsample_ratio: int = 1,
        extended_obs_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        assert extended_obs_dict is not None, "extended_obs is required for RNN temporal condition"

        obs_tokens = self._predict_obs_tokens(obs_dict)

        tc_10 = self.get_temporal_cond(extended_obs_dict, head="10hz", allow_partial=True, post_process=True)
        tc_50 = self.get_temporal_cond(extended_obs_dict, head="50hz", allow_partial=True, post_process=True)
        action_10hz, action_pred_10hz = self._predict_one_head(
            obs_tokens, tc_10, "10hz", self.num_inference_steps_10hz
        )
        action_50hz, action_pred_50hz = self._predict_one_head(
            obs_tokens, tc_50, "50hz", self.num_inference_steps_50hz
        )

        result = {
            "action": action_10hz,
            "action_pred": action_pred_10hz,
            "action_10hz": action_10hz,
            "action_50hz": action_50hz,
            "action_pred_10hz": action_pred_10hz,
            "action_pred_50hz": action_pred_50hz,
        }
        return result

    def predict_action(self, *args, **kwargs):
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_action(*args, **kwargs)

    def sample_noisy_trajectory(
        self,
        B: int = 1,
        generator=None,
        obs_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, str]:
        head = self.inference_head
        horizon = self.horizon_50hz if head == "50hz" else self.horizon_10hz
        noisy = torch.randn(
            (B, horizon, self.action_dim),
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )
        noisy = noisy.reshape(B, -1)
        return noisy, head

    #need review
    def _predict_from_obs_tokens_and_noisy_trajectory(
        self,
        obs_tokens_and_noisy_trajectory_list: List[torch.Tensor],
        extended_obs_dict: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
        extended_obs_last_step_list: List[int],
        dataset_obs_temporal_downsample_ratio: int,
        extend_obs_pad_after: bool = False,
        head: Optional[str] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        obs_tokens = torch.cat(obs_tokens_and_noisy_trajectory_list, dim=0)
        batch_size = obs_tokens.shape[0]
        obs_tokens = obs_tokens.to(self.device).reshape(batch_size, -1, self.n_emb)
        noisy_trajectory = obs_tokens[:, -self.horizon_50hz * self.action_dim:]
        selected_head, confidence_value = self._get_inference_head(obs_tokens)

        target_horizon = self.horizon_50hz if selected_head == "50hz" else self.horizon_10hz

        steps = (
            self.num_inference_steps_50hz
            if selected_head == "50hz"
            else self.num_inference_steps_10hz
        )

        max_last_step = max(extended_obs_last_step_list)
        assert extended_obs_dict is not None, "extended_obs is required for RNN temporal condition"
        pad_after_n = target_horizon - max_last_step if extend_obs_pad_after else None

        if isinstance(extended_obs_dict, list):
            temporal_cond = [
                self.get_temporal_cond(
                    item,
                    head=selected_head,
                    extended_obs_last_step=max_last_step,
                    extend_obs_pad_after_n=pad_after_n,
                )
                for item in extended_obs_dict
            ]
        else:
            temporal_cond = self.get_temporal_cond(
                extended_obs_dict,
                head=selected_head,
                extended_obs_last_step=max_last_step,
                extend_obs_pad_after_n=pad_after_n,
            )

        temporal_cond_list = []
        for i in range(batch_size):
            source = temporal_cond[i] if isinstance(temporal_cond, list) else temporal_cond
            if extended_obs_last_step_list[i] < max_last_step:
                padding_obs = source[:, -1:, :].repeat(
                    1, max_last_step - extended_obs_last_step_list[i], 1
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
        if noisy_trajectory is not None:
            noisy_trajectory = noisy_trajectory[:, :temporal_cond.shape[1], :]

        action, action_pred = self._predict_one_head(
            obs_tokens,
            temporal_cond,
            selected_head,
            num_inference_steps=steps,
            noisy_trajectory=noisy_trajectory,
        )
        if selected_head == "50hz":
            start = self.action_start_50hz
            end = self.horizon_50hz
        else:
            start = self.action_start_10hz
            end = start + self.n_action_steps

        results = []
        for i in range(batch_size):
            item = {
                "selected_freq": selected_head,
            }
            if confidence_value is not None:
                item["confidence"] = confidence_value
            if extend_obs_pad_after:
                item.update({
                    "action": action[i:i + 1],
                    "action_pred": action_pred[i:i + 1],
                })
            else:
                valid_len = max(0, min(extended_obs_last_step_list[i] - start, end - start))
                item.update({
                    "action": action[i:i + 1, :valid_len, :],
                    "action_pred": action_pred[i:i + 1, :extended_obs_last_step_list[i], :],
                })
            results.append(item)
        return results

    def predict_from_obs_tokens_and_noisy_trajectory(self, *args, **kwargs):
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_from_obs_tokens_and_noisy_trajectory(*args, **kwargs)

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
        optim_groups = []

        if self.training_stage == 1:
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
        elif self.training_stage == 2:
            if self.confidence_head is None:
                raise ValueError("Stage 2 requires use_confidence_head=True.")
            if hasattr(self.confidence_head, "get_optim_groups"):
                optim_groups += self.confidence_head.get_optim_groups(weight_decay=weight_decay)
            else:
                optim_groups.append({
                    "params": list(self.confidence_head.parameters()),
                    "weight_decay": weight_decay,
                })
        elif self.training_stage == 3:
            optim_groups.extend(self.model.get_optim_groups(weight_decay=weight_decay))
        else:
            raise ValueError(f"Invalid training_stage={self.training_stage}")

        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
