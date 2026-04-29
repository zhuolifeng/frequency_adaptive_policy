"""
Teacher-free multi-frequency rectified-flow consistency policy.

This policy trains a single CT-only model from scratch. It does not train an FM
teacher, does not create a frozen teacher, and does not use distillation-mode
tokens in the transformer backbone.
"""
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from loguru import logger

from ImplicitRDP.policy.fm_multifreq_policy import FMMultiFreqPolicy
from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.model.diffusion.transformer_for_consistency_multifreq import (
    TransformerForConsistencyMultiFreq,
)


class FMMultiFreqConsistencyPolicy(FMMultiFreqPolicy):
    """
    Multi-frequency consistency policy trained from scratch.

    The network predicts a rectified-flow velocity v(x_t, t). The consistency
    function is f(x_t, t) = x_t - t * v(x_t, t). Training matches f from the
    student at t against f from the EMA target at the adjacent lower-noise time.
    """

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: torch.nn.Module,
        model: TransformerForConsistencyMultiFreq,
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
        ct_num_scales: int = 40,
        ct_beta_a: Optional[float] = None,
        ct_beta_b: Optional[float] = None,
        loss_weight_10hz: float = 1.0,
        loss_weight_50hz: float = 1.0,
        boundary_loss_weight: float = 0.05,
        boundary_t: float = 0.001,
        inference_head: str = "50hz",
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
            cd_num_scales=ct_num_scales,
            cd_time_sampling="beta",
            cd_beta_a=beta_a if ct_beta_a is None else ct_beta_a,
            cd_beta_b=beta_b if ct_beta_b is None else ct_beta_b,
            inference_head=inference_head,
            training_stage=1,
            **kwargs,
        )

        self.ct_num_scales = ct_num_scales
        self.ct_beta_a = beta_a if ct_beta_a is None else ct_beta_a
        self.ct_beta_b = beta_b if ct_beta_b is None else ct_beta_b
        self.loss_weight_10hz = loss_weight_10hz
        self.loss_weight_50hz = loss_weight_50hz
        self.boundary_loss_weight = boundary_loss_weight
        self.boundary_t = boundary_t

        assert self.ct_num_scales >= 2, "ct_num_scales must be >= 2"
        assert self.loss_weight_10hz >= 0.0
        assert self.loss_weight_50hz >= 0.0
        assert self.loss_weight_10hz + self.loss_weight_50hz > 0.0
        assert self.boundary_t > 0.0 and self.boundary_t < 1.0

        logger.info(
            "FMMultiFreqConsistencyPolicy initialized "
            f"(ct_num_scales={ct_num_scales}, ct_beta=({self.ct_beta_a}, {self.ct_beta_b}), "
            f"loss_weight_10hz={loss_weight_10hz}, loss_weight_50hz={loss_weight_50hz}, "
            f"boundary_loss_weight={boundary_loss_weight}, inference_head={inference_head})"
        )

    def set_training_stage(self, stage: int):
        """Compatibility no-op. CT-only policy has a single training mode."""
        self.training_stage = 1

    def set_teacher_model(self, teacher):
        raise RuntimeError("FMMultiFreqConsistencyPolicy does not use a teacher model.")

    def _sample_adjacent_times(self, B: int, device: torch.device):
        noise_level = torch.distributions.Beta(
            self.ct_beta_a, self.ct_beta_b
        ).sample((B,)).to(device)
        indices = ((1.0 - noise_level) * (self.ct_num_scales - 1)).long()
        indices = indices.clamp(0, self.ct_num_scales - 2)
        t = 1.0 - indices.float() / self.ct_num_scales
        t_next = 1.0 - (indices + 1).float() / self.ct_num_scales
        return t, t_next

    def _consistency_function(
        self,
        model,
        x_t: torch.Tensor,
        t: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond: torch.Tensor,
        freq_mode: str,
    ):
        timesteps = (t * 1000).long()
        v = model(
            x_t,
            timesteps,
            cond=obs_tokens,
            temporal_cond=temporal_cond,
            freq_mode=freq_mode,
        )
        return x_t - t.view(x_t.shape[0], 1, 1) * v

    def _compute_ct_loss_one_freq(
        self,
        x0: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond: torch.Tensor,
        freq_mode: str,
    ):
        assert self.ema_model_ref is not None, (
            "ema_model_ref must be set before CT training. "
            "Call set_ema_model_ref() in workspace."
        )
        B = x0.shape[0]
        device = x0.device
        noise = torch.randn_like(x0)
        t, t_next = self._sample_adjacent_times(B, device)

        t_exp = t.view(B, 1, 1)
        t_next_exp = t_next.view(B, 1, 1)
        x_t = t_exp * noise + (1.0 - t_exp) * x0
        x_t_next = t_next_exp * noise + (1.0 - t_next_exp) * x0

        f_student = self._consistency_function(
            self.model, x_t, t, obs_tokens, temporal_cond, freq_mode
        )
        with torch.no_grad():
            f_target = self._consistency_function(
                self.ema_model_ref,
                x_t_next,
                t_next,
                obs_tokens,
                temporal_cond,
                freq_mode,
            )

        consistency_loss = F.mse_loss(f_student, f_target.detach())

        if self.boundary_loss_weight <= 0.0:
            boundary_loss = x0.new_tensor(0.0)
        else:
            t_boundary = torch.full((B,), self.boundary_t, device=device)
            t_boundary_exp = t_boundary.view(B, 1, 1)
            x_boundary = t_boundary_exp * noise + (1.0 - t_boundary_exp) * x0
            f_boundary = self._consistency_function(
                self.model,
                x_boundary,
                t_boundary,
                obs_tokens,
                temporal_cond,
                freq_mode,
            )
            boundary_loss = F.mse_loss(f_boundary, x0)

        total = consistency_loss + self.boundary_loss_weight * boundary_loss
        return total, consistency_loss.detach(), boundary_loss.detach()

    def _compute_ct_loss(self, batch):
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

        loss_10 = nactions_10.new_tensor(0.0)
        loss_50 = nactions_50.new_tensor(0.0)
        ct_10 = nactions_10.new_tensor(0.0)
        ct_50 = nactions_50.new_tensor(0.0)
        boundary_10 = nactions_10.new_tensor(0.0)
        boundary_50 = nactions_50.new_tensor(0.0)

        if self.loss_weight_10hz > 0.0:
            loss_10, ct_10, boundary_10 = self._compute_ct_loss_one_freq(
                nactions_10, obs_tokens, temporal_cond_10hz, "10hz"
            )
        if self.loss_weight_50hz > 0.0:
            loss_50, ct_50, boundary_50 = self._compute_ct_loss_one_freq(
                nactions_50, obs_tokens, temporal_cond_50hz, "50hz"
            )

        total_loss = self.loss_weight_10hz * loss_10 + self.loss_weight_50hz * loss_50
        loss_dict = {
            "ct_loss": total_loss.item(),
            "ct_loss_10hz": ct_10.item(),
            "ct_loss_50hz": ct_50.item(),
            "boundary_loss_10hz": boundary_10.item(),
            "boundary_loss_50hz": boundary_50.item(),
            "total_loss": total_loss.item(),
        }
        return total_loss, loss_dict

    def compute_loss(self, batch, loss_mask=None):
        return self._compute_ct_loss(batch)

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
        model = self.model
        steps = num_inference_steps if num_inference_steps is not None else 1

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
        t_schedule = torch.linspace(1.0, 1.0 / steps, steps, device=x_t.device)

        for i in range(steps):
            t = t_schedule[i]
            t_tensor = torch.full((B,), t.item() * 1000, device=x_t.device).long()
            v = model(
                x_t,
                t_tensor,
                cond=cond,
                temporal_cond=temporal_cond,
                freq_mode=freq_mode,
            )
            x_0 = x_t - t * v
            if i < steps - 1:
                t_next = t_schedule[i + 1]
                eps = torch.randn_like(x_0)
                x_t = t_next * eps + (1.0 - t_next) * x_0
            else:
                x_t = x_0

        return x_t

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
        if extend_obs_pad_after:
            extend_obs_pad_after_n = target_horizon - max_extended_obs_last_step
        else:
            extend_obs_pad_after_n = None

        if isinstance(extended_obs_dict, list):
            temporal_cond = [
                self.get_temporal_cond(
                    item,
                    max_extended_obs_last_step,
                    extend_obs_pad_after_n=extend_obs_pad_after_n,
                    head=head,
                )
                for item in extended_obs_dict
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
            source = temporal_cond[i] if isinstance(temporal_cond, list) else temporal_cond
            if extended_obs_last_step_list[i] < max_extended_obs_last_step:
                padding_obs = source[:, -1:, :].repeat(
                    1,
                    max_extended_obs_last_step - extended_obs_last_step_list[i],
                    1,
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
        nsample = self.consistency_sample(
            cond=obs_tokens,
            temporal_cond=temporal_cond,
            freq_mode=head,
            noisy_trajectory=noisy_trajectory,
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
                results.append({
                    "action": action[i: i + 1],
                    "action_pred": action_pred[i: i + 1],
                })
            else:
                valid_len = max(
                    0,
                    min(extended_obs_last_step_list[i] - start, end - start),
                )
                results.append({
                    "action": action[i: i + 1, :valid_len, :],
                    "action_pred": action_pred[
                        i: i + 1, :extended_obs_last_step_list[i], :
                    ],
                })

        return results

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
            ), "extended_obs is required for RNN temporal condition"

        temporal_cond_10hz = self.get_temporal_cond(extended_obs_dict, head="10hz")
        temporal_cond_50hz = self.get_temporal_cond(extended_obs_dict, head="50hz")
        steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps

        nsample_10 = self.consistency_sample(
            cond=obs_tokens,
            temporal_cond=temporal_cond_10hz,
            freq_mode="10hz",
            num_inference_steps=steps,
        )
        nsample_50 = self.consistency_sample(
            cond=obs_tokens,
            temporal_cond=temporal_cond_50hz,
            freq_mode="50hz",
            num_inference_steps=steps,
        )

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
