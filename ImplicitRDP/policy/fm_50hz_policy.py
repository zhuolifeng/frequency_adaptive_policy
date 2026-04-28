from typing import Dict, List, Union, Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import reduce

from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from ImplicitRDP.model.force.rnn import RNN
from ImplicitRDP.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from ImplicitRDP.model.vision.timm_obs_encoder import TimmObsEncoder
from ImplicitRDP.model.vision.transformer_obs_encoder import TransformerObsEncoder
from ImplicitRDP.policy.fm_policy import FlowMatchingTransformerImagePolicy


class FlowMatchingTransformerImage50HzPolicy(FlowMatchingTransformerImagePolicy):
    """Single-frequency 50Hz Flow Matching policy using batch["action_stack"]."""

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: Union[TransformerObsEncoder, MultiImageObsEncoder, TimmObsEncoder],
        model: TransformerForDiffusion,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=10,
        beta_a: float = 1.5,
        beta_b: float = 1.0,
        n_emb: int = 768,
        use_rnn_obs_encoder=False,
        rnn_obs_encoder_n_layer=1,
        rnn_obs_encoder_hidden_dim=512,
        use_amp_for_inference: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        highfreq_extended_obs_key: str = "right_robot_tcp_wrench_stack",
        action_start_50hz: int = 13,
        action_normalizer_key: str = "action_stack",
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
            use_rnn_obs_encoder=False,
            rnn_obs_encoder_n_layer=rnn_obs_encoder_n_layer,
            rnn_obs_encoder_hidden_dim=rnn_obs_encoder_hidden_dim,
            use_amp_for_inference=use_amp_for_inference,
            amp_dtype=amp_dtype,
            **kwargs,
        )

        self.highfreq_extended_obs_key = highfreq_extended_obs_key
        self.action_start_50hz = action_start_50hz
        self.action_normalizer_key = action_normalizer_key

        if use_rnn_obs_encoder:
            assert "extended_obs" in shape_meta, "extended_obs is required for RNNObsEncoder"
            assert highfreq_extended_obs_key in shape_meta["extended_obs"], (
                f"{highfreq_extended_obs_key} is required for 50Hz temporal condition"
            )
            temporal_cond_dim = shape_meta["extended_obs"][highfreq_extended_obs_key]["shape"][-1]
            self.rnn_obs_encoder = RNN(
                input_dim=temporal_cond_dim,
                hidden_dim=rnn_obs_encoder_hidden_dim,
                layer_num=rnn_obs_encoder_n_layer,
                n_emb=n_emb,
            )
        else:
            self.rnn_obs_encoder = None

    def get_temporal_cond(
        self,
        extended_obs_dict,
        extended_obs_last_step: Optional[int] = None,
        extend_obs_pad_after_n: Optional[int] = None,
    ):
        key = self.highfreq_extended_obs_key
        obs = extended_obs_dict[key]
        if extended_obs_last_step is not None:
            obs = obs[:, -extended_obs_last_step:, ...]
        if extend_obs_pad_after_n is not None:
            padding_obs = obs[:, -1:, ...].repeat(1, extend_obs_pad_after_n, 1)
            obs = torch.cat([padding_obs, obs], dim=-2)
        return self.normalizer[f"extended_{key}"].normalize(obs)

    def _predict_from_obs_tokens_and_noisy_trajectory(
        self,
        obs_tokens_and_noisy_trajectory_list: List[torch.Tensor],
        extended_obs_dict: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
        extended_obs_last_step_list: List[int],
        dataset_obs_temporal_downsample_ratio: int,
        extend_obs_pad_after: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        obs_tokens_and_noisy_trajectory = torch.cat(obs_tokens_and_noisy_trajectory_list, dim=0)
        B = obs_tokens_and_noisy_trajectory.shape[0]
        obs_tokens = obs_tokens_and_noisy_trajectory[:, :- self.action_horizon * self.action_dim]
        noisy_trajectory = obs_tokens_and_noisy_trajectory[:, - self.action_horizon * self.action_dim:]
        obs_tokens = obs_tokens.to(self.device).reshape(B, -1, self.n_emb)
        noisy_trajectory = noisy_trajectory.to(self.device).reshape(B, self.action_horizon, self.action_dim)

        max_extended_obs_last_step = max(extended_obs_last_step_list)
        assert extended_obs_dict is not None, "extended_obs is required for RNNObsEncoder"
        if extend_obs_pad_after:
            extend_obs_pad_after_n = self.action_horizon - max_extended_obs_last_step
        else:
            extend_obs_pad_after_n = None

        if isinstance(extended_obs_dict, list):
            temporal_cond = [
                self.get_temporal_cond(
                    item,
                    max_extended_obs_last_step,
                    extend_obs_pad_after_n=extend_obs_pad_after_n,
                )
                for item in extended_obs_dict
            ]
        else:
            temporal_cond = self.get_temporal_cond(
                extended_obs_dict,
                max_extended_obs_last_step,
                extend_obs_pad_after_n=extend_obs_pad_after_n,
            )

        temporal_cond_list = []
        for i in range(B):
            this_temporal_cond = temporal_cond[i] if isinstance(temporal_cond, list) else temporal_cond
            if extended_obs_last_step_list[i] < max_extended_obs_last_step:
                padding_obs = this_temporal_cond[:, -1:, :].repeat(
                    1, max_extended_obs_last_step - extended_obs_last_step_list[i], 1
                )
                temporal_cond_list.append(
                    torch.cat(
                        [this_temporal_cond[:, -extended_obs_last_step_list[i]:, :], padding_obs],
                        dim=-2,
                    )
                )
            else:
                temporal_cond_list.append(this_temporal_cond)
        temporal_cond = torch.cat(temporal_cond_list, dim=0)

        cond_data = torch.zeros(
            size=(B, temporal_cond.shape[1], self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        noisy_trajectory = noisy_trajectory[:, :temporal_cond.shape[1], :]

        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            cond=obs_tokens,
            temporal_cond=temporal_cond,
            noisy_trajectory=noisy_trajectory,
            **self.kwargs,
        )

        assert nsample.shape == (B, temporal_cond.shape[1], self.action_dim)
        action_pred = self.normalizer[self.action_normalizer_key].unnormalize(nsample)

        start = self.action_start_50hz
        end = min(self.action_horizon, action_pred.shape[1])
        action = action_pred[:, start:end]

        results = []
        for i in range(B):
            if extend_obs_pad_after:
                results.append({
                    "action": action[i: i + 1],
                    "action_pred": action_pred[i: i + 1],
                })
            else:
                valid_len = max(0, min(extended_obs_last_step_list[i] - start, end - start))
                results.append({
                    "action": action[i: i + 1, :valid_len, :],
                    "action_pred": action_pred[i: i + 1, :extended_obs_last_step_list[i], :],
                })
        return results

    def _predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        dataset_obs_temporal_downsample_ratio: int = 1,
        extended_obs_dict: Dict[str, torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict

        obs_tokens = self.predict_obs_tokens(obs_dict)
        B = obs_tokens.shape[0]
        obs_tokens = obs_tokens.reshape(B, -1, self.n_emb)

        if self.rnn_obs_encoder is not None:
            assert extended_obs_dict is not None, "extended_obs is required for RNNObsEncoder"
            temporal_cond = self.get_temporal_cond(extended_obs_dict)
        else:
            temporal_cond = None

        if temporal_cond is None:
            cond_data = torch.zeros(
                size=(B, self.action_horizon, self.action_dim),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            cond_data = torch.zeros(
                size=(B, temporal_cond.shape[1], self.action_dim),
                device=self.device,
                dtype=self.dtype,
            )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            cond=obs_tokens,
            temporal_cond=temporal_cond,
            **self.kwargs,
        )

        assert nsample.shape == (B, cond_data.shape[1], self.action_dim)
        action_pred = self.normalizer[self.action_normalizer_key].unnormalize(nsample)

        start = self.action_start_50hz
        end = min(self.action_horizon, action_pred.shape[1])
        action = action_pred[:, start:end]

        return {
            "action": action,
            "action_pred": action_pred,
        }

    def model_forward(self, batch):
        assert "valid_mask" not in batch

        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer[self.action_normalizer_key].normalize(batch["action_stack"])
        trajectory = nactions
        B = nactions.shape[0]

        if self.rnn_obs_encoder is not None:
            assert "extended_obs" in batch, "extended_obs is required for RNNObsEncoder"
            temporal_cond = self.get_temporal_cond(batch["extended_obs"])
        else:
            temporal_cond = None

        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])

        t = torch.distributions.Beta(self.beta_a, self.beta_b).sample((B,)).to(trajectory.device)
        t = t * 0.999 + 0.001

        x1 = torch.randn(trajectory.shape, device=trajectory.device)
        t_expanded = t.view(B, 1, 1)
        x_t = t_expanded * x1 + (1 - t_expanded) * trajectory
        timesteps = (t * 1000).long()

        if self.rnn_obs_encoder is not None and temporal_cond is not None:
            temporal_cond_tokens = self.rnn_obs_encoder(temporal_cond)
        else:
            temporal_cond_tokens = None

        pred = self.model(
            x_t,
            timesteps,
            cond=obs_tokens,
            temporal_cond=temporal_cond_tokens if temporal_cond_tokens is not None else temporal_cond,
        )

        target = x1 - trajectory
        return trajectory, x1, t, pred, target

    def compute_loss(self, batch, loss_mask=None):
        trajectory, x1, t, pred, target = self.model_forward(batch)

        if loss_mask is None:
            loss_mask = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss[loss_mask]
        loss = loss.mean(dim=1)
        loss = loss.sum() / (loss_mask.sum() + 1e-8)

        return loss

    def predict_action(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_action(*args, **kwargs)
