import copy
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange, reduce

from ImplicitRDP.model.common.normalizer import LinearNormalizer
from ImplicitRDP.policy.base_image_policy import BaseImagePolicy
from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.model.vision.transformer_obs_encoder import TransformerObsEncoder
from ImplicitRDP.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from ImplicitRDP.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from ImplicitRDP.model.vision.timm_obs_encoder import TimmObsEncoder
from ImplicitRDP.model.force.rnn import RNN

from loguru import logger


class FlowMatchingTransformerImagePolicy(BaseImagePolicy):
    """
    Flow Matching (Rectified Flow) policy using TransformerForDiffusion backbone.

    Replaces DDPM/DDIM noise scheduling with Rectified Straight Flow:
    - Training: x_t = t * noise + (1-t) * x_0, predict velocity v = noise - x_0
    - Inference: Euler ODE integration from t=1 (noise) to t=0 (data)
    - Time sampling: Beta(1.5, 1.0) following pi0
    """
    def __init__(self,
            shape_meta: dict,
            obs_encoder: Union[TransformerObsEncoder, MultiImageObsEncoder, TimmObsEncoder],
            model: TransformerForDiffusion,
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=10,
            # flow matching time sampling
            beta_a: float = 1.5,
            beta_b: float = 1.0,
            # arch
            n_emb: int=768,
            use_rnn_obs_encoder=False,
            rnn_obs_encoder_n_layer=1,
            rnn_obs_encoder_hidden_dim=512,
            use_amp_for_inference: bool = False,
            amp_dtype: torch.dtype = torch.float16,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = horizon

        obs_shape = obs_encoder.output_shape()
        assert obs_shape[-1] == n_emb
        # compatible with multi_image_obs_encoder and timm_obs_encoder
        obs_tokens = obs_shape[-2] if len(obs_shape) == 3 else n_obs_steps

        all_extented_obs_keys = list(shape_meta['extended_obs'].keys()) if 'extended_obs' in shape_meta else []
        self.extented_obs_keys = sorted(all_extented_obs_keys)
        temporal_cond_dim = sum([shape_meta['extended_obs'][extented_obs_key]['shape'][-1] for extented_obs_key in self.extented_obs_keys])
        if use_rnn_obs_encoder:
            assert len(self.extented_obs_keys) > 0, "extended_obs is required for RNNObsEncoder"
            self.rnn_obs_encoder = RNN(
                input_dim=temporal_cond_dim,
                hidden_dim=rnn_obs_encoder_hidden_dim,
                layer_num=rnn_obs_encoder_n_layer,
                n_emb=n_emb,
            )
        else:
            self.rnn_obs_encoder = None

        self.shape_meta = shape_meta
        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.n_emb = n_emb
        self.num_inference_steps = num_inference_steps
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.use_amp_for_inference = use_amp_for_inference
        self.kwargs = kwargs

        if use_amp_for_inference:
            if amp_dtype == 'bf16' and torch.cuda.is_bf16_supported():
                logger.info(f"Using bf16 for AMP during inference")
                self.amp_dtype = torch.bfloat16
            elif amp_dtype == 'fp16':
                logger.info(f"Using fp16 for AMP during inference")
                self.amp_dtype = torch.float16
            else:
                logger.warning(f"AMP dtype {amp_dtype} not supported or bf16 not available, falling back to fp16 during inference")
                self.amp_dtype = torch.float16
        else:
            self.amp_dtype = None

    # ========= inference  ============
    def conditional_sample(self,
            condition_data, condition_mask,
            cond=None,
            temporal_cond=None,
            noisy_trajectory=None,
            generator=None,
            **kwargs
            ):
        """
        Euler ODE solver for Rectified Flow.
        Integrates from t=1 (noise) to t=0 (data).

        Args:
            condition_data: data to condition on (for inpainting-style conditioning)
            condition_mask: mask indicating which parts are conditioned
            cond: observation tokens (B, N, n_emb)
            temporal_cond: temporal condition (raw, before RNN)
            noisy_trajectory: initial noise, if None will sample from N(0,I)
            generator: random generator for reproducibility
        """
        model = self.model

        if noisy_trajectory is None:
            trajectory = torch.randn(
                size=condition_data.shape,
                dtype=condition_data.dtype,
                device=condition_data.device,
                generator=generator)
        else:
            trajectory = noisy_trajectory

        B = trajectory.shape[0]

        # RNN computed once outside the Euler loop
        if self.rnn_obs_encoder is not None and temporal_cond is not None:
            temporal_cond_tokens = self.rnn_obs_encoder(temporal_cond)
        else:
            temporal_cond_tokens = None

        # Euler integration: t from 1 -> 0
        t_schedule = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=trajectory.device)

        for i in range(self.num_inference_steps):
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. current time and step size
            t_cur = t_schedule[i]
            dt = t_schedule[i + 1] - t_schedule[i]  # negative value (moving from 1->0)

            # 3. prepare timestep for model (scale to match SinusoidalPosEmb range)
            t_tensor = torch.full((B,), t_cur * 1000, device=trajectory.device).long()

            # 4. predict velocity
            model_output = model(
                trajectory,
                t_tensor,
                cond=cond,
                temporal_cond=temporal_cond_tokens if temporal_cond_tokens is not None else temporal_cond,
            )

            # 5. Euler update: x_{t+dt} = x_t + dt * v(x_t, t)
            trajectory = trajectory + dt * model_output

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def sample_noisy_trajectory(self,
                                B=1,
                                generator=None
                                ) -> torch.Tensor:
        noisy_trajectory = torch.randn(
            size=(B, self.action_horizon, self.action_dim),
            dtype=torch.float32,
            device=self.device,
            generator=generator)

        noisy_trajectory = noisy_trajectory.reshape(B, -1)

        return noisy_trajectory

    def _predict_obs_tokens(self,
                           obs_dict: Dict[str, torch.Tensor],
                           return_expanded_obs_tokens=False
                           ) -> torch.Tensor:
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        # process input
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])
        # (B, N, n_emb)

        obs_tokens = obs_tokens.reshape(B, -1)
        if return_expanded_obs_tokens:
            obs_tokens = obs_tokens.unsqueeze(1).expand(-1, self.action_horizon, -1)

        return obs_tokens

    def predict_obs_tokens(self,
                           *args,
                           **kwargs,
                           ) -> torch.Tensor:
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_obs_tokens(*args, **kwargs)

    def _predict_from_obs_tokens_and_noisy_trajectory(self,
                                obs_tokens_and_noisy_trajectory_list: List[torch.Tensor],
                                extended_obs_dict: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
                                extended_obs_last_step_list: List[int],
                                dataset_obs_temporal_downsample_ratio: int,
                                extend_obs_pad_after: bool = False
                                ) -> List[Dict[str, torch.Tensor]]:
        """
        support batch prediction
        """
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
            temporal_cond = []
            for i in range(len(extended_obs_dict)):
                temporal_cond.append(
                    self.get_temporal_cond(extended_obs_dict[i], max_extended_obs_last_step, extend_obs_pad_after_n=extend_obs_pad_after_n)
                )
        else:
            temporal_cond = self.get_temporal_cond(extended_obs_dict, max_extended_obs_last_step, extend_obs_pad_after_n=extend_obs_pad_after_n)

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

        # empty data for action
        cond_data = torch.zeros(size=(B, temporal_cond.shape[1], self.action_dim),
                                device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        noisy_trajectory = noisy_trajectory[:, :temporal_cond.shape[1], :]

        # run sampling via Euler ODE
        nsample = self.conditional_sample(
            condition_data = cond_data,
            condition_mask = cond_mask,
            cond = obs_tokens,
            temporal_cond = temporal_cond,
            noisy_trajectory = noisy_trajectory,
            **self.kwargs)

        # unnormalize prediction
        assert nsample.shape == (B, temporal_cond.shape[1], self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)

        # hack: align with the training process
        To = self.n_obs_steps * dataset_obs_temporal_downsample_ratio
        # get action
        start = To - 1
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
                results.append({
                    'action': action[i: i + 1, :extended_obs_last_step_list[i] - start, :],
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

    def _predict_action(self,
                       obs_dict: Dict[str, torch.Tensor],
                       dataset_obs_temporal_downsample_ratio: int = 1,
                       extended_obs_dict: Dict[str, torch.Tensor] = None
                       ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        obs_tokens = self.predict_obs_tokens(obs_dict)
        B = obs_tokens.shape[0]
        obs_tokens = obs_tokens.reshape(B, -1, self.n_emb)

        if self.rnn_obs_encoder is not None:
            assert extended_obs_dict is not None, "extended_obs is required for RNNObsEncoder"
            temporal_cond = self.get_temporal_cond(extended_obs_dict)
        else:
            temporal_cond = None

        # empty data for action
        if temporal_cond is None:
            cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        else:
            cond_data = torch.zeros(size=(B, temporal_cond.shape[1], self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling via Euler ODE
        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            cond=obs_tokens,
            temporal_cond=temporal_cond,
            **self.kwargs)

        # unnormalize prediction
        assert nsample.shape == (B, cond_data.shape[1], self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)

        # hack: align with the training process
        To = self.n_obs_steps * dataset_obs_temporal_downsample_ratio
        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    def predict_action(self,
                       *args,
                       **kwargs,
                       ) -> Dict[str, torch.Tensor]:
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_action(*args, **kwargs)

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self,
            lr: float,
            weight_decay: float,
            obs_encoder_lr: float,
            obs_encoder_weight_decay: float,
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=weight_decay)

        if self.rnn_obs_encoder is not None:
            optim_groups += self.rnn_obs_encoder.get_optim_groups(
                weight_decay=weight_decay
            )

        backbone_params = list()
        other_obs_params = list()
        for key, value in self.obs_encoder.named_parameters():
            if key.startswith('key_model_map'):
                backbone_params.append(value)
            else:
                other_obs_params.append(value)
        optim_groups.append({
            "params": backbone_params,
            "weight_decay": obs_encoder_weight_decay,
            "lr": obs_encoder_lr # for fine tuning
        })
        optim_groups.append({
            "params": other_obs_params,
            "weight_decay": obs_encoder_weight_decay
        })

        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas
        )
        return optimizer

    def get_temporal_cond(self, extended_obs_dict, extended_obs_last_step=None, extend_obs_pad_after_n=None):
        temporal_cond = []
        for extented_obs_key in self.extented_obs_keys:
            if extended_obs_last_step is not None:
                if len(self.shape_meta['extended_obs'][extented_obs_key]['shape']) == 1:
                    extented_obs = extended_obs_dict[extented_obs_key][..., -extended_obs_last_step:, :]
                elif len(self.shape_meta['extended_obs'][extented_obs_key]['shape']) == 2:
                    extented_obs = extended_obs_dict[extented_obs_key][..., -extended_obs_last_step:, :, :]
                else:
                    raise ValueError(f"Only support 1d or 2d extended obs")
            else:
                extented_obs = extended_obs_dict[extented_obs_key]
            if extend_obs_pad_after_n is not None:
                if len(self.shape_meta['extended_obs'][extented_obs_key]['shape']) == 1:
                    padding_obs = extended_obs_dict[extented_obs_key][..., -1:, :].repeat(1, extend_obs_pad_after_n, 1)
                elif len(self.shape_meta['extended_obs'][extented_obs_key]['shape']) == 2:
                    padding_obs = extended_obs_dict[extented_obs_key][..., -1:, :, :].repeat(1, extend_obs_pad_after_n, 1, 1)
                else:
                    raise ValueError(f"Only support 1d or 2d extended obs")
                extented_obs = torch.cat([padding_obs, extented_obs], dim=-2)
            extented_obs = self.normalizer[f"extended_{extented_obs_key}"].normalize(extented_obs)
            temporal_cond.append(extented_obs)
        temporal_cond = torch.cat(temporal_cond, dim=-1)
        return temporal_cond

    def model_forward(self, batch):
        """
        Rectified Flow forward pass (following pi0 convention):
        1. Sample t ~ Beta(beta_a, beta_b) * 0.999 + 0.001
        2. Interpolate: x_t = t * noise + (1-t) * x_0
        3. Model predicts velocity v(x_t, t)
        4. Target: v_target = noise - x_0 (pi0 convention)
        """
        # normalize input
        assert 'valid_mask' not in batch

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        trajectory = nactions  # x_0: clean action
        B = nactions.shape[0]

        if self.rnn_obs_encoder is not None:
            assert 'extended_obs' in batch, "extended_obs is required for RNNObsEncoder"
            temporal_cond = self.get_temporal_cond(batch["extended_obs"])
        else:
            temporal_cond = None

        # process input
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])
        # (B, N, n_emb)

        # --- Rectified Flow forward process ---
        # Sample continuous time t ~ Beta(beta_a, beta_b) * 0.999 + 0.001
        t = torch.distributions.Beta(self.beta_a, self.beta_b).sample((B,)).to(trajectory.device)
        t = t * 0.999 + 0.001  # avoid exact 0 and 1

        # Sample noise x_1 ~ N(0, I)
        x1 = torch.randn(trajectory.shape, device=trajectory.device)

        # Rectified Flow interpolation: x_t = t * x_1 + (1-t) * x_0
        # Convention: t=1 is pure noise, t=0 is clean data
        t_expanded = t.view(B, 1, 1)
        x_t = t_expanded * x1 + (1 - t_expanded) * trajectory

        # Scale timestep for SinusoidalPosEmb (original expects integer timesteps)
        timesteps = (t * 1000).long()

        # Compute temporal conditioning
        if self.rnn_obs_encoder is not None and temporal_cond is not None:
            temporal_cond_tokens = self.rnn_obs_encoder(temporal_cond)
        else:
            temporal_cond_tokens = None

        # Predict velocity
        pred = self.model(
            x_t,
            timesteps,
            cond=obs_tokens,
            temporal_cond=temporal_cond_tokens if temporal_cond_tokens is not None else temporal_cond,
        )

        # Target velocity: x_1 - x_0 (noise - data, following pi0 convention)
        # At inference, Euler uses dt < 0, so x += dt * v moves from noise toward data.
        target = x1 - trajectory

        return trajectory, x1, t, pred, target

    def compute_loss(self, batch, loss_mask=None):
        """Compute Rectified Flow velocity matching loss."""
        trajectory, x1, t, pred, target = self.model_forward(batch)

        if loss_mask is None:
            loss_mask = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss[loss_mask]
        loss = loss.mean(dim=1)
        loss = loss.sum() / (loss_mask.sum() + 1e-8)

        return loss

    def forward(self, batch):
        return self.compute_loss(batch)
