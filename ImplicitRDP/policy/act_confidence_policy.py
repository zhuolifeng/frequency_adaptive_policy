from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import reduce
from loguru import logger

from ImplicitRDP.model.common.normalizer import LinearNormalizer
from ImplicitRDP.policy.base_image_policy import BaseImagePolicy
from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.model.act.act_transformer import ACTTransformer
from ImplicitRDP.model.confidence.transformer import TransformerConfidenceHead
from ImplicitRDP.model.force.rnn import RNN
from ImplicitRDP.model.force.force_encoder import ForceEncoder
from ImplicitRDP.model.vector_quantize_pytorch.residual_vq import ResidualVQ
from ImplicitRDP.policy.curriculum_utils import get_scale, gaussian_2d_smoothing_separate


class ACTConfidencePolicy(BaseImagePolicy):
    """A pure ACT policy with optional confidence head (no diffusion branch)."""

    def __init__(self,
                 shape_meta: dict,
                 act_model: ACTTransformer,
                 obs_encoder: Optional[torch.nn.Module],
                 horizon: int,
                 n_action_steps: int,
                 n_obs_steps: int,
                 input_pertub: float = 0.0,
                 obs_drop: float = 0.0,
                 obs_drop_schedule_type: str = "linear",
                 obs_drop_keys: Optional[List[str]] = None,
                 obs_drop_type: str = 'zero',
                 obs_drop_noise_scale: float = 1.0,
                 sca_drop: float = 0.0,
                 gaussian_start: float = 10.0,
                 gaussian_end: float = 0.0,
                 gaussian_ratio: float = 2 / 3,
                 n_emb: int = 768,
                 use_rnn_obs_encoder: bool = False,
                 rnn_obs_encoder_n_layer: int = 1,
                 rnn_obs_encoder_hidden_dim: int = 512,
                 use_force_encoder: bool = False,
                 force_encoder_hidden_dims: Optional[List[int]] = None,
                 force_encoder_output_dim: int = 128,
                 use_vq: bool = False,
                 vq_num_quantizers: int = 2,
                 vq_codebook_size: int = 32,
                 vq_quantize_dropout: bool = False,
                 vq_ema_update: bool = True,
                 vq_learnable_codebook: bool = False,
                 use_amp_for_inference: bool = False,
                 amp_dtype: torch.dtype = torch.float16,
                 confidence_hidden_dim: Optional[int] = None,
                 confidence_gamma: float = 1.0,
                 confidence_alpha: float = 0.2,
                 confidence_log_sigma_min: float = -5.0,
                 confidence_log_sigma_max: float = 5.0,
                 confidence_clamp_log_sigma: bool = True,
                 confidence_obs_encoder_n_layers: int = 2,
                 confidence_obs_encoder_hidden_dim: Optional[int] = None,
                 use_confidence_head: bool = True,
                 confidence_head_type: str = 'transformer',
                 confidence_transformer_n_heads: int = 8,
                 confidence_transformer_dropout: float = 0.1,
                 confidence_action_dims: int = 6,
                 normalize_confidence_loss: bool = True,
                 **kwargs):
        super().__init__()
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        self.action_dim = action_shape[0]
        self.action_horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.input_pertub = input_pertub

        self.shape_meta = shape_meta
        self.obs_encoder = obs_encoder
        self.act_model = act_model
        self.normalizer = LinearNormalizer()
        self.kwargs = kwargs

        self.obs_drop = obs_drop
        self.obs_drop_schedule_type = obs_drop_schedule_type
        self.obs_drop_keys = [] if obs_drop_keys is None else obs_drop_keys
        self.obs_drop_type = obs_drop_type
        self.obs_drop_noise_scale = obs_drop_noise_scale
        self.sca_drop = sca_drop
        self.gaussian_start = gaussian_start
        self.gaussian_end = gaussian_end
        self.gaussian_ratio = gaussian_ratio
        self.n_emb = n_emb
        self.use_amp_for_inference = use_amp_for_inference
        self.amp_dtype = amp_dtype if use_amp_for_inference else None
        self.progress = 0.0

        all_extented_obs_keys = list(shape_meta['extended_obs'].keys()) if 'extended_obs' in shape_meta else []
        self.extented_obs_keys = sorted(all_extented_obs_keys)
        temporal_cond_dim = sum([shape_meta['extended_obs'][extented_obs_key]['shape'][-1]
                                 for extented_obs_key in self.extented_obs_keys])

        self.wrench_stack_obs_keys = []
        self.force_encoder = None
        if use_force_encoder:
            force_encoder_hidden_dims = force_encoder_hidden_dims or [32, 64, 128]
            if len(self.extented_obs_keys) != 0:
                assert all(["wrench_stack" in key for key in self.extented_obs_keys]), \
                    "Force encoder requires extended_obs only contains wrench_stack"
                input_dim = temporal_cond_dim
                key_in_obs_count = 0
                for extented_obs_key in self.extented_obs_keys:
                    if extented_obs_key in shape_meta['obs']:
                        key_in_obs_count += 1
                assert key_in_obs_count == 0 or key_in_obs_count == len(self.extented_obs_keys), \
                    "All wrench_stack keys in extended_obs must exist in obs or not exist in obs"
                if key_in_obs_count == len(self.extented_obs_keys):
                    self.wrench_stack_obs_keys = self.extented_obs_keys
            else:
                obs_keys = list(shape_meta['obs'].keys())
                all_wrench_stack_obs_keys = [key for key in obs_keys if 'wrench_stack' in key]
                input_dim = sum([shape_meta['obs'][key]['shape'][-1] for key in all_wrench_stack_obs_keys])
                self.wrench_stack_obs_keys = sorted(all_wrench_stack_obs_keys)

            self.force_encoder = ForceEncoder(
                input_dim=input_dim,
                output_dim=force_encoder_output_dim,
                hidden_dims=force_encoder_hidden_dims
            )
            temporal_cond_dim = force_encoder_output_dim

        self.rnn_obs_encoder = None
        if use_rnn_obs_encoder:
            assert len(self.extented_obs_keys) > 0, "extended_obs is required when using RNN obs encoder"
            self.rnn_obs_encoder = RNN(
                input_dim=temporal_cond_dim,
                hidden_dim=rnn_obs_encoder_hidden_dim,
                layer_num=rnn_obs_encoder_n_layer,
                n_emb=n_emb,
            )

        self.vq = None
        if use_vq:
            assert self.rnn_obs_encoder is not None, "VQ is only used when using RNNObsEncoder"
            self.vq = ResidualVQ(
                dim=n_emb,
                num_quantizers=vq_num_quantizers,
                codebook_size=vq_codebook_size,
                quantize_dropout=vq_quantize_dropout,
                ema_update=vq_ema_update,
                learnable_codebook=vq_learnable_codebook,
            )

        self.confidence_gamma = confidence_gamma
        self.confidence_alpha = confidence_alpha
        self.confidence_log_sigma_min = confidence_log_sigma_min
        self.confidence_log_sigma_max = confidence_log_sigma_max
        self.use_confidence_head = use_confidence_head
        self.confidence_action_dims = confidence_action_dims
        self.normalize_confidence_loss = normalize_confidence_loss
        self.confidence_head_type = (confidence_head_type or 'transformer').lower()
        hidden_dim = confidence_hidden_dim or n_emb
        self.confidence_head = None
        if self.use_confidence_head:
            if self.confidence_head_type == 'transformer':
                self.confidence_head = TransformerConfidenceHead(
                    n_emb=n_emb,
                    hidden_dim=hidden_dim,
                    obs_encoder_n_layers=confidence_obs_encoder_n_layers,
                    obs_encoder_hidden_dim=confidence_obs_encoder_hidden_dim,
                    n_heads=confidence_transformer_n_heads,
                    dropout=confidence_transformer_dropout,
                    log_sigma_min=confidence_log_sigma_min,
                    log_sigma_max=confidence_log_sigma_max,
                    clamp_log_sigma=confidence_clamp_log_sigma
                )
            else:
                raise ValueError(
                    f"Unsupported confidence_head_type '{confidence_head_type}'. "
                    "Choose from 'transformer'."
                )
        else:
            self.confidence_head = None
            self.available_heads = ['act']
            print(f"######################## use_confidence_head is False, available_heads: {self.available_heads}")

        self.inference_head = 'act'
        self.action_history = None
        logger.info("ACTConfidencePolicy initialized (single ACT head with confidence head)")

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def reset(self):
        self.action_history = None


    def _compute_confidence_logits(self,
                                    obs_tokens: torch.Tensor) -> Optional[torch.Tensor]:
        if self.confidence_head is None:
            return None
        return self.confidence_head(obs_tokens)

    def get_temporal_cond(self,
                          extended_obs_dict: Optional[Dict[str, torch.Tensor]],
                          extended_obs_last_step: Optional[int] = None,
                          extend_obs_pad_after_n: Optional[int] = None,
                          allow_partial: bool = False,
                          post_process: bool = False) -> Optional[torch.Tensor]:
        if extended_obs_dict is None or len(self.extented_obs_keys) == 0:
            return None

        temporal_cond = []
        for extented_obs_key in self.extented_obs_keys:
            if extented_obs_key not in extended_obs_dict:
                if allow_partial:
                    continue
                raise KeyError(f"Missing extended obs key: {extented_obs_key}")

            extented_obs = extended_obs_dict[extented_obs_key]
            obs_dim = len(self.shape_meta['extended_obs'][extented_obs_key]['shape'])

            if extended_obs_last_step is not None:
                if obs_dim == 1:
                    extented_obs = extented_obs[..., -extended_obs_last_step:, :]
                elif obs_dim == 2:
                    extented_obs = extented_obs[..., -extended_obs_last_step:, :, :]
                else:
                    raise ValueError("Only support 1d or 2d extended obs")

            if extend_obs_pad_after_n is not None:
                if obs_dim == 1:
                    padding_obs = extended_obs_dict[extented_obs_key][..., -1:, :].repeat(1, extend_obs_pad_after_n, 1)
                elif obs_dim == 2:
                    padding_obs = extended_obs_dict[extented_obs_key][..., -1:, :, :].repeat(1, extend_obs_pad_after_n, 1, 1)
                else:
                    raise ValueError("Only support 1d or 2d extended obs")
                extented_obs = torch.cat([padding_obs, extented_obs], dim=-2)
            extented_obs = self.normalizer[f"extended_{extented_obs_key}"].normalize(extented_obs)
            temporal_cond.append(extented_obs)

        if len(temporal_cond) == 0:
            return None

        temporal_cond = torch.cat(temporal_cond, dim=-1)
        if post_process:
            processed_cond = temporal_cond

            if self.force_encoder is not None and len(self.wrench_stack_obs_keys) == 0:
                B, T, _ = processed_cond.shape
                processed_cond_4d = processed_cond.unsqueeze(1)
                processed_cond = self.force_encoder(processed_cond_4d)
                processed_cond = processed_cond.squeeze(1)
                processed_cond = processed_cond.unsqueeze(1).expand(B, T, -1)

            if self.rnn_obs_encoder is not None:
                processed_cond = self.rnn_obs_encoder(processed_cond)
                if self.vq is not None:
                    processed_cond, _, _ = self.vq(processed_cond)
            else:
                assert self.vq is None, "VQ is only used when using RNNObsEncoder"

            temporal_cond = processed_cond

        return temporal_cond

    def predict_action(self,
                       obs_dict: Dict[str, torch.Tensor],
                       extended_obs_dict: Optional[Dict[str, torch.Tensor]] = None,
                       **kwargs) -> Dict[str, torch.Tensor]:
        return self._predict_action_act(obs_dict, extended_obs_dict)

    def _predict_obs_tokens(self,
                            obs_dict: Dict[str, torch.Tensor],
                            extended_obs_dict: Optional[Dict[str, torch.Tensor]] = None,
                            return_expanded_obs_tokens: bool = False,
                            return_temporal_cond: bool = False
                            ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))

        if self.force_encoder is not None and self.wrench_stack_obs_keys:
            wrench_stack_cond = []
            for wrench_stack_obs_key in self.wrench_stack_obs_keys:
                wrench_stack_obs = this_nobs[wrench_stack_obs_key]
                wrench_stack_cond.append(wrench_stack_obs)
            wrench_stack_cond = torch.cat(wrench_stack_cond, dim=-1)
            wrench_stack_cond = wrench_stack_cond.reshape(B, -1, *wrench_stack_cond.shape[1:])
            force_emb = self.force_encoder(wrench_stack_cond)
            force_emb = force_emb.reshape(-1, *force_emb.shape[2:])
            this_nobs["force_emb"] = force_emb

        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])

        temporal_cond = None
        if self.rnn_obs_encoder is not None:
            if extended_obs_dict is not None:
                temporal_cond = self.get_temporal_cond(
                    extended_obs_dict,
                    allow_partial=True,
                    post_process=True
                )
            else:
                if len(self.extented_obs_keys) > 0:
                    extended_obs_list = []
                    for key in self.extented_obs_keys:
                        if key in obs_dict:
                            extended_obs_list.append(obs_dict[key])
                    if len(extended_obs_list) > 0:
                        extended_obs_dict_local = {}
                        for i, key in enumerate(self.extented_obs_keys):
                            if key in obs_dict:
                                extended_obs_dict_local[key] = obs_dict[key]
                        temporal_cond = self.get_temporal_cond(
                            extended_obs_dict_local,
                            allow_partial=True,
                            post_process=True
                        )

        if return_temporal_cond:
            return obs_tokens, temporal_cond

        obs_flat = obs_tokens.reshape(B, -1)
        if return_expanded_obs_tokens:
            obs_flat = obs_flat.unsqueeze(1).expand(-1, self.action_horizon, -1)
        return obs_flat

    def predict_obs_tokens(self, *args, **kwargs) -> torch.Tensor:
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_obs_tokens(*args, **kwargs)

    def sample_noisy_trajectory(self,
                                B: int = 1,
                                generator=None,
                                return_expanded_noisy_trajectory: bool = False
                                ) -> torch.Tensor:

        noisy_trajectory = torch.zeros((B,0), dtype=self.dtype, device=self.device)
        if return_expanded_noisy_trajectory:
            noisy_trajectory = noisy_trajectory.unsqueeze(1).expand(-1, self.action_horizon, -1)
        return noisy_trajectory

    def _predict_from_obs_tokens_and_noisy_trajectory(self,
                                obs_tokens_and_noisy_trajectory_list: List[torch.Tensor],
                                extended_obs_dict: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
                                extended_obs_last_step_list: List[int],
                                dataset_obs_temporal_downsample_ratio: int,
                                extend_obs_pad_after: bool = False,
                                inpainting_action: torch.Tensor = None,
                                inpainting_resample_steps: int = 1,
                                sample_per_trajectory: Optional[int] = None,
                                inference_batch_size: Optional[int] = None,
                                return_noisy_trajectory: bool = False
                                ) -> List[Dict[str, torch.Tensor]]:
        obs_tokens_and_noisy_trajectory = torch.cat(obs_tokens_and_noisy_trajectory_list, dim=0)
        B = obs_tokens_and_noisy_trajectory.shape[0]

        obs_tokens = obs_tokens_and_noisy_trajectory.to(self.device).reshape(B, -1, self.n_emb)

        if extended_obs_dict is None:
            temporal_cond = None
            temporal_cond_tokens = None
            max_extended_obs_last_step = None
            extend_obs_pad_after_n = None
        else:
            max_extended_obs_last_step = max(extended_obs_last_step_list)
            if extend_obs_pad_after:
                extend_obs_pad_after_n = self.action_horizon - max_extended_obs_last_step
            else:
                extend_obs_pad_after_n = None

            if isinstance(extended_obs_dict, list):
                temporal_cond = [
                    self.get_temporal_cond(
                        extended_obs_dict[i],
                        max_extended_obs_last_step,
                        extend_obs_pad_after_n=extend_obs_pad_after_n,
                        post_process=True
                    )
                    for i in range(len(extended_obs_dict))
                ]
            else:
                temporal_cond = self.get_temporal_cond(
                    extended_obs_dict,
                    max_extended_obs_last_step,
                    extend_obs_pad_after_n=extend_obs_pad_after_n,
                    post_process=True
                )

            if temporal_cond is None:
                temporal_cond = None
                temporal_cond_tokens = None
            else:
                temporal_cond_list = []
                for i in range(B):
                    if isinstance(temporal_cond, list):
                        cond = temporal_cond[i]
                    else:
                        cond = temporal_cond

                    if cond is None:
                        temporal_cond_list.append(None)
                        continue

                    if extended_obs_last_step_list[i] < max_extended_obs_last_step:
                        padding_obs = cond[:, -1:, :].repeat(
                            1,
                            max_extended_obs_last_step - extended_obs_last_step_list[i],
                            1
                        )
                        temporal_cond_list.append(
                            torch.cat([cond[:, -extended_obs_last_step_list[i]:, :], padding_obs], dim=-2)
                        )
                    else:
                        temporal_cond_list.append(cond)

                if any(tc is None for tc in temporal_cond_list):
                    temporal_cond = None
                    temporal_cond_tokens = None
                else:
                    temporal_cond = torch.cat(temporal_cond_list, dim=0)
                    temporal_cond_tokens = temporal_cond

        temporal_cond = temporal_cond_tokens if ('temporal_cond_tokens' in locals() and temporal_cond_tokens is not None) else temporal_cond

        results = []
        To = self.n_obs_steps * dataset_obs_temporal_downsample_ratio
        start = To - 1

        for i in range(B):
            obs_tokens_single = obs_tokens[i:i + 1]
            temporal_cond_single = temporal_cond[i:i + 1] if temporal_cond is not None else None
            action_dict = self._predict_action_act_from_tokens(
                obs_tokens=obs_tokens_single,
                temporal_cond=temporal_cond_single,
                action_history=None,
                update_history=False
            )
            action_chunk = action_dict['action']

            total_steps = temporal_cond_single.shape[1] if temporal_cond_single is not None else self.action_horizon
            action_pred = torch.zeros(
                (1, total_steps, self.action_dim),
                dtype=action_chunk.dtype,
                device=action_chunk.device
            )
            available = min(action_chunk.shape[1], max(total_steps - start, 0))
            if available > 0:
                action_pred[:, start:start + available, :] = action_chunk[:, :available, :]

            valid_window = max(extended_obs_last_step_list[i] - start, 0)
            action_window = action_chunk[:, :min(valid_window, action_chunk.shape[1]), :]

            if sample_per_trajectory is not None:
                action_pred = action_pred.unsqueeze(1).repeat(1, sample_per_trajectory, 1, 1)
                action_window = action_window.unsqueeze(1).repeat(1, sample_per_trajectory, 1, 1)

            results.append({
                'action': action_window,
                'action_pred': action_pred
            })

        if sample_per_trajectory is not None:
            noisy_shape = (B, sample_per_trajectory, self.action_horizon, 0)
        else:
            noisy_shape = (B, self.action_horizon, 0)
        noisy_placeholder = torch.zeros(noisy_shape, dtype=self.dtype, device=self.device)
        if return_noisy_trajectory:
            return results, noisy_placeholder
        else:
            return results

    def predict_from_obs_tokens_and_noisy_trajectory(self,
                                *args,
                                **kwargs,
                            ) -> List[Dict[str, torch.Tensor]]:
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_from_obs_tokens_and_noisy_trajectory(
                *args,
                **kwargs
            )

    def _predict_action_act(self,
                            obs_dict: Dict[str, torch.Tensor],
                            extended_obs_dict: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        obs_tokens, temporal_cond = self._predict_obs_tokens(obs_dict, extended_obs_dict, return_temporal_cond=True)
        B = obs_tokens.shape[0]
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, self.n_emb)
        action_history = self.action_history
        return self._predict_action_act_from_tokens(
            obs_tokens=obs_tokens,
            temporal_cond=temporal_cond,
            action_history=action_history
        )

    def _predict_action_act_from_tokens(self,
                                        obs_tokens: torch.Tensor,
                                        temporal_cond: Optional[torch.Tensor],
                                        action_history: Optional[torch.Tensor] = None,
                                        update_history: bool = True) -> Dict[str, torch.Tensor]:
        if action_history is None:
            action_history = self.action_history

        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            action_chunk_normalized = self.act_model(
                obs_tokens=obs_tokens,
                action_history=action_history,
                temporal_cond=temporal_cond
            )

        action_chunk = self.normalizer['action'].unnormalize(action_chunk_normalized)
        if update_history:
            self._update_action_history(action_chunk_normalized)
        return {'action': action_chunk}

    def _update_action_history(self, action_chunk_normalized: torch.Tensor):
        if action_chunk_normalized is None:
            return
        if self.action_history is None:
            self.action_history = action_chunk_normalized
            return
        max_history = self.act_model.max_action_history
        self.action_history = torch.cat([
            self.action_history[:, -max_history:, :],
            action_chunk_normalized
        ], dim=1)
        max_total = max_history + self.act_model.action_chunk_size
        if self.action_history.shape[1] > max_total:
            self.action_history = self.action_history[:, -max_total:, :]

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def apply_obs_drop(self, batch):
        if self.obs_drop_type == "gaussian":
            obs_drop_scale = get_scale(
                scheduler=self.obs_drop_schedule_type,
                start=self.gaussian_start,
                end=self.gaussian_end,
                cur_step=self.progress,
                max_step=1,
                ratio=self.gaussian_ratio
            )
        else:
            if self.obs_drop_schedule_type == "constant":
                obs_drop_scale = 1
            elif self.obs_drop_schedule_type == "linear":
                obs_drop_scale = self.progress
            else:
                raise ValueError(f"Unsupported obs_drop_schedule_type {self.obs_drop_schedule_type}")
            obs_drop = self.obs_drop * obs_drop_scale

        if self.obs_drop_type == "gaussian" or (self.obs_drop > 0.0 and obs_drop_scale > 0.0):
            for key in self.obs_drop_keys:
                batch_size = next(iter(batch['obs'].values())).shape[0]
                if key in batch['obs']:
                    if self.obs_drop_type == 'gaussian':
                        assert len(batch['obs'][key].shape) == 5, "Gaussian dropout only supports 5D obs"
                        batch['obs'][key] = gaussian_2d_smoothing_separate(batch['obs'][key], scale=obs_drop_scale)
                    else:
                        drop_obs_mask = torch.rand(batch_size) < obs_drop
                        if self.obs_drop_type == 'zero':
                            batch['obs'][key][drop_obs_mask] = 0.0
                        elif self.obs_drop_type == 'learnable':
                            raise NotImplementedError("learnable obs_drop_type is not yet implemented")
                        elif self.obs_drop_type == 'noise':
                            batch['obs'][key][drop_obs_mask] = self.obs_drop_noise_scale * torch.randn_like(batch['obs'][key][drop_obs_mask])

                if 'extended_obs' in batch and key in batch['extended_obs']:
                    if self.obs_drop_type == 'zero':
                        drop_obs_mask = torch.rand(batch_size) < obs_drop
                        batch['extended_obs'][key][drop_obs_mask] = 0.0
                    elif self.obs_drop_type == 'noise':
                        drop_obs_mask = torch.rand(batch_size) < obs_drop
                        batch['extended_obs'][key][drop_obs_mask] = self.obs_drop_noise_scale * torch.randn_like(batch['extended_obs'][key][drop_obs_mask])

    def _compute_act_loss(self,
                          action: torch.Tensor,
                          obs_tokens: torch.Tensor,
                          temporal_cond: Optional[torch.Tensor],
                          loss_mask=None):
        B = action.shape[0]
        device = action.device
        action_chunk_size = self.act_model.action_chunk_size

        if action.shape[1] < action_chunk_size:
            target_chunk = action
            logger.warning(f"Action sequence length {action.shape[1]} < chunk size {action_chunk_size}")
        else:
            target_chunk = action[:, :action_chunk_size, :]

        action_history = None
        if action.shape[1] > action_chunk_size:
            history_start = max(0, action.shape[1] - action_chunk_size - self.act_model.max_action_history)
            action_history = action[:, history_start:-action_chunk_size, :]
            action_history = self.normalizer['action'].normalize(action_history)

        target_chunk_normalized = self.normalizer['action'].normalize(target_chunk)

        pred_actions = self.act_model(
            obs_tokens=obs_tokens,
            action_history=action_history,
            temporal_cond=temporal_cond
        )

        if pred_actions.shape[1] != target_chunk_normalized.shape[1]:
            min_len = min(pred_actions.shape[1], target_chunk_normalized.shape[1])
            pred_actions = pred_actions[:, :min_len, :]
            target_chunk_normalized = target_chunk_normalized[:, :min_len, :]

        # per-dimension MSE (B, T, A)
        per_sample_loss_all = F.mse_loss(pred_actions, target_chunk_normalized, reduction='none')
        per_sample_loss = reduce(per_sample_loss_all, 'b t a -> b', 'mean')

        # Confidence loss: MSE / gt_mse element-wise (optional), then reduce
        conf_dims = min(self.confidence_action_dims, per_sample_loss_all.shape[-1])
        sel_loss = per_sample_loss_all[..., :conf_dims]  # (B, T, conf_dims)
        if self.normalize_confidence_loss:
            gt_mse_conf = target_chunk_normalized[..., :conf_dims] * target_chunk_normalized[..., :conf_dims]  # (B, T, conf_dims)
            per_sample_loss_conf_all = sel_loss / gt_mse_conf  # element-wise division
        else:
            per_sample_loss_conf_all = sel_loss  # no normalization
        per_sample_loss_conf = reduce(per_sample_loss_conf_all, 'b t a -> b', 'mean')

        # Virtual loss: remaining dimensions
        if per_sample_loss_all.shape[-1] > conf_dims:
            sel_virtual = per_sample_loss_all[..., conf_dims:]
            per_sample_loss_virtual = reduce(sel_virtual, 'b t a -> b', 'mean')
        else:
            per_sample_loss_virtual = torch.zeros(B, device=device, dtype=per_sample_loss_all.dtype)

        if loss_mask is None:
            loss_mask = torch.ones(B, dtype=torch.bool, device=device)
        masked_loss = per_sample_loss[loss_mask]
        loss = masked_loss.mean()
        return loss, per_sample_loss, per_sample_loss_conf, per_sample_loss_virtual

    def _compute_confidence_loss(self,
                                 obs_tokens: torch.Tensor,
                                 act_loss_vec_conf: torch.Tensor,
                                 loss_mask=None):
        if self.confidence_head is None:
            return None, None, None

        log_sigma_raw = self._compute_confidence_logits(obs_tokens)
        if log_sigma_raw is None:
            return None, None, None, None

        log_sigma = log_sigma_raw[loss_mask] if loss_mask is not None else log_sigma_raw
        act_loss_masked = act_loss_vec_conf[loss_mask] if loss_mask is not None else act_loss_vec_conf
        sigma = 1.0 + torch.exp(log_sigma)
        sigma_stats = sigma
        log_sigma_term = torch.log(sigma)
        act_loss_normalized = act_loss_masked
        gamma = self.confidence_gamma
        alpha = self.confidence_alpha
        act_conf_loss = gamma * act_loss_normalized * sigma - alpha * log_sigma_term
        act_conf_loss = act_conf_loss.mean()
        return act_conf_loss, log_sigma, sigma_stats, act_loss_normalized

    def compute_loss(self, batch: Dict[str, torch.Tensor], loss_mask=None):
        action = batch['action']
        T_action = action.shape[1]

        self.apply_obs_drop(batch)

        obs_dict = batch['obs']
        extended_obs_dict = batch.get('extended_obs', None)

        obs_tokens, _ = self._predict_obs_tokens(obs_dict, extended_obs_dict, return_temporal_cond=True)

        if extended_obs_dict is not None and len(self.extented_obs_keys) > 0:
            first_val = next(iter(extended_obs_dict.values()))
            T_extended = first_val.shape[1]
            extend_pad_n = T_action - T_extended if T_action > T_extended else None
            temporal_cond = self.get_temporal_cond(
                extended_obs_dict,
                extended_obs_last_step=T_action,
                extend_obs_pad_after_n=extend_pad_n,
                post_process=True
            )
        else:
            temporal_cond = None
        act_loss, act_loss_vec, act_loss_vec_conf, act_loss_vec_virtual = self._compute_act_loss(
            action, obs_tokens, temporal_cond, loss_mask
        )

        log_sigma = None
        sigma_stats = None
        act_conf_loss = None
        if self.confidence_head is not None:
            print("####here")
            act_conf_loss, log_sigma, sigma_stats, act_loss_normalized = self._compute_confidence_loss(
                obs_tokens, act_loss_vec_conf, loss_mask
            )

        if loss_mask is None:
            loss_mask = torch.ones_like(act_loss_vec, dtype=torch.bool)
        masked_virtual = act_loss_vec_virtual[loss_mask] if loss_mask is not None else act_loss_vec_virtual

        # If confidence-based ACT loss exists, use it for the pose part; otherwise fall back to the plain act_loss.
        # confidence influences all action dims but gradient is stopped for virtual part
        if act_conf_loss is not None:
            weighted_virtual = masked_virtual * sigma_stats.detach()
            virtual_loss_component = weighted_virtual.mean()
            total_loss = virtual_loss_component + act_conf_loss
        else:
            total_loss = act_loss

        loss_dict = {
            'act_loss': act_loss.item(),
            'total_loss': total_loss.item()
        }

        if log_sigma is not None and sigma_stats is not None:
            loss_dict.update({
                'confidence/log_sigma': log_sigma.mean().item(),
                'confidence/log_sigma_min': log_sigma.min().item(),
                'confidence/log_sigma_max': log_sigma.max().item(),
                'confidence/sigma': sigma_stats.mean().item(),
                'confidence/act_virtual_loss': virtual_loss_component.item(),
                'confidence/act_origin_loss': act_loss_normalized.mean().item(),
            })
            if act_conf_loss is not None:
                loss_dict['confidence/act_conf_loss'] = act_conf_loss.item()

        return total_loss, loss_dict

    def get_optimizer(self,
                      lr: float,
                      weight_decay: float,
                      obs_encoder_lr: Optional[float] = None,
                      obs_encoder_weight_decay: Optional[float] = None,
                      betas: Tuple[float, float] = (0.9, 0.95),
                      **kwargs) -> torch.optim.Optimizer:
        if obs_encoder_lr is None:
            obs_encoder_lr = lr
        if obs_encoder_weight_decay is None:
            obs_encoder_weight_decay = weight_decay

        optim_groups = []

        if self.rnn_obs_encoder is not None:
            optim_groups += self.rnn_obs_encoder.get_optim_groups(weight_decay=weight_decay)

        if self.vq is not None:
            optim_groups += [{
                "params": self.vq.parameters(),
                "weight_decay": 0.0,
            }]

        if self.force_encoder is not None:
            optim_groups += [{
                "params": self.force_encoder.parameters(),
                "weight_decay": weight_decay,
            }]

        if self.confidence_head is not None:
            if hasattr(self.confidence_head, 'get_optim_groups'):
                optim_groups += self.confidence_head.get_optim_groups(weight_decay=weight_decay)
            else:
                optim_groups += [{
                    "params": self.confidence_head.parameters(),
                    "weight_decay": weight_decay,
                }]

        backbone_params = []
        other_obs_params = []
        for key, value in self.obs_encoder.named_parameters():
            if key.startswith('key_model_map'):
                backbone_params.append(value)
            else:
                other_obs_params.append(value)
        optim_groups.append({
            "params": backbone_params,
            "weight_decay": obs_encoder_weight_decay,
            "lr": obs_encoder_lr
        })
        optim_groups.append({
            "params": other_obs_params,
            "weight_decay": obs_encoder_weight_decay
        })

        if hasattr(self.act_model, 'get_optim_groups'):
            optim_groups.extend(self.act_model.get_optim_groups(weight_decay=weight_decay))
        else:
            optim_groups.append({
                "params": list(self.act_model.parameters()),
                "weight_decay": weight_decay,
            })

        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas
        )
        return optimizer

    # ------------------------------------------------------------------
    def update_progress(self, progress: float):
        self.progress = progress
