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


class ACTConfidenceDualEncoderStagewisePolicy(BaseImagePolicy):
    def __init__(self,
                 shape_meta: dict,
                 act_encoder: torch.nn.Module,
                 confidence_encoder: Optional[torch.nn.Module],
                 act_model: ACTTransformer,
                 horizon: int,
                 n_action_steps: int,
                 n_obs_steps: int,
                 n_emb: int = 768,
                 use_rnn_obs_encoder: bool = False,
                 rnn_obs_encoder_n_layer: int = 1,
                 rnn_obs_encoder_hidden_dim: int = 512,
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
                 # Freeze encoders
                 freeze_act_encoder: bool = False,
                 freeze_confidence_encoder: bool = False,
                 # Stagewise training: freeze ACT model and confidence head
                 freeze_act_model: bool = False,
                 freeze_confidence_head: bool = False,
                 # Current training stage: 1 or 2
                 training_stage: int = 1,
                 **kwargs):
        super().__init__()
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        self.action_dim = action_shape[0]
        self.action_horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps

        self.shape_meta = shape_meta
        self.act_encoder = act_encoder
        self.confidence_encoder = confidence_encoder if confidence_encoder is not None else act_encoder
        self.obs_encoder = None
        self.act_model = act_model
        self.normalizer = LinearNormalizer()
        self.kwargs = kwargs

        self.freeze_act_encoder = freeze_act_encoder
        self.freeze_confidence_encoder = freeze_confidence_encoder
        self.freeze_act_model = freeze_act_model
        self.freeze_confidence_head = freeze_confidence_head
        self.training_stage = training_stage

        self.n_emb = n_emb
        self.use_amp_for_inference = use_amp_for_inference
        self.amp_dtype = amp_dtype if use_amp_for_inference else None
        self.progress = 0.0

        all_extented_obs_keys = list(shape_meta['extended_obs'].keys()) if 'extended_obs' in shape_meta else []
        self.extented_obs_keys = sorted(all_extented_obs_keys)
        temporal_cond_dim = sum([shape_meta['extended_obs'][extented_obs_key]['shape'][-1]
                                 for extented_obs_key in self.extented_obs_keys])

        self.rnn_obs_encoder = None
        self.rnn_act_encoder = None
        self.rnn_confidence_encoder = None
        
        if use_rnn_obs_encoder:
            assert len(self.extented_obs_keys) > 0, "extended_obs is required when using RNN obs encoder"
            
            self.rnn_act_encoder = RNN(
                input_dim=temporal_cond_dim,
                hidden_dim=rnn_obs_encoder_hidden_dim,
                layer_num=rnn_obs_encoder_n_layer,
                n_emb=n_emb,
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
        logger.info(f"ACTConfidenceDualEncoderStagewisePolicy initialized (stage={training_stage})")
        logger.info(f"  extended_obs_keys: {self.extented_obs_keys}")

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
            if self.rnn_act_encoder is not None:
                temporal_cond = self.rnn_act_encoder(temporal_cond)
            elif self.rnn_confidence_encoder is not None:
                temporal_cond = self.rnn_confidence_encoder(temporal_cond)

        return temporal_cond

    def get_temporal_cond_for_act(self,
                                  extended_obs_dict: Optional[Dict[str, torch.Tensor]],
                                  extended_obs_last_step: Optional[int] = None,
                                  extend_obs_pad_after_n: Optional[int] = None,
                                  allow_partial: bool = False,
                                  post_process: bool = False) -> Optional[torch.Tensor]:
        return self.get_temporal_cond(extended_obs_dict, extended_obs_last_step, 
                                     extend_obs_pad_after_n, allow_partial, post_process)

    def predict_action(self,
                       obs_dict: Dict[str, torch.Tensor],
                       extended_obs_dict: Optional[Dict[str, torch.Tensor]] = None,
                       **kwargs) -> Dict[str, torch.Tensor]:
        return self._predict_action_act(obs_dict, extended_obs_dict)

    def _predict_obs_tokens(self,
                            obs_dict: Dict[str, torch.Tensor],
                            extended_obs_dict: Optional[Dict[str, torch.Tensor]] = None,
                            encoder_type: str = 'act',  # 'act' or 'confidence'
                            return_expanded_obs_tokens: bool = False,
                            return_temporal_cond: bool = False
                            ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        
        if encoder_type == 'act':
            self.obs_encoder = self.act_encoder
            self.rnn_obs_encoder = self.rnn_act_encoder
        else:  # confidence
            self.obs_encoder = self.confidence_encoder
            self.rnn_obs_encoder = self.rnn_confidence_encoder

        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])

        temporal_cond = None
        if encoder_type == 'act' and self.rnn_obs_encoder is not None:
            if extended_obs_dict is not None:
                temporal_cond = self.get_temporal_cond(
                    extended_obs_dict,
                    allow_partial=True,
                    post_process=True
                )
            else:
                if len(self.extented_obs_keys) > 0:
                    extended_obs_dict_local = {}
                    for key in self.extented_obs_keys:
                        if key in obs_dict:
                            extended_obs_dict_local[key] = obs_dict[key]
                    if len(extended_obs_dict_local) > 0:
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

    def predict_obs_tokens(self, *args, encoder_type='act', **kwargs) -> torch.Tensor:
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_obs_tokens(*args, encoder_type=encoder_type, **kwargs)

    def sample_noisy_trajectory(self,
                                B: int = 1,
                                generator=None,
                                return_expanded_noisy_trajectory: bool = False
                                ) -> torch.Tensor:

        noisy_trajectory = torch.zeros((B, 0), dtype=self.dtype, device=self.device)
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
        obs_tokens, temporal_cond = self._predict_obs_tokens(obs_dict, extended_obs_dict, encoder_type='act', return_temporal_cond=True)
        B = obs_tokens.shape[0]
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, self.n_emb)
        action_history = self.action_history
        return self._predict_action_act_from_tokens(obs_tokens=obs_tokens, temporal_cond=temporal_cond, action_history=action_history)

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
            return None, None, None, None

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

    def forward(self, batch: Dict[str, torch.Tensor], loss_mask=None):
        """
        Forward method for DDP compatibility.
        """
        return self.compute_loss(batch, loss_mask)

    def compute_loss(self, batch: Dict[str, torch.Tensor], loss_mask=None):
        """
        Compute total loss using dual encoder architecture with stagewise training:
        
        Stage 1: Train ACT (act_encoder + act_model), freeze Confidence (confidence_encoder + confidence_head)
        Stage 2: Train Confidence (confidence_encoder + confidence_head), freeze ACT (act_encoder + act_model)
        
        Total loss = ACT_loss (stage 1) + Confidence_loss (stage 2)
        """
        action = batch['action']
        T_action = action.shape[1]

        obs_dict = batch['obs']
        extended_obs_dict = batch.get('extended_obs', None)

        # ===== ACT path: use ACT encoder =====
        obs_tokens_act, temporal_cond_act = self._predict_obs_tokens(
            obs_dict, extended_obs_dict, 
            encoder_type='act',
            return_temporal_cond=True
        )

        # ===== Confidence path: use confidence encoder =====
        obs_tokens_conf, temporal_cond_conf = self._predict_obs_tokens(
            obs_dict, extended_obs_dict, 
            encoder_type='confidence',
            return_temporal_cond=True
        )

        # Compute temporal conditioning for ACT
        if extended_obs_dict is not None and len(self.extented_obs_keys) > 0:
            first_val = next(iter(extended_obs_dict.values()))
            T_extended = first_val.shape[1]
            extend_pad_n = T_action - T_extended if T_action > T_extended else None
            temporal_cond_act = self.get_temporal_cond(
                extended_obs_dict,
                extended_obs_last_step=T_action,
                extend_obs_pad_after_n=extend_pad_n,
                post_process=True
            )
        else:
            temporal_cond_act = None

        # ===== Stagewise training: compute loss based on training stage =====
        
        # Stage 1: Train ACT, freeze Confidence
        if self.training_stage == 1:
            # Compute ACT loss using ACT encoder
            act_loss, act_loss_vec, act_loss_vec_conf, act_loss_vec_virtual = self._compute_act_loss(
                action, obs_tokens_act, temporal_cond_act, loss_mask
            )
            
            # Don't compute confidence loss in stage 1 (frozen)
            total_loss = act_loss
            
            loss_dict = {
                'act_loss': act_loss.item(),
                'total_loss': total_loss.item(),
                'training_stage': 1.0
            }
            
        # Stage 2: Train Confidence, freeze ACT
        elif self.training_stage == 2:
            # Don't compute ACT loss in stage 2 (frozen)
            # Instead, use the ACT model to get predictions for confidence training
            
            # Get ACT predictions (but don't backprop through ACT model)
            with torch.no_grad():
                _, _, act_loss_vec_conf, _ = self._compute_act_loss(
                    action, obs_tokens_act, temporal_cond_act, loss_mask
                )
            
            # Compute confidence loss using confidence encoder + confidence head
            log_sigma = None
            sigma_stats = None
            act_conf_loss = None
            
            if self.confidence_head is not None:
                act_conf_loss, log_sigma, sigma_stats, act_loss_normalized = self._compute_confidence_loss(
                    obs_tokens_conf, act_loss_vec_conf, loss_mask
                )
            
            if act_conf_loss is not None:
                total_loss = act_conf_loss
                loss_dict = {
                    'confidence_loss': act_conf_loss.item(),
                    'total_loss': total_loss.item(),
                    'training_stage': 2.0
                }
                
                if log_sigma is not None and sigma_stats is not None:
                    loss_dict.update({
                        'confidence/log_sigma': log_sigma.mean().item(),
                        'confidence/log_sigma_min': log_sigma.min().item(),
                        'confidence/log_sigma_max': log_sigma.max().item(),
                        'confidence/sigma': sigma_stats.mean().item(),
                    })
            else:
                total_loss = torch.tensor(0.0, device=action.device)
                loss_dict = {
                    'confidence_loss': 0.0,
                    'total_loss': total_loss.item(),
                    'training_stage': 2.0
                }
        else:
            raise ValueError(f"Invalid training_stage: {self.training_stage}. Must be 1 or 2.")

        return total_loss, loss_dict

    def get_optimizer(self,
                      lr: float,
                      weight_decay: float,
                      obs_encoder_lr: Optional[float] = None,
                      obs_encoder_weight_decay: Optional[float] = None,
                      confidence_encoder_lr: Optional[float] = None,
                      confidence_encoder_weight_decay: Optional[float] = None,
                      betas: Tuple[float, float] = (0.9, 0.95),
                      **kwargs) -> torch.optim.Optimizer:
        if obs_encoder_lr is None:
            obs_encoder_lr = lr
        if obs_encoder_weight_decay is None:
            obs_encoder_weight_decay = weight_decay
        if confidence_encoder_lr is None:
            confidence_encoder_lr = obs_encoder_lr
        if confidence_encoder_weight_decay is None:
            confidence_encoder_weight_decay = obs_encoder_weight_decay

        optim_groups = []

        # RNN encoders - depends on training stage
        if self.training_stage == 1:
            # Stage 1: train RNN for ACT
            if self.rnn_act_encoder is not None:
                optim_groups += self.rnn_act_encoder.get_optim_groups(weight_decay=weight_decay)
        else:
            # Stage 2: train RNN for Confidence (if separate)
            if self.rnn_confidence_encoder is not None and self.rnn_confidence_encoder is not self.rnn_act_encoder:
                optim_groups += self.rnn_confidence_encoder.get_optim_groups(weight_decay=weight_decay)

        # Confidence head - only train in stage 2
        if self.confidence_head is not None and self.training_stage == 2:
            if hasattr(self.confidence_head, 'get_optim_groups'):
                optim_groups += self.confidence_head.get_optim_groups(weight_decay=weight_decay)
            else:
                optim_groups += [{
                    "params": self.confidence_head.parameters(),
                    "weight_decay": weight_decay,
                }]

        # ACT encoder - only train in stage 1
        if self.training_stage == 1:
            act_encoder_params = []
            other_obs_params = []
            for key, value in self.act_encoder.named_parameters():
                if key.startswith('key_model_map'):
                    act_encoder_params.append(value)
                else:
                    other_obs_params.append(value)
            
            if not self.freeze_act_encoder:
                optim_groups.append({
                    "params": act_encoder_params,
                    "weight_decay": obs_encoder_weight_decay,
                    "lr": obs_encoder_lr
                })
                optim_groups.append({
                    "params": other_obs_params,
                    "weight_decay": obs_encoder_weight_decay
                })

        # Confidence encoder - only train in stage 2
        if self.training_stage == 2 and self.confidence_encoder is not self.act_encoder:
            conf_encoder_params = []
            conf_other_params = []
            for key, value in self.confidence_encoder.named_parameters():
                if key.startswith('key_model_map'):
                    conf_encoder_params.append(value)
                else:
                    conf_other_params.append(value)
            
            if not self.freeze_confidence_encoder:
                optim_groups.append({
                    "params": conf_encoder_params,
                    "weight_decay": confidence_encoder_weight_decay,
                    "lr": confidence_encoder_lr
                })
                optim_groups.append({
                    "params": conf_other_params,
                    "weight_decay": confidence_encoder_weight_decay
                })

        # ACT model - only train in stage 1
        if self.training_stage == 1:
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

    def set_training_stage(self, stage: int):
        """Set the training stage (1 or 2)."""
        assert stage in [1, 2], f"Invalid stage: {stage}"
        self.training_stage = stage
        #logger.info(f"Training stage set to {stage}")
