import copy
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from ImplicitRDP.model.common.normalizer import LinearNormalizer
from ImplicitRDP.policy.base_image_policy import BaseImagePolicy
from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.model.vision.transformer_obs_encoder import TransformerObsEncoder
from ImplicitRDP.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from ImplicitRDP.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from ImplicitRDP.model.vision.timm_obs_encoder import TimmObsEncoder
from ImplicitRDP.model.act.act_transformer import ACTTransformer
from ImplicitRDP.model.confidence.transformer import TransformerConfidenceHead
from ImplicitRDP.model.force.rnn import RNN

from loguru import logger


class HybridDiffusionACTPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            diffusion_model: Union[TransformerForDiffusion],
            act_model: ACTTransformer,
            obs_encoder: Union[TransformerObsEncoder, MultiImageObsEncoder, TimmObsEncoder],
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            input_pertub=0.1,
            n_emb: int=768,
            use_rnn_obs_encoder=False,
            rnn_obs_encoder_n_layer=1,
            rnn_obs_encoder_hidden_dim=512,
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
            inference_head: str = 'diffusion',
            use_confidence_based_head_selection: bool = True,
            confidence_threshold_for_head_selection: float = 0.0,
            confidence_action_dims: int = 6,
            normalize_confidence_loss: bool = True,
            # Two-stage training: first train ACT only, then add DP
            dp_warmup_epochs: int = 300,  # Epochs to train ACT only (DP weight = 0), then switch to DP (DP weight = 1)
            **kwargs):
        super().__init__()
        print("we get confidence_head_type: ", confidence_head_type)
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = horizon
        
        obs_shape = obs_encoder.output_shape()
        assert obs_shape[-1] == n_emb
        obs_tokens = obs_shape[-2] if len(obs_shape) == 3 else n_obs_steps
        
        all_extented_obs_keys = list(shape_meta['extended_obs'].keys()) if 'extended_obs' in shape_meta else []
        self.extented_obs_keys = sorted(all_extented_obs_keys)
        temporal_cond_dim = sum([shape_meta['extended_obs'][extented_obs_key]['shape'][-1] 
                                 for extented_obs_key in self.extented_obs_keys])
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
        self.diffusion_model = diffusion_model
        self.act_model = act_model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.input_pertub = input_pertub
        self.n_emb = n_emb
        self.use_amp_for_inference = use_amp_for_inference
        self.kwargs = kwargs
        self.confidence_gamma = confidence_gamma
        self.confidence_alpha = confidence_alpha
        self.confidence_log_sigma_min = confidence_log_sigma_min
        self.confidence_log_sigma_max = confidence_log_sigma_max
        self.use_confidence_head = use_confidence_head
        self.confidence_head_type = (confidence_head_type or 'transformer').lower()
        self.confidence_action_dims = confidence_action_dims
        self.normalize_confidence_loss = normalize_confidence_loss
        
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

        self.available_heads = ['diffusion', 'act']
        if inference_head not in self.available_heads:
            raise ValueError(f"inference_head must be one of {self.available_heads}, got {inference_head}")
        self.inference_head = inference_head
        
        # Confidence-based head selection
        self.use_confidence_based_head_selection = use_confidence_based_head_selection
        self.confidence_threshold_for_head_selection = confidence_threshold_for_head_selection
        if self.use_confidence_based_head_selection:
            assert self.confidence_head is not None, "confidence_head must be enabled to use confidence-based head selection"
            logger.info(f"Confidence-based head selection enabled with threshold: {confidence_threshold_for_head_selection}")
            logger.info("Head selection logic: confidence = log(sigma) where sigma = 1 + exp(log_sigma), if confidence > threshold -> use ACT, else -> use Diffusion")
        
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        if use_amp_for_inference:
            if amp_dtype == 'bf16' and torch.cuda.is_bf16_supported():
                logger.info(f"Using bf16 for AMP during inference")
                self.amp_dtype = torch.bfloat16
            elif amp_dtype == 'fp16':
                logger.info(f"Using fp16 for AMP during inference")
                self.amp_dtype = torch.float16
            else:
                logger.warning(f"AMP dtype {amp_dtype} not supported, falling back to fp16")
                self.amp_dtype = torch.float16
        else:
            self.amp_dtype = None
        
        self.action_history = None
        
        # Two-stage training: first train ACT only, then add DP
        self.dp_warmup_epochs = dp_warmup_epochs
        self.current_epoch = 0  # Will be updated via update_progress or set_epoch
        
        logger.info(f"HybridDiffusionACTPolicy initialized (two independent heads; no confidence or fusion)")
        if dp_warmup_epochs > 0:
            logger.info(f"Two-stage training: ACT only for {dp_warmup_epochs} epochs, then DP training starts")
        else:
            logger.info("Training both ACT and DP from the start")
    
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
            # inline post-processing previously implemented in _post_process_temporal_cond
            processed_cond = temporal_cond

            if self.rnn_obs_encoder is not None:
                processed_cond = self.rnn_obs_encoder(processed_cond)

            temporal_cond = processed_cond
        
        return temporal_cond
    
    def predict_action(self, 
                       *args,
                       **kwargs) -> Dict[str, torch.Tensor]:

        if self.use_confidence_based_head_selection and self.confidence_head is not None:
            obs_dict = args[0] if len(args) > 0 else kwargs.get('obs_dict')
            extended_obs_dict = args[1] if len(args) > 1 else kwargs.get('extended_obs_dict', None)
            
            obs_tokens, temporal_cond = self._predict_obs_tokens(obs_dict, extended_obs_dict, return_temporal_cond=True)
            
            log_sigma = self._compute_confidence_logits(obs_tokens)
            
            if log_sigma is not None:
                sigma = 1.0 + torch.exp(log_sigma)
                confidence = torch.log(sigma)
                
                if confidence.numel() > 1:
                    mean_confidence = confidence.mean().item()
                    use_act = mean_confidence > self.confidence_threshold_for_head_selection
                    logger.debug(f"Batch confidence-based head selection: mean_confidence={mean_confidence:.4f}, threshold={self.confidence_threshold_for_head_selection:.4f}, using {'ACT' if use_act else 'Diffusion'}")
                else:
                    confidence_value = confidence.item()
                    use_act = confidence_value > self.confidence_threshold_for_head_selection
                    logger.debug(f"Confidence-based head selection: confidence={confidence_value:.4f}, threshold={self.confidence_threshold_for_head_selection:.4f}, using {'ACT' if use_act else 'Diffusion'}")
                
                if use_act:
                    action_history = self.action_history
                    return self._predict_action_act_from_tokens(
                        obs_tokens=obs_tokens,
                        temporal_cond=temporal_cond,
                        action_history=action_history
                    )
                else:
                    return self.predict_action_diffusion(*args, **kwargs)
            else:
                logger.warning("Confidence computation returned None, falling back to default inference_head")
                inference_head = getattr(self, 'inference_head', 'diffusion')
        
        inference_head = getattr(self, 'inference_head', 'diffusion')
        
        if inference_head == 'diffusion':
            return self.predict_action_diffusion(*args, **kwargs)
        elif inference_head == 'act':
            obs_dict = args[0] if len(args) > 0 else kwargs.get('obs_dict')
            extended_obs_dict = args[1] if len(args) > 1 else kwargs.get('extended_obs_dict', None)
            return self.predict_action_act(obs_dict, extended_obs_dict)
        else:
            raise ValueError(f"Unknown inference_head: {inference_head}. Must be 'diffusion' or 'act'.")

    def predict_action_diffusion(self,
                                 obs_dict: Dict[str, torch.Tensor],
                                 dataset_obs_temporal_downsample_ratio: int = 1,
                                 extended_obs_dict: Optional[Dict[str, torch.Tensor]] = None,
                                 inpainting_action: Optional[torch.Tensor] = None,
                                 inpainting_resample_steps: int = 1,
                                 sample_per_trajectory: Optional[int] = None,
                                 inference_batch_size: Optional[int] = None,
                                 return_noisy_trajectory: bool = False) -> Dict[str, torch.Tensor]:
        return self._predict_action_diffusion(
            obs_dict, 
            dataset_obs_temporal_downsample_ratio,
            extended_obs_dict,
            inpainting_action,
            inpainting_resample_steps,
            sample_per_trajectory,
            inference_batch_size,
            return_noisy_trajectory
        )

    def predict_action_act(self,
                           obs_dict: Dict[str, torch.Tensor],
                           extended_obs_dict: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        return self._predict_action_act(obs_dict, extended_obs_dict)

    def _predict_obs_tokens(self,
                            obs_dict: Dict[str, torch.Tensor],
                            extended_obs_dict: Optional[Dict[str, torch.Tensor]] = None,
                            return_expanded_obs_tokens: bool = False,
                            return_temporal_cond: bool = False,
                            return_action_history: bool = False
                            ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Normalize/encode obs and optionally return temporal_cond (post-processed) or action history.
        Mirrors act_confidence_policy / diffusion style.
        """
        # normalize and encode observations (inlined from former `_encode_obs`)
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))

        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])

        # compute temporal_cond if RNN obs encoder is used (inlined from former `_encode_obs`)
        temporal_cond = None
        if self.rnn_obs_encoder is not None:
            if extended_obs_dict is not None:
                temporal_cond = self.get_temporal_cond(
                    extended_obs_dict,
                    allow_partial=True,
                    post_process=True
                )
            elif len(self.extented_obs_keys) > 0:
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

        if return_action_history:
            return obs_tokens, temporal_cond, self.action_history

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
        inference_head = getattr(self, 'inference_head', 'diffusion')
        if inference_head == 'act':
            noisy_trajectory = torch.zeros(
                (B, 0),
                dtype=self.dtype,
                device=self.device
            )
        else:
            noisy_trajectory = torch.randn(
                size=(B, self.action_horizon, self.action_dim),
                dtype=self.dtype,
                device=self.device,
                generator=generator
            )
            noisy_trajectory = noisy_trajectory.reshape(B, -1)

        if return_expanded_noisy_trajectory:
            noisy_trajectory = noisy_trajectory.unsqueeze(1).expand(-1, self.action_horizon, -1)
        return noisy_trajectory

    def conditional_sample(self,
                           condition_data: torch.Tensor,
                           condition_mask: torch.Tensor,
                           cond: Optional[torch.Tensor] = None,
                           temporal_cond: Optional[torch.Tensor] = None,
                           noisy_trajectory: Optional[torch.Tensor] = None,
                           generator=None,
                           inpainting_mask: Optional[torch.Tensor] = None,
                           inpainting_data: Optional[torch.Tensor] = None,
                           inpainting_resample_steps: int = 1,
                           sample_per_trajectory: Optional[int] = None,
                           inference_batch_size: Optional[int] = None,
                           return_noisy_trajectory: bool = False,
                           **kwargs):
        model = self.diffusion_model
        scheduler = self.noise_scheduler

        if noisy_trajectory is None:
            trajectory = torch.randn(
                size=condition_data.shape,
                dtype=condition_data.dtype,
                device=condition_data.device,
                generator=generator)
        else:
            trajectory = noisy_trajectory

        batch_size = trajectory.shape[0]
        if sample_per_trajectory is not None:
            trajectory = trajectory.unsqueeze(1).repeat(1, sample_per_trajectory, 1, 1)
            trajectory[:, :, -1] = torch.randn_like(trajectory[:, :, -1])
            trajectory = trajectory.reshape(-1, *trajectory.shape[2:])

            condition_data = condition_data.unsqueeze(1).repeat(1, sample_per_trajectory, 1, 1)
            condition_data = condition_data.reshape(-1, *condition_data.shape[2:])

            condition_mask = condition_mask.unsqueeze(1).repeat(1, sample_per_trajectory, 1, 1)
            condition_mask = condition_mask.reshape(-1, *condition_mask.shape[2:])

            if cond is not None:
                cond = cond.unsqueeze(1).repeat(1, sample_per_trajectory, 1, 1)
                cond = cond.reshape(-1, *cond.shape[2:])

            if temporal_cond is not None:
                if len(temporal_cond.shape) == 3:
                    temporal_cond = temporal_cond.unsqueeze(1).repeat(1, sample_per_trajectory, 1, 1)
                elif len(temporal_cond.shape) == 4:
                    temporal_cond = temporal_cond.unsqueeze(1).repeat(1, sample_per_trajectory, 1, 1, 1)
                else:
                    raise ValueError(f"Invalid temporal_cond shape: {temporal_cond.shape}")
                temporal_cond = temporal_cond.reshape(-1, *temporal_cond.shape[2:])

            if inpainting_mask is not None and inpainting_data is not None:
                inpainting_mask = inpainting_mask.unsqueeze(1).repeat(1, sample_per_trajectory, 1, 1)
                inpainting_mask = inpainting_mask.reshape(-1, *inpainting_mask.shape[2:])

                inpainting_data = inpainting_data.unsqueeze(1).repeat(1, sample_per_trajectory, 1, 1)
                inpainting_data = inpainting_data.reshape(-1, *inpainting_data.shape[2:])

        noisy_trajectory = trajectory.clone()

        if inference_batch_size is None:
            inference_batch_size = trajectory.shape[0]
        n_inference = trajectory.shape[0] // inference_batch_size
        trajectory_list = []
        for i in range(n_inference):
            start = i * inference_batch_size
            end = start + inference_batch_size
            trajectory_batch = trajectory[start:end]
            condition_data_batch = condition_data[start:end]
            condition_mask_batch = condition_mask[start:end]
            cond_batch = None if cond is None else cond[start:end]
            if temporal_cond is None:
                temporal_cond_batch = None
            else:
                temporal_cond_batch = temporal_cond[start:end]

            if inpainting_mask is not None and inpainting_data is not None:
                inpainting_mask_batch = inpainting_mask[start:end]
                inpainting_data_batch = inpainting_data[start:end]
            else:
                inpainting_mask_batch = None
                inpainting_data_batch = None

            scheduler.set_timesteps(self.num_inference_steps)
            timesteps = scheduler.timesteps
            num_steps = len(timesteps)

            for step_idx in range(num_steps):
                trajectory_batch[condition_mask_batch] = condition_data_batch[condition_mask_batch]

                if inpainting_mask_batch is not None and inpainting_data_batch is not None:
                    if step_idx < num_steps - 1 and (step_idx + 1) % inpainting_resample_steps == 0:
                        next_t = timesteps[step_idx + 1]
                        noise = torch.randn_like(inpainting_data_batch)
                        noisy_inpainting = scheduler.add_noise(inpainting_data_batch, noise, next_t)
                        trajectory_batch[inpainting_mask_batch] = noisy_inpainting[inpainting_mask_batch]
                    else:
                        trajectory_batch[inpainting_mask_batch] = inpainting_data_batch[inpainting_mask_batch]

                t = timesteps[step_idx]

                if self.rnn_obs_encoder is not None and temporal_cond_batch is not None:
                    temporal_cond_tokens_batch = self.rnn_obs_encoder(temporal_cond_batch)
                else:
                    temporal_cond_tokens_batch = None

                model_output = model(
                    trajectory_batch,
                    t,
                    cond=cond_batch,
                    temporal_cond=temporal_cond_tokens_batch if temporal_cond_tokens_batch is not None else temporal_cond_batch,
                )

                trajectory_batch = scheduler.step(
                    model_output,
                    t,
                    trajectory_batch,
                    generator=generator,
                    **kwargs
                ).prev_sample

            trajectory_batch[condition_mask_batch] = condition_data_batch[condition_mask_batch]
            if inpainting_mask_batch is not None and inpainting_data_batch is not None:
                trajectory_batch[inpainting_mask_batch] = inpainting_data_batch[inpainting_mask_batch]
            trajectory_list.append(trajectory_batch)

        trajectory_list = torch.cat(trajectory_list, dim=0)
        if sample_per_trajectory is not None:
            trajectory_list = trajectory_list.reshape(batch_size, sample_per_trajectory, *trajectory_list.shape[1:])
            noisy_trajectory = noisy_trajectory.reshape(batch_size, sample_per_trajectory, *noisy_trajectory.shape[1:])

        if return_noisy_trajectory:
            return trajectory_list, noisy_trajectory
        else:
            return trajectory_list

    # `_prepare_reactive_temporal_cond` removed; temporal condition preparation is inlined where needed

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
        inference_head = getattr(self, 'inference_head', 'diffusion')
        obs_tokens_and_noisy_trajectory = torch.cat(obs_tokens_and_noisy_trajectory_list, dim=0)
        B = obs_tokens_and_noisy_trajectory.shape[0]

        expected_noise_feature_dim = self.action_horizon * self.action_dim
        actual_feature_dim = obs_tokens_and_noisy_trajectory.shape[-1]
        
        if actual_feature_dim >= expected_noise_feature_dim and (actual_feature_dim - expected_noise_feature_dim) % self.n_emb == 0:
            obs_tokens_flat = obs_tokens_and_noisy_trajectory[:, :-expected_noise_feature_dim]
            noisy_flat = obs_tokens_and_noisy_trajectory[:, -expected_noise_feature_dim:]
            noisy_trajectory = noisy_flat.to(self.device).reshape(B, self.action_horizon, self.action_dim)
        elif actual_feature_dim % self.n_emb == 0:
            obs_tokens_flat = obs_tokens_and_noisy_trajectory
            noisy_trajectory = torch.zeros(
                (B, self.action_horizon, 0),
                dtype=self.dtype,
                device=self.device
            )

        obs_tokens = obs_tokens_flat.to(self.device).reshape(B, -1, self.n_emb)

        # Prepare temporal_cond similar to diffusion/act policy: compute per-sample padded/truncated temporal cond
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
                        extend_obs_pad_after_n=extend_obs_pad_after_n
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
                    # For list mode (reactive transformer), temporal_cond is raw (not post-processed)
                    # We need to post-process it manually to create temporal_cond_tokens for ACT
                    # while keeping temporal_cond raw for diffusion
                    B_concat, T, D = temporal_cond.shape
                    
                    temporal_cond_encoded = temporal_cond

                    if self.rnn_obs_encoder is not None:
                        temporal_cond_tokens = self.rnn_obs_encoder(temporal_cond_encoded)
                    else:
                        temporal_cond_tokens = temporal_cond_encoded

        if self.use_confidence_based_head_selection and self.confidence_head is not None:
            log_sigma = self._compute_confidence_logits(obs_tokens, None)
            
            if log_sigma is not None:
                sigma = 1.0 + torch.exp(log_sigma)
                confidence = torch.log(sigma)
                
                if confidence.numel() > 1:
                    mean_confidence = confidence.mean().item()
                    use_act = mean_confidence > self.confidence_threshold_for_head_selection
                    logger.debug(f"Batch confidence-based head selection (reactive): mean_confidence={mean_confidence:.4f}, threshold={self.confidence_threshold_for_head_selection:.4f}, using {'ACT' if use_act else 'Diffusion'}")
                else:
                    confidence_value = confidence.item()
                    use_act = confidence_value > self.confidence_threshold_for_head_selection
                    logger.debug(f"Confidence-based head selection (reactive): confidence={confidence_value:.4f}, threshold={self.confidence_threshold_for_head_selection:.4f}, using {'ACT' if use_act else 'Diffusion'}")
                
                inference_head = 'act' if use_act else 'diffusion'
                
                if use_act and noisy_trajectory.shape[-1] > 0:
                    noisy_trajectory = torch.zeros(
                        (B, self.action_horizon, 0),
                        dtype=self.dtype,
                        device=self.device
                    )
            else:
                logger.warning("Confidence computation returned None in reactive transformer, falling back to default inference_head")

        if inference_head == 'diffusion':
            return self._predict_from_obs_tokens_and_noisy_trajectory_diffusion(
                obs_tokens,
                noisy_trajectory,
                temporal_cond,
                extended_obs_last_step_list,
                dataset_obs_temporal_downsample_ratio,
                extend_obs_pad_after,
                extend_obs_pad_after_n,
                inpainting_action,
                inpainting_resample_steps,
                sample_per_trajectory,
                inference_batch_size,
                return_noisy_trajectory
            )
        elif inference_head == 'act':
            return self._predict_from_obs_tokens_and_noisy_trajectory_act(
                obs_tokens,
                temporal_cond_tokens if temporal_cond_tokens is not None else temporal_cond,
                extended_obs_last_step_list,
                dataset_obs_temporal_downsample_ratio,
                sample_per_trajectory,
                return_noisy_trajectory
            )
        else:
            raise ValueError(f"Unknown inference_head {inference_head}")

    def _predict_from_obs_tokens_and_noisy_trajectory_diffusion(self,
                                obs_tokens: torch.Tensor,
                                noisy_trajectory: torch.Tensor,
                                temporal_cond: Optional[torch.Tensor],
                                extended_obs_last_step_list: List[int],
                                dataset_obs_temporal_downsample_ratio: int,
                                extend_obs_pad_after: bool,
                                extend_obs_pad_after_n: Optional[int],
                                inpainting_action: torch.Tensor = None,
                                inpainting_resample_steps: int = 1,
                                sample_per_trajectory: Optional[int] = None,
                                inference_batch_size: Optional[int] = None,
                                return_noisy_trajectory: bool = False):
        B = obs_tokens.shape[0]
        if temporal_cond is None:
            cond_len = self.action_horizon
        else:
            cond_len = temporal_cond.shape[1]

        cond_data = torch.zeros(
            size=(B, cond_len, self.action_dim),
            device=self.device,
            dtype=self.dtype
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        noisy_trajectory = noisy_trajectory[:, :cond_len, :]

        if inpainting_action is not None:
            normalized_inpainting_action = self.normalizer['action'].normalize(inpainting_action)
            inpainting_data = torch.zeros_like(cond_data)
            inpainting_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            n_inpainting_step = normalized_inpainting_action.shape[1]
            inpainting_data[:, :n_inpainting_step, :] = normalized_inpainting_action
            inpainting_mask[:, :n_inpainting_step, :] = True
        else:
            inpainting_data = None
            inpainting_mask = None

        result = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            cond=obs_tokens,
            temporal_cond=temporal_cond,
            noisy_trajectory=noisy_trajectory if noisy_trajectory.numel() > 0 else None,
            inpainting_data=inpainting_data,
            inpainting_mask=inpainting_mask,
            inpainting_resample_steps=inpainting_resample_steps,
            sample_per_trajectory=sample_per_trajectory,
            inference_batch_size=inference_batch_size,
            return_noisy_trajectory=return_noisy_trajectory,
            **self.kwargs
        )

        if return_noisy_trajectory:
            nsample, noisy_result = result
        else:
            nsample = result
            noisy_result = None

        if sample_per_trajectory is not None:
            assert nsample.shape == (B, sample_per_trajectory, cond_len, self.action_dim)
        else:
            assert nsample.shape == (B, cond_len, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)

        To = self.n_obs_steps * dataset_obs_temporal_downsample_ratio
        start = To - 1
        end = start + self.n_action_steps
        if sample_per_trajectory is not None:
            action = action_pred[:, :, start:end]
        else:
            action = action_pred[:, start:end]

        results = []
        for i in range(B):
            if extend_obs_pad_after:
                results.append({
                    'action': action[i: i + 1],
                    'action_pred': action_pred[i: i + 1],
                })
            else:
                if sample_per_trajectory is not None:
                    results.append({
                        'action': action[i: i + 1, :, :extended_obs_last_step_list[i] - start, :],
                        'action_pred': action_pred[i: i + 1, :, :extended_obs_last_step_list[i], :],
                    })
                else:
                    results.append({
                        'action': action[i: i + 1, :extended_obs_last_step_list[i] - start, :],
                        'action_pred': action_pred[i: i + 1, :extended_obs_last_step_list[i], :],
                    })

        if return_noisy_trajectory:
            return results, noisy_result
        else:
            return results

    def _predict_from_obs_tokens_and_noisy_trajectory_act(self,
                                obs_tokens: torch.Tensor,
                                temporal_cond: Optional[torch.Tensor],
                                extended_obs_last_step_list: List[int],
                                dataset_obs_temporal_downsample_ratio: int,
                                sample_per_trajectory: Optional[int],
                                return_noisy_trajectory: bool):
        B = obs_tokens.shape[0]
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
    
    def _predict_action_diffusion(self, 
                                  obs_dict: Dict[str, torch.Tensor],
                                  dataset_obs_temporal_downsample_ratio: int = 1,
                                  extended_obs_dict: Optional[Dict[str, torch.Tensor]] = None,
                                  inpainting_action: Optional[torch.Tensor] = None,
                                  inpainting_resample_steps: int = 1,
                                  sample_per_trajectory: Optional[int] = None,
                                  inference_batch_size: Optional[int] = None,
                                  return_noisy_trajectory: bool = False) -> Dict[str, torch.Tensor]:
        obs_tokens, temporal_cond = self._predict_obs_tokens(obs_dict, extended_obs_dict, return_temporal_cond=True)
        B = obs_tokens.shape[0]
        device = obs_tokens.device
        dtype = obs_tokens.dtype
        
        if temporal_cond is None:
            cond_data = torch.zeros(
                size=(B, self.action_horizon, self.action_dim), 
                device=device, 
                dtype=dtype
            )
        else:
            cond_data = torch.zeros(
                size=(B, temporal_cond.shape[1], self.action_dim), 
                device=device, 
                dtype=dtype
            )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        
        if inpainting_action is not None:
            normalized_inpainting_action = self.normalizer['action'].normalize(inpainting_action)
            
            inpainting_data = torch.zeros_like(cond_data)
            inpainting_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            n_inpainting_step = normalized_inpainting_action.shape[1]
            inpainting_data[:, :n_inpainting_step, :] = normalized_inpainting_action
            inpainting_mask[:, :n_inpainting_step, :] = True
        else:
            inpainting_data = None
            inpainting_mask = None
        
        trajectory = torch.randn(
            size=cond_data.shape,
            dtype=dtype,
            device=device
        )
        
        trajectory = self.normalizer['action'].normalize(trajectory)
        
        if inpainting_data is not None:
            trajectory[inpainting_mask] = inpainting_data[inpainting_mask]
        
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        timesteps = self.noise_scheduler.timesteps
        
        for step_idx, t in enumerate(timesteps):
            trajectory[cond_mask] = cond_data[cond_mask]
            
            if inpainting_mask is not None and step_idx < inpainting_resample_steps:
                trajectory[inpainting_mask] = inpainting_data[inpainting_mask]
            
            t_tensor = t.expand(B).to(device)
            
            with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
                model_output = self.diffusion_model(
                    sample=trajectory,
                    timestep=t_tensor,
                    cond=obs_tokens,
                    temporal_cond=temporal_cond
                )
            
            trajectory = self.noise_scheduler.step(
                model_output, t, trajectory, return_dict=False
            )[0]

        trajectory[cond_mask] = cond_data[cond_mask]
        
        action_pred = self.normalizer['action'].unnormalize(trajectory)
        
        To = self.n_obs_steps * dataset_obs_temporal_downsample_ratio
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        
        if return_noisy_trajectory:
            result['noisy_trajectory'] = None 
        
        return result
    
    def _predict_action_act(self, 
                            obs_dict: Dict[str, torch.Tensor],
                            extended_obs_dict: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        obs_tokens, temporal_cond, action_history = self._predict_obs_tokens(
            obs_dict, extended_obs_dict, return_action_history=True
        )
        return self._predict_action_act_from_tokens(
            obs_tokens=obs_tokens,
            temporal_cond=temporal_cond,
            action_history=action_history
        )

    def _predict_action_act_from_tokens(self,
                                        obs_tokens: torch.Tensor,
                                        temporal_cond: Optional[torch.Tensor],
                                        action_history: Optional[torch.Tensor] = None,
                                        update_history: bool = True
                                        ) -> Dict[str, torch.Tensor]:
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

    
    def update_progress(self, progress):
        self.progress = progress
    
    def set_epoch(self, epoch: int):
        """Set current epoch for DP loss weight scheduling"""
        self.current_epoch = epoch
    
    def compute_loss(self, batch: Dict[str, torch.Tensor], loss_mask=None):
        action = batch['action'] 
        T_action = action.shape[1] 

        if 'extended_obs' in batch and len(self.extented_obs_keys) > 0:
            for key, value in batch['extended_obs'].items():
                if key in self.extented_obs_keys:
                    T_extended = value.shape[1]
                    if T_extended > T_action:
                        batch['extended_obs'][key] = value[:, -T_action:, ...]
                    elif T_extended < T_action:
                        last_step = value[:, -1:, ...]
                        padding = last_step.repeat(1, T_action - T_extended, *([1] * (len(value.shape) - 2)))
                        batch['extended_obs'][key] = torch.cat([value, padding], dim=1)

        trajectory, noise, timesteps, pred = self.model_forward(batch)
        
        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            target = self.noise_scheduler.get_velocity(trajectory, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        
        B = trajectory.shape[0]
        device = trajectory.device
        if loss_mask is None:
            loss_mask = torch.ones(B, dtype=torch.bool, device=device)
        
        diffusion_loss_vec = F.mse_loss(pred, target, reduction='none')
        diffusion_loss_vec = reduce(diffusion_loss_vec, 'b ... -> b', 'mean')
        diffusion_loss_masked = diffusion_loss_vec[loss_mask]
        diffusion_loss_mean = diffusion_loss_masked.mean()
        
        obs_dict = batch['obs']
        extended_obs_dict = batch.get('extended_obs', None)
        obs_tokens, temporal_cond = self._predict_obs_tokens(obs_dict, extended_obs_dict, return_temporal_cond=True)
        act_loss, act_loss_vec, act_loss_vec_conf, act_loss_vec_virtual = self._compute_act_loss(
            action, obs_tokens, temporal_cond, loss_mask
        )
        
        sigma_stats = None
        log_sigma = None
        act_conf_loss = None
        log_sigma_raw = None
        if self.confidence_head is not None:
            print("##############here")
            log_sigma_raw = self._compute_confidence_logits(obs_tokens)

        if log_sigma_raw is not None:
            log_sigma = log_sigma_raw[loss_mask] if loss_mask is not None else log_sigma_raw
            act_loss_masked = act_loss_vec_conf[loss_mask] if loss_mask is not None else act_loss_vec_conf
            sigma = 1.0 + torch.exp(log_sigma)
            log_sigma_term = torch.log(sigma)
            sigma_stats = sigma
            act_loss_normalized = act_loss_masked
            gamma = self.confidence_gamma
            alpha = self.confidence_alpha
            act_conf_loss = gamma * act_loss_normalized * sigma - alpha * log_sigma_term
            act_conf_loss = act_conf_loss.mean()

        if loss_mask is None:
            loss_mask = torch.ones_like(act_loss_vec, dtype=torch.bool)
        masked_virtual = act_loss_vec_virtual[loss_mask] if loss_mask is not None else act_loss_vec_virtual

        if act_conf_loss is not None:
            weighted_virtual = masked_virtual * sigma_stats.detach()
            virtual_loss_component = weighted_virtual.mean()
            act_loss_component = virtual_loss_component + act_conf_loss
        else:
            act_loss_component = act_loss
        
        # Two-stage training: first train ACT only, then add DP
        # Stage 1 (epochs < dp_warmup_epochs): Only train ACT (dp_loss_weight = 0)
        # Stage 2 (epochs >= dp_warmup_epochs): Train both ACT and DP (dp_loss_weight = 1)
        if self.dp_warmup_epochs > 0 and self.confidence_head is not None:
            if self.current_epoch < self.dp_warmup_epochs:
                dp_loss_weight = 0.0  # Only train ACT
            else:
                dp_loss_weight = 1.0  # Train both ACT and DP
        else:
            dp_loss_weight = 1.0  # No warmup, train both from the start
        
        total_loss = diffusion_loss_mean * dp_loss_weight + act_loss_component
        
        loss_dict = {
            'diffusion_loss': diffusion_loss_mean.item(),
            'act_loss': act_loss.item(),  # act_loss already mean over mask
            'total_loss': total_loss.item(),
            'dp_loss_weight': dp_loss_weight
        }
        
        if log_sigma is not None:
            sigma_mean = sigma_stats.mean().item() if sigma_stats is not None else 0.0
            loss_dict.update({
                'confidence/log_sigma': log_sigma.mean().item(),
                'confidence/log_sigma_min': log_sigma.min().item(),
                'confidence/log_sigma_max': log_sigma.max().item(),
                'confidence/sigma': sigma_mean,
                'confidence/act_virtual_loss': virtual_loss_component.item(),
                'confidence/act_conf_loss': act_conf_loss.item(),
                'confidence/act_origin_loss': act_loss_normalized.mean().item(),
            })
        
        return total_loss, loss_dict
    
    def model_forward(self, batch, noise=None, timesteps=None):
        # normalize input
        assert 'valid_mask' not in batch

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        trajectory = nactions
        B = nactions.shape[0]

        if self.rnn_obs_encoder is not None or self.diffusion_model.use_rnn_head:
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
        
        if noise is None:
            noise = torch.randn(trajectory.shape, device=trajectory.device)
        else:
            assert self.input_pertub == 0.0, "input_pertub is not supported when noise is provided"
        
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        if timesteps is None:
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (nactions.shape[0],), device=trajectory.device
            ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)
        
        if self.rnn_obs_encoder is not None and temporal_cond is not None:
            temporal_cond_tokens = self.rnn_obs_encoder(temporal_cond)
        else:
            temporal_cond_tokens = None
        
        # Predict the noise residual
        pred = self.diffusion_model(
            noisy_trajectory,
            timesteps, 
            cond=obs_tokens,
            temporal_cond=temporal_cond_tokens if temporal_cond_tokens is not None else temporal_cond
        )
        return trajectory, noise, timesteps, pred
    
    
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
        
        pred_actions = self.act_model.forward(
            obs_tokens=obs_tokens,
            action_history=action_history,
            temporal_cond=temporal_cond
        )
        
        if pred_actions.shape[1] != target_chunk_normalized.shape[1]:
            min_len = min(pred_actions.shape[1], target_chunk_normalized.shape[1])
            pred_actions = pred_actions[:, :min_len, :]
            target_chunk_normalized = target_chunk_normalized[:, :min_len, :]
        
        # per-dimension MSE (B, T, A)
        per_sample_loss_all = F.mse_loss(pred_actions, target_chunk_normalized, reduction='none')  # (B, T, action_dim)
        per_sample_loss = reduce(per_sample_loss_all, 'b t a -> b', 'mean')  # (B,)

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
    
    def get_optimizer(
            self, 
            lr: float,
            weight_decay: float,
            obs_encoder_lr: float = None,
            obs_encoder_weight_decay: float = None,
            betas: Tuple[float, float] = (0.9, 0.95),
            **kwargs
        ) -> torch.optim.Optimizer:
        if obs_encoder_lr is None:
            obs_encoder_lr = lr
        if obs_encoder_weight_decay is None:
            obs_encoder_weight_decay = weight_decay
        
        optim_groups = self.diffusion_model.get_optim_groups(
            weight_decay=weight_decay)

        if self.rnn_obs_encoder is not None:
            optim_groups += self.rnn_obs_encoder.get_optim_groups(
                weight_decay=weight_decay
            )

        if self.confidence_head is not None:
            if hasattr(self.confidence_head, 'get_optim_groups'):
                optim_groups += self.confidence_head.get_optim_groups(weight_decay=weight_decay)
            else:
                optim_groups += [{
                    "params": self.confidence_head.parameters(),
                    "weight_decay": weight_decay,
                }]

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
        
        if hasattr(self.act_model, 'get_optim_groups'):
            act_groups = self.act_model.get_optim_groups(weight_decay=weight_decay)
            optim_groups.extend(act_groups)
        else:
            optim_groups.append({
                "params": list(self.act_model.parameters()),
                "weight_decay": weight_decay,
            })

        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas
        )
        return optimizer

