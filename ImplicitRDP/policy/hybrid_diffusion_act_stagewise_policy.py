from typing import Dict, List, Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import reduce

from ImplicitRDP.model.common.normalizer import LinearNormalizer
from ImplicitRDP.policy.hybrid_diffusion_act_policy import HybridDiffusionACTPolicy
from ImplicitRDP.model.confidence.transformer import TransformerConfidenceHead
from loguru import logger


class HybridDiffusionACTStagewisePolicy(HybridDiffusionACTPolicy):
    def __init__(self,
                 shape_meta: dict,
                 noise_scheduler,
                 diffusion_model,
                 act_model,
                 obs_encoder,
                 horizon,
                 n_action_steps,
                 n_obs_steps,
                 num_inference_steps=None,
                 input_pertub=0.1,
                 n_emb: int = 768,
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
                 # Three-stage training parameters
                 training_stage: int = 1,
                 freeze_diffusion_model: bool = True,
                 freeze_act_model: bool = False,
                 freeze_confidence_head: bool = True,
                 **kwargs):
        
        # Initialize parent class
        super().__init__(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            diffusion_model=diffusion_model,
            act_model=act_model,
            obs_encoder=obs_encoder,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            num_inference_steps=num_inference_steps,
            input_pertub=input_pertub,
            n_emb=n_emb,
            use_rnn_obs_encoder=use_rnn_obs_encoder,
            rnn_obs_encoder_n_layer=rnn_obs_encoder_n_layer,
            rnn_obs_encoder_hidden_dim=rnn_obs_encoder_hidden_dim,
            use_amp_for_inference=use_amp_for_inference,
            amp_dtype=amp_dtype,
            confidence_hidden_dim=confidence_hidden_dim,
            confidence_gamma=confidence_gamma,
            confidence_alpha=confidence_alpha,
            confidence_log_sigma_min=confidence_log_sigma_min,
            confidence_log_sigma_max=confidence_log_sigma_max,
            confidence_clamp_log_sigma=confidence_clamp_log_sigma,
            confidence_obs_encoder_n_layers=confidence_obs_encoder_n_layers,
            confidence_obs_encoder_hidden_dim=confidence_obs_encoder_hidden_dim,
            use_confidence_head=use_confidence_head,
            confidence_head_type=confidence_head_type,
            confidence_transformer_n_heads=confidence_transformer_n_heads,
            confidence_transformer_dropout=confidence_transformer_dropout,
            inference_head=inference_head,
            use_confidence_based_head_selection=use_confidence_based_head_selection,
            confidence_threshold_for_head_selection=confidence_threshold_for_head_selection,
            confidence_action_dims=confidence_action_dims,
            normalize_confidence_loss=normalize_confidence_loss,
            dp_warmup_epochs=0,  # Disable the old two-stage training
            **kwargs
        )
        
        # Three-stage training parameters
        self.training_stage = training_stage
        self.freeze_diffusion_model = freeze_diffusion_model
        self.freeze_act_model = freeze_act_model
        self.freeze_confidence_head = freeze_confidence_head
        
        logger.info(f"HybridDiffusionACTStagewisePolicy initialized (three-stage training)")
        logger.info(f"  Stage 1: Train ACT + encoder + rnn, freeze DP + Confidence")
        logger.info(f"  Stage 2: Train ACT + DP + encoder + rnn, freeze Confidence")
        logger.info(f"  Stage 3: Train Confidence only, freeze all others")

    def set_training_stage(self, stage: int):
        assert stage in [1, 2, 3], f"Invalid stage: {stage}. Must be 1, 2, or 3."
        
        self.training_stage = stage

    def forward(self, batch: Dict[str, torch.Tensor], loss_mask=None):
        return self.compute_loss(batch, loss_mask)

    def compute_loss(self, batch: Dict[str, torch.Tensor], loss_mask=None):
        action_act = batch['action_stack']
        action_dp = batch['action']
        T_action = action_dp.shape[1]
        if 'extended_obs' in batch and len(self.extented_obs_keys) > 0:
            for key, value in batch['extended_obs'].items():
                if key in self.extented_obs_keys:
                    if 'wrench_stack' in key:
                        continue
                    T_extended = value.shape[1]
                    if T_extended > T_action:
                        batch['extended_obs'][key] = value[:, -T_action:, ...]
                    elif T_extended < T_action:
                        last_step = value[:, -1:, ...]
                        padding = last_step.repeat(1, T_action - T_extended, *([1] * (len(value.shape) - 2)))
                        batch['extended_obs'][key] = torch.cat([value, padding], dim=1)
        # Prepare obs for loss computation
        obs_dict = batch['obs']
        extended_obs_dict = batch.get('extended_obs', None)
        
        loss_dict = dict()
        
        # ==================== Stage 1: ACT only ====================
        if self.training_stage == 1:
            obs_tokens, temporal_cond = self._predict_obs_tokens(
                obs_dict, extended_obs_dict, return_temporal_cond=True, use_act_rnn=True
            )
            
            act_loss, act_loss_vec, act_loss_vec_conf, act_loss_vec_virtual = self._compute_act_loss(
                action_act, obs_tokens, temporal_cond, loss_mask
            )
            
            total_loss = act_loss
            
            loss_dict = {
                'act_loss': act_loss.item(),
                'diffusion_loss': 0.0,
                'confidence_loss': 0.0,
                'total_loss': total_loss.item(),
                'training_stage': 1.0
            }
        
        # ==================== Stage 2: ACT + DP ====================
        elif self.training_stage == 2:
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
            print(f"for stage 2, we get the shape of target which are {target.shape}")
            B = trajectory.shape[0]
            device = trajectory.device
            if loss_mask is None:
                loss_mask = torch.ones(B, dtype=torch.bool, device=device)
            
            diffusion_loss_vec = F.mse_loss(pred, target, reduction='none')
            diffusion_loss_vec = reduce(diffusion_loss_vec, 'b ... -> b (...)', 'mean')
            diffusion_loss_masked = diffusion_loss_vec[loss_mask]
            diffusion_loss_mean = diffusion_loss_masked.mean(dim=1)
            diffusion_loss_mean = diffusion_loss_mean.sum() / (loss_mask.sum() + 1e-8)
            
            obs_tokens, temporal_cond = self._predict_obs_tokens(
                obs_dict, extended_obs_dict, return_temporal_cond=True, use_act_rnn=True
            )
            act_loss, act_loss_vec, act_loss_vec_conf, act_loss_vec_virtual = self._compute_act_loss(
                action_act, obs_tokens, temporal_cond, loss_mask
            )
            
            log_sigma = None
            sigma_stats = None
            if self.confidence_head is not None:
                log_sigma_raw = self._compute_confidence_logits(obs_tokens)
                if log_sigma_raw is not None:
                    log_sigma = log_sigma_raw[loss_mask] if loss_mask is not None else log_sigma_raw
                    sigma = 1.0 + torch.exp(log_sigma)
                    sigma_stats = sigma
            
            total_loss = diffusion_loss_mean + act_loss
            
            loss_dict = {
                'diffusion_loss': diffusion_loss_mean.item(),
                'act_loss': act_loss.item(),
                'confidence_loss': 0.0,
                'total_loss': total_loss.item(),
                'training_stage': 2.0,
                'dp_loss_weight': 1.0
            }
            
            if log_sigma is not None and sigma_stats is not None:
                loss_dict.update({
                    'confidence/log_sigma': log_sigma.mean().item(),
                    'confidence/sigma': sigma_stats.mean().item(),
                })
        
        # ==================== Stage 3: Confidence only ====================
        elif self.training_stage == 3:
            with torch.no_grad():
                obs_tokens, temporal_cond = self._predict_obs_tokens(
                    obs_dict, extended_obs_dict, return_temporal_cond=True, use_act_rnn=True
                )
                _, _, act_loss_vec_conf, _ = self._compute_act_loss(
                    action_act, obs_tokens, temporal_cond, loss_mask
                )
            
            log_sigma = None
            sigma_stats = None
            act_conf_loss = None
            
            if self.confidence_head is not None:
                log_sigma_raw = self._compute_confidence_logits(obs_tokens)
                
                if log_sigma_raw is not None:
                    log_sigma = log_sigma_raw[loss_mask] if loss_mask is not None else log_sigma_raw
                    act_loss_masked = act_loss_vec_conf[loss_mask] if loss_mask is not None else act_loss_vec_conf
                    
                    sigma = 1.0 + torch.exp(log_sigma)
                    sigma_stats = sigma
                    log_sigma_term = torch.log(sigma)
                    act_loss_normalized = act_loss_masked
                    
                    gamma = self.confidence_gamma
                    alpha = self.confidence_alpha
                    
                    # L_confidence = gamma * sigma * L_act_norm - alpha * log(sigma)
                    act_conf_loss = gamma * act_loss_normalized * sigma - alpha * log_sigma_term
                    act_conf_loss = act_conf_loss.mean()
            
            if act_conf_loss is not None:
                total_loss = act_conf_loss
                
                loss_dict = {
                    'diffusion_loss': 0.0,
                    'act_loss': 0.0,
                    'confidence_loss': act_conf_loss.item(),
                    'total_loss': total_loss.item(),
                    'training_stage': 3.0
                }
                
                if log_sigma is not None and sigma_stats is not None:
                    loss_dict.update({
                        'confidence/log_sigma': log_sigma.mean().item(),
                        'confidence/log_sigma_min': log_sigma.min().item(),
                        'confidence/log_sigma_max': log_sigma.max().item(),
                        'confidence/sigma': sigma_stats.mean().item(),
                    })
        
        else:
            raise ValueError(f"Invalid training_stage: {self.training_stage}. Must be 1, 2, or 3.")
        
        return total_loss, loss_dict

    def get_optimizer(self,
                      lr: float,
                      weight_decay: float,
                      obs_encoder_lr: float = None,
                      obs_encoder_weight_decay: float = None,
                      betas: Tuple[float, float] = (0.9, 0.95),
                      **kwargs) -> torch.optim.Optimizer:
        if obs_encoder_lr is None:
            obs_encoder_lr = lr
        if obs_encoder_weight_decay is None:
            obs_encoder_weight_decay = weight_decay
        
        optim_groups = []
        
        # ==================== Stage 1: Only ACT components ====================
        if self.training_stage == 1:
            # Diffusion model - frozen
            # Confidence head - frozen
            
            # ACT model - train
            if hasattr(self.act_model, 'get_optim_groups'):
                optim_groups.extend(self.act_model.get_optim_groups(weight_decay=weight_decay))
            else:
                optim_groups.append({
                    "params": list(self.act_model.parameters()),
                    "weight_decay": weight_decay,
                })
            
            # RNN encoder - train ACT RNN
            if self.rnn_obs_encoder_act is not None:
                optim_groups += self.rnn_obs_encoder_act.get_optim_groups(weight_decay=weight_decay)
            
            # Obs encoder - train
            if self.obs_encoder is not None:
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
            
            # Diffusion model - frozen
            # Confidence head - frozen
        
        # ==================== Stage 2: ACT + DP ====================
        elif self.training_stage == 2:
            # Diffusion model - train
            optim_groups.extend(self.diffusion_model.get_optim_groups(weight_decay=weight_decay))
            
            # RNN encoder - train (both ACT and DP RNN)
            if self.rnn_obs_encoder_dp is not None:
                optim_groups += self.rnn_obs_encoder_dp.get_optim_groups(weight_decay=weight_decay)
            if self.rnn_obs_encoder_act is not None:
                optim_groups += self.rnn_obs_encoder_act.get_optim_groups(weight_decay=weight_decay)
            
            # Obs encoder - train
            if self.obs_encoder is not None:
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
            
            # ACT model - train
            if hasattr(self.act_model, 'get_optim_groups'):
                optim_groups.extend(self.act_model.get_optim_groups(weight_decay=weight_decay))
            else:
                optim_groups.append({
                    "params": list(self.act_model.parameters()),
                    "weight_decay": weight_decay,
                })
            
            # Confidence head - frozen
        
        # ==================== Stage 3: Confidence only ====================
        elif self.training_stage == 3:
            # All models frozen except confidence head
            
            # Confidence head - train
            if self.confidence_head is not None:
                if hasattr(self.confidence_head, 'get_optim_groups'):
                    optim_groups += self.confidence_head.get_optim_groups(weight_decay=weight_decay)
                else:
                    optim_groups += [{
                        "params": self.confidence_head.parameters(),
                        "weight_decay": weight_decay,
                    }]
            
            # All other components are frozen - don't add to optimizer
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas
        )
        
        return optimizer
