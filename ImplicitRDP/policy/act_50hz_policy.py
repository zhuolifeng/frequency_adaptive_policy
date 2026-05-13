from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import reduce
from loguru import logger
from torch.cuda.amp import autocast

from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.model.act.act_50hz_transformer import ACT50HzTransformer
from ImplicitRDP.model.common.normalizer import LinearNormalizer
from ImplicitRDP.model.force.rnn import RNN
from ImplicitRDP.policy.base_image_policy import BaseImagePolicy


class ACT50HzPolicy(BaseImagePolicy):
    """Single ACT policy trained on the full 50Hz action stack.

    The model predicts the complete 80-step action_stack chunk. During
    inference, action_start_50hz selects the executable suffix, matching the
    50Hz FM policy convention.
    """

    def __init__(
        self,
        shape_meta: dict,
        act_model: ACT50HzTransformer,
        obs_encoder: Optional[torch.nn.Module],
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        n_emb: int = 768,
        use_rnn_obs_encoder: bool = True,
        rnn_obs_encoder_n_layer: int = 1,
        rnn_obs_encoder_hidden_dim: int = 512,
        use_amp_for_inference: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        highfreq_extended_obs_key: str = "right_robot_tcp_wrench_stack",
        action_start_50hz: int = 13,
        action_normalizer_key: str = "action_stack",
        action_train_key: str = "action_stack",
        **kwargs,
    ):
        super().__init__()
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1

        self.action_dim = action_shape[0]
        self.action_horizon = horizon
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.n_emb = n_emb
        self.shape_meta = shape_meta
        self.obs_encoder = obs_encoder
        self.act_model = act_model
        self.normalizer = LinearNormalizer()
        self.kwargs = kwargs

        self.highfreq_extended_obs_key = highfreq_extended_obs_key
        self.action_start_50hz = action_start_50hz
        self.action_normalizer_key = action_normalizer_key
        self.action_train_key = action_train_key

        self.use_amp_for_inference = use_amp_for_inference
        if use_amp_for_inference:
            if amp_dtype == "bf16" and torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
            elif amp_dtype == "fp16":
                self.amp_dtype = torch.float16
            elif isinstance(amp_dtype, torch.dtype):
                self.amp_dtype = amp_dtype
            else:
                self.amp_dtype = torch.float16
        else:
            self.amp_dtype = None
        self.progress = 0.0

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

        logger.info(
            "ACT50HzPolicy initialized "
            f"(horizon={horizon}, action_start_50hz={action_start_50hz}, "
            f"train_key={action_train_key})"
        )

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def update_progress(self, progress: float):
        self.progress = progress

    def reset(self):
        pass

    def get_temporal_cond(
        self,
        extended_obs_dict: Optional[Dict[str, torch.Tensor]],
        extended_obs_last_step: Optional[int] = None,
        extend_obs_pad_after_n: Optional[int] = None,
        post_process: bool = False,
    ) -> Optional[torch.Tensor]:
        if extended_obs_dict is None or self.rnn_obs_encoder is None:
            return None

        key = self.highfreq_extended_obs_key
        if key not in extended_obs_dict:
            raise KeyError(f"Missing extended obs key: {key}")

        obs = extended_obs_dict[key]
        if extended_obs_last_step is not None:
            obs = obs[:, -extended_obs_last_step:, ...]
        if extend_obs_pad_after_n is not None:
            padding_obs = obs[:, -1:, ...].repeat(1, extend_obs_pad_after_n, 1)
            obs = torch.cat([padding_obs, obs], dim=-2)

        obs = self.normalizer[f"extended_{key}"].normalize(obs)
        if post_process:
            obs = self.rnn_obs_encoder(obs)
        return obs

    def _predict_obs_tokens(
        self,
        obs_dict: Dict[str, torch.Tensor],
        extended_obs_dict: Optional[Dict[str, torch.Tensor]] = None,
        return_expanded_obs_tokens: bool = False,
        return_temporal_cond: bool = False,
    ):
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))

        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])

        temporal_cond = None
        if self.rnn_obs_encoder is not None and extended_obs_dict is not None:
            temporal_cond = self.get_temporal_cond(extended_obs_dict, post_process=True)

        if return_temporal_cond:
            return obs_tokens, temporal_cond

        obs_flat = obs_tokens.reshape(B, -1)
        if return_expanded_obs_tokens:
            obs_flat = obs_flat.unsqueeze(1).expand(-1, self.action_horizon, -1)
        return obs_flat

    def predict_obs_tokens(self, *args, **kwargs) -> torch.Tensor:
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_obs_tokens(*args, **kwargs)

    def sample_noisy_trajectory(
        self,
        B: int = 1,
        generator=None,
        return_expanded_noisy_trajectory: bool = False,
    ) -> torch.Tensor:
        noisy_trajectory = torch.zeros((B, 0), dtype=self.dtype, device=self.device)
        if return_expanded_noisy_trajectory:
            noisy_trajectory = noisy_trajectory.unsqueeze(1).expand(-1, self.action_horizon, -1)
        return noisy_trajectory

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        extended_obs_dict: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            obs_tokens, temporal_cond = self._predict_obs_tokens(
                obs_dict,
                extended_obs_dict,
                return_temporal_cond=True,
            )
            return self._predict_action_act_from_tokens(
                obs_tokens,
                temporal_cond,
            )

    def _predict_action_act_from_tokens(
        self,
        obs_tokens: torch.Tensor,
        temporal_cond: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        action_chunk_normalized = self.act_model(
            obs_tokens=obs_tokens,
            action_history=None,
            temporal_cond=temporal_cond,
        )
        action_pred = self.normalizer[self.action_normalizer_key].unnormalize(action_chunk_normalized)

        action = action_pred[:, self.action_start_50hz:]
        return {
            "action": action,
            "action_pred": action_pred,
            "action_50hz": action,
            "action_pred_50hz": action_pred,
        }

    def _predict_from_obs_tokens_and_noisy_trajectory(
        self,
        obs_tokens_and_noisy_trajectory_list: List[torch.Tensor],
        extended_obs_dict: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
        extended_obs_last_step_list: List[int],
        dataset_obs_temporal_downsample_ratio: int,
        extend_obs_pad_after: bool = False,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        obs_tokens_and_noisy_trajectory = torch.cat(obs_tokens_and_noisy_trajectory_list, dim=0)
        B = obs_tokens_and_noisy_trajectory.shape[0]
        obs_tokens = obs_tokens_and_noisy_trajectory.to(self.device).reshape(B, -1, self.n_emb)

        if extended_obs_dict is None:
            temporal_cond = None
            max_extended_obs_last_step = self.action_horizon
            extend_obs_pad_after_n = None
        else:
            max_extended_obs_last_step = max(extended_obs_last_step_list)
            extend_obs_pad_after_n = (
                self.action_horizon - max_extended_obs_last_step if extend_obs_pad_after else None
            )
            if isinstance(extended_obs_dict, list):
                temporal_cond = [
                    self.get_temporal_cond(
                        item,
                        max_extended_obs_last_step,
                        extend_obs_pad_after_n=extend_obs_pad_after_n,
                        post_process=True,
                    )
                    for item in extended_obs_dict
                ]
            else:
                temporal_cond = self.get_temporal_cond(
                    extended_obs_dict,
                    max_extended_obs_last_step,
                    extend_obs_pad_after_n=extend_obs_pad_after_n,
                    post_process=True,
                )

            temporal_cond_list = []
            for i in range(B):
                cond = temporal_cond[i] if isinstance(temporal_cond, list) else temporal_cond
                if extended_obs_last_step_list[i] < max_extended_obs_last_step:
                    padding_obs = cond[:, -1:, :].repeat(
                        1,
                        max_extended_obs_last_step - extended_obs_last_step_list[i],
                        1,
                    )
                    cond = torch.cat(
                        [cond[:, -extended_obs_last_step_list[i]:, :], padding_obs],
                        dim=-2,
                    )
                temporal_cond_list.append(cond)
            temporal_cond = torch.cat(temporal_cond_list, dim=0)

        results = []
        start = self.action_start_50hz
        for i in range(B):
            temporal_cond_single = temporal_cond[i:i + 1] if temporal_cond is not None else None
            action_dict = self._predict_action_act_from_tokens(
                obs_tokens=obs_tokens[i:i + 1],
                temporal_cond=temporal_cond_single,
            )
            action_pred = action_dict["action_pred"]
            end = min(action_pred.shape[1], self.action_horizon)
            action = action_pred[:, start:end]

            if extend_obs_pad_after:
                results.append({"action": action, "action_pred": action_pred})
            else:
                valid_len = max(0, min(extended_obs_last_step_list[i] - start, end - start))
                results.append({
                    "action": action[:, :valid_len],
                    "action_pred": action_pred[:, :extended_obs_last_step_list[i]],
                })
        return results

    def predict_from_obs_tokens_and_noisy_trajectory(self, *args, **kwargs):
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_from_obs_tokens_and_noisy_trajectory(*args, **kwargs)

    def _compute_act_loss(
        self,
        action: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond: Optional[torch.Tensor],
        loss_mask=None,
    ):
        action_chunk_size = self.act_model.action_chunk_size
        if action.shape[1] != action_chunk_size:
            logger.warning(
                f"50Hz ACT target length {action.shape[1]} != chunk size {action_chunk_size}; "
                "using the shared prefix."
            )
        target_chunk = action[:, :action_chunk_size, :]
        target_chunk_normalized = self.normalizer[self.action_normalizer_key].normalize(target_chunk)

        pred_actions = self.act_model(
            obs_tokens=obs_tokens,
            action_history=None,
            temporal_cond=temporal_cond,
        )
        if pred_actions.shape[1] != target_chunk_normalized.shape[1]:
            min_len = min(pred_actions.shape[1], target_chunk_normalized.shape[1])
            pred_actions = pred_actions[:, :min_len, :]
            target_chunk_normalized = target_chunk_normalized[:, :min_len, :]

        per_sample_loss = F.mse_loss(pred_actions, target_chunk_normalized, reduction="none")
        per_sample_loss = reduce(per_sample_loss, "b t a -> b", "mean")

        if loss_mask is None:
            loss_mask = torch.ones(action.shape[0], dtype=torch.bool, device=action.device)
        loss = per_sample_loss[loss_mask].mean()
        return loss, per_sample_loss

    def compute_loss(self, batch: Dict[str, torch.Tensor], loss_mask=None):
        action = batch[self.action_train_key]
        obs_dict = batch["obs"]
        extended_obs_dict = batch.get("extended_obs", None)

        obs_tokens, _ = self._predict_obs_tokens(
            obs_dict,
            extended_obs_dict,
            return_temporal_cond=True,
        )

        if extended_obs_dict is not None and self.rnn_obs_encoder is not None:
            first_val = extended_obs_dict[self.highfreq_extended_obs_key]
            T_extended = first_val.shape[1]
            T_action = action.shape[1]
            extend_pad_n = T_action - T_extended if T_action > T_extended else None
            temporal_cond = self.get_temporal_cond(
                extended_obs_dict,
                extended_obs_last_step=T_action,
                extend_obs_pad_after_n=extend_pad_n,
                post_process=True,
            )
        else:
            temporal_cond = None

        act_loss, act_loss_vec = self._compute_act_loss(
            action,
            obs_tokens,
            temporal_cond,
            loss_mask,
        )
        return act_loss, {
            "act_loss": act_loss.item(),
            "total_loss": act_loss.item(),
            "act_loss_std": act_loss_vec.std().item(),
        }

    def get_optimizer(
        self,
        lr: float,
        weight_decay: float,
        obs_encoder_lr: Optional[float] = None,
        obs_encoder_weight_decay: Optional[float] = None,
        betas: Tuple[float, float] = (0.9, 0.95),
        **kwargs,
    ) -> torch.optim.Optimizer:
        if obs_encoder_lr is None:
            obs_encoder_lr = lr
        if obs_encoder_weight_decay is None:
            obs_encoder_weight_decay = weight_decay

        optim_groups = []
        if self.rnn_obs_encoder is not None:
            optim_groups += self.rnn_obs_encoder.get_optim_groups(weight_decay=weight_decay)

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
        optim_groups.extend(self.act_model.get_optim_groups(weight_decay=weight_decay))

        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
