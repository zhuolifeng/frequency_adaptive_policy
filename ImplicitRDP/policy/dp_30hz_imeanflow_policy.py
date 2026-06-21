from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import reduce
from loguru import logger
from torch.cuda.amp import autocast

from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.model.common.normalizer import LinearNormalizer
from ImplicitRDP.model.diffusion.transformer_for_dp_30hz import TransformerForDP30Hz
from ImplicitRDP.model.force.rnn import RNN
from ImplicitRDP.policy.base_image_policy import BaseImagePolicy


class DP30HziMeanFlowPolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: torch.nn.Module,
        model: TransformerForDP30Hz,
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
        action_start_30hz: Optional[int] = None,
        action_normalizer_key: str = "action_stack",
        action_train_key: str = "action_stack",
        num_inference_steps: int = 1,
        p_mean: float = -0.4,
        p_std: float = 1.0,
        data_proportion: float = 0.5,
        norm_p: float = 1.0,
        norm_eps: float = 0.01,
        loss_u_weight: float = 1.0,
        loss_v_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        action_shape = shape_meta.get(action_train_key, shape_meta["action"])["shape"]
        assert len(action_shape) == 1
        self.action_dim = action_shape[0]
        self.action_horizon = horizon
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.n_emb = n_emb
        self.shape_meta = shape_meta
        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        self.kwargs = kwargs

        self.highfreq_extended_obs_key = highfreq_extended_obs_key
        self.action_start_30hz = 7 if action_start_30hz is None else int(action_start_30hz)
        self.action_normalizer_key = action_normalizer_key
        self.action_train_key = action_train_key
        self.num_inference_steps = int(num_inference_steps)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.data_proportion = float(data_proportion)
        self.norm_p = float(norm_p)
        self.norm_eps = float(norm_eps)
        self.loss_u_weight = float(loss_u_weight)
        self.loss_v_weight = float(loss_v_weight)
        self.progress = 0.0

        if use_rnn_obs_encoder:
            assert "extended_obs" in shape_meta, "extended_obs is required for RNNObsEncoder"
            assert highfreq_extended_obs_key in shape_meta["extended_obs"], (
                f"{highfreq_extended_obs_key} is required for 30Hz temporal condition"
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

        logger.info(
            "DP30HziMeanFlowPolicy initialized "
            f"(horizon={horizon}, action_start_30hz={self.action_start_30hz}, "
            f"train_key={action_train_key}, num_inference_steps={self.num_inference_steps})"
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
        return obs_tokens

    def predict_obs_tokens(self, *args, **kwargs) -> torch.Tensor:
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_obs_tokens(*args, **kwargs)

    def _prepare_inputs(self, batch):
        actions = batch[self.action_train_key]
        nactions = self.normalizer[self.action_normalizer_key].normalize(actions)
        obs_tokens = self._predict_obs_tokens(batch["obs"])

        extended_obs_dict = batch.get("extended_obs", None)
        if extended_obs_dict is not None and self.rnn_obs_encoder is not None:
            first_val = extended_obs_dict[self.highfreq_extended_obs_key]
            t_extended = first_val.shape[1]
            t_action = actions.shape[1]
            extend_pad_n = t_action - t_extended if t_action > t_extended else None
            temporal_cond = self.get_temporal_cond(
                extended_obs_dict,
                extended_obs_last_step=t_action,
                extend_obs_pad_after_n=extend_pad_n,
                post_process=True,
            )
        else:
            temporal_cond = None
        return obs_tokens, nactions, temporal_cond

    def _sample_tr(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        t = torch.sigmoid(torch.randn(batch_size, device=device, dtype=dtype) * self.p_std + self.p_mean)
        r = torch.sigmoid(torch.randn(batch_size, device=device, dtype=dtype) * self.p_std + self.p_mean)
        t, r = torch.maximum(t, r), torch.minimum(t, r)

        data_size = int(batch_size * self.data_proportion)
        if data_size > 0:
            fm_mask = torch.arange(batch_size, device=device) < data_size
            r = torch.where(fm_mask, t, r)
        else:
            fm_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        return t, r, fm_mask

    def _model_uv(
        self,
        sample: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond: Optional[torch.Tensor],
    ):
        h = (t - r).clamp_min(0.0)
        out = self.model(
            sample,
            t,
            cond=obs_tokens,
            temporal_cond=temporal_cond,
            time_cond=h,
        )
        if out.shape[-1] != self.action_dim * 2:
            raise ValueError(
                f"iMeanFlow backbone must output {self.action_dim * 2} dims, got {out.shape[-1]}."
            )
        return out[..., :self.action_dim], out[..., self.action_dim:]

    def _adaptive_weighted_mean(self, loss_per_sample: torch.Tensor):
        weights = (loss_per_sample + self.norm_eps) ** self.norm_p
        return (loss_per_sample / weights.detach()).mean()

    def compute_loss(self, batch: Dict[str, torch.Tensor], loss_mask=None):
        obs_tokens, x, temporal_cond = self._prepare_inputs(batch)
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        t, r, _ = self._sample_tr(batch_size, device, dtype)
        noise = torch.randn_like(x)
        z_t = (1.0 - t.view(-1, 1, 1)) * x + t.view(-1, 1, 1) * noise
        v_target = noise - x

        def uv_fn(z, t_in, r_in):
            return self._model_uv(z, t_in, r_in, obs_tokens, temporal_cond)

        dtdt = torch.ones_like(t)
        dtdr = torch.zeros_like(r)
        (u, v), (du_dt, _) = torch.func.jvp(
            uv_fn,
            (z_t, t, r),
            (v_target, dtdt, dtdr),
        )
        V = u + (t - r).view(-1, 1, 1) * du_dt.detach()
        v_target = v_target.detach()

        loss_u_per = reduce((V - v_target) ** 2, "b ... -> b", "sum")
        loss_v_per = reduce((v - v_target) ** 2, "b ... -> b", "sum")
        if loss_mask is not None:
            loss_u_per = loss_u_per[loss_mask]
            loss_v_per = loss_v_per[loss_mask]

        loss_u = self._adaptive_weighted_mean(loss_u_per)
        loss_v = self._adaptive_weighted_mean(loss_v_per)
        total = self.loss_u_weight * loss_u + self.loss_v_weight * loss_v

        raw_loss_u = F.mse_loss(V, v_target)
        raw_loss_v = F.mse_loss(v, v_target)
        return total, {
            "imeanflow_loss_u": loss_u.item(),
            "imeanflow_loss_v": loss_v.item(),
            "imeanflow_raw_mse_u": raw_loss_u.item(),
            "imeanflow_raw_mse_v": raw_loss_v.item(),
            "imeanflow_t_mean": t.mean().item(),
            "imeanflow_r_mean": r.mean().item(),
            "total_loss": total.item(),
        }

    def conditional_sample(
        self,
        cond: torch.Tensor,
        temporal_cond: Optional[torch.Tensor] = None,
        generator=None,
        num_inference_steps: Optional[int] = None,
        noisy_trajectory: Optional[torch.Tensor] = None,
    ):
        B = cond.shape[0]
        steps = self.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        steps = max(1, steps)
        if noisy_trajectory is None:
            trajectory = torch.randn(
                (B, self.action_horizon, self.action_dim),
                dtype=cond.dtype,
                device=cond.device,
                generator=generator,
            )
        else:
            trajectory = noisy_trajectory

        t_steps = torch.linspace(1.0, 0.0, steps + 1, device=trajectory.device, dtype=trajectory.dtype)
        for i in range(steps):
            t = torch.full((B,), t_steps[i].item(), device=trajectory.device, dtype=trajectory.dtype)
            r = torch.full((B,), t_steps[i + 1].item(), device=trajectory.device, dtype=trajectory.dtype)
            u, _ = self._model_uv(trajectory, t, r, cond, temporal_cond)
            trajectory = trajectory - (t - r).view(-1, 1, 1) * u
        return trajectory

    def _predict_action_from_tokens(
        self,
        obs_tokens: torch.Tensor,
        temporal_cond: Optional[torch.Tensor],
        noisy_trajectory: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
    ):
        nsample = self.conditional_sample(
            cond=obs_tokens,
            temporal_cond=temporal_cond,
            noisy_trajectory=noisy_trajectory,
            num_inference_steps=num_inference_steps,
        )
        action_pred = self.normalizer[self.action_normalizer_key].unnormalize(nsample)
        action = action_pred[:, self.action_start_30hz:self.action_horizon]
        return action, action_pred

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        extended_obs_dict: Optional[Dict[str, torch.Tensor]] = None,
        num_inference_steps: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            obs_tokens, temporal_cond = self._predict_obs_tokens(
                obs_dict,
                extended_obs_dict,
                return_temporal_cond=True,
            )
            action, action_pred = self._predict_action_from_tokens(
                obs_tokens,
                temporal_cond,
                num_inference_steps=num_inference_steps,
            )
            return {
                "action": action,
                "action_pred": action_pred,
                "action_30hz": action,
                "action_pred_30hz": action_pred,
            }

    def sample_noisy_trajectory(
        self,
        B: int = 1,
        generator=None,
        return_expanded_noisy_trajectory: bool = False,
    ) -> torch.Tensor:
        noisy = torch.randn(
            (B, self.action_horizon, self.action_dim),
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )
        if return_expanded_noisy_trajectory:
            return noisy
        return noisy.reshape(B, -1)

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
        noisy_dim = self.action_horizon * self.action_dim
        obs_tokens = obs_tokens_and_noisy_trajectory[:, :-noisy_dim].to(self.device).reshape(B, -1, self.n_emb)
        noisy_trajectory = obs_tokens_and_noisy_trajectory[:, -noisy_dim:].to(self.device).reshape(
            B, self.action_horizon, self.action_dim
        )

        max_last_step = max(extended_obs_last_step_list)
        pad_after_n = self.action_horizon - max_last_step if extend_obs_pad_after else None
        if isinstance(extended_obs_dict, list):
            temporal_cond = [
                self.get_temporal_cond(
                    item,
                    extended_obs_last_step=max_last_step,
                    extend_obs_pad_after_n=pad_after_n,
                    post_process=True,
                )
                for item in extended_obs_dict
            ]
        else:
            temporal_cond = self.get_temporal_cond(
                extended_obs_dict,
                extended_obs_last_step=max_last_step,
                extend_obs_pad_after_n=pad_after_n,
                post_process=True,
            )

        temporal_cond_list = []
        for i in range(B):
            source = temporal_cond[i] if isinstance(temporal_cond, list) else temporal_cond
            if extended_obs_last_step_list[i] < max_last_step:
                padding_obs = source[:, -1:, :].repeat(
                    1,
                    max_last_step - extended_obs_last_step_list[i],
                    1,
                )
                source = torch.cat(
                    [source[:, -extended_obs_last_step_list[i]:, :], padding_obs],
                    dim=-2,
                )
            temporal_cond_list.append(source)
        temporal_cond = torch.cat(temporal_cond_list, dim=0)
        noisy_trajectory = noisy_trajectory[:, :temporal_cond.shape[1], :]

        action, action_pred = self._predict_action_from_tokens(
            obs_tokens,
            temporal_cond,
            noisy_trajectory=noisy_trajectory,
        )

        results = []
        start = self.action_start_30hz
        end = self.action_horizon
        for i in range(B):
            if extend_obs_pad_after:
                results.append({"action": action[i:i + 1], "action_pred": action_pred[i:i + 1]})
            else:
                valid_len = max(0, min(extended_obs_last_step_list[i] - start, end - start))
                results.append({
                    "action": action[i:i + 1, :valid_len, :],
                    "action_pred": action_pred[i:i + 1, :extended_obs_last_step_list[i], :],
                })
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
        optim_groups.extend(self.model.get_optim_groups(weight_decay=weight_decay))
        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
