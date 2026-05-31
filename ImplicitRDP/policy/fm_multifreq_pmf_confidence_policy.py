"""
Multi-frequency PMF policy with a confidence head.

Stage 1 trains a pMF-style multi-frequency base policy:
  - shared backbone with u/v heads
  - heads predict clean action samples
  - velocities are derived as (x_t - x0_pred) / clamp(t)
  - MeanFlow/JVP objective trains both 10hz and 50hz heads

Stage 2 freezes the base policy and trains only the confidence head on a
separate dataset. The confidence target is the 10hz PMF sample MSE. Confidence
selection is used only for inference, not while computing the Stage 2 target.
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import reduce
from loguru import logger
from torch.cuda.amp import autocast
from torch.func import jvp

from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.model.confidence.transformer import TransformerConfidenceHead
from ImplicitRDP.model.diffusion.transformer_for_pmf_multifreq import (
    TransformerForPMFMultiFreq,
)
from ImplicitRDP.policy.fm_multifreq_policy import FMMultiFreqPolicy


class FMMultiFreqPMFConfidencePolicy(FMMultiFreqPolicy):
    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: torch.nn.Module,
        model: TransformerForPMFMultiFreq,
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
        inference_head: str = "50hz",
        data_proportion: float = 0.5,
        norm_p: float = 1.0,
        norm_eps: float = 0.01,
        sample_pred_weight: float = 0.1,
        confidence_hidden_dim: Optional[int] = None,
        confidence_gamma: float = 1.0,
        confidence_alpha: float = 0.2,
        confidence_log_sigma_min: float = -5.0,
        confidence_log_sigma_max: float = 5.0,
        confidence_clamp_log_sigma: bool = False,
        confidence_obs_encoder_n_layers: int = 2,
        confidence_obs_encoder_hidden_dim: Optional[int] = None,
        use_confidence_head: bool = True,
        confidence_transformer_n_heads: int = 8,
        confidence_transformer_dropout: float = 0.1,
        confidence_action_dims: int = 6,
        normalize_confidence_loss: bool = False,
        confidence_n_samples: int = 1,
        confidence_num_inference_steps: int = 1,
        confidence_threshold: float = 0.0,
        use_confidence_based_head_selection: bool = True,
        confidence_threshold_for_head_selection: float = 0.0,
        training_stage: int = 1,
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
            inference_head=inference_head,
            training_stage=training_stage,
            **kwargs,
        )

        self.data_proportion = data_proportion
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.sample_pred_weight = sample_pred_weight

        self.confidence_gamma = confidence_gamma
        self.confidence_alpha = confidence_alpha
        self.use_confidence_head = use_confidence_head
        self.confidence_action_dims = confidence_action_dims
        self.normalize_confidence_loss = normalize_confidence_loss
        self.confidence_n_samples = confidence_n_samples
        self.confidence_num_inference_steps = confidence_num_inference_steps
        self.confidence_threshold = confidence_threshold
        self.use_confidence_based_head_selection = use_confidence_based_head_selection
        self.confidence_threshold_for_head_selection = confidence_threshold_for_head_selection

        self.confidence_head = None
        if self.use_confidence_head:
            hidden_dim = confidence_hidden_dim or n_emb
            self.confidence_head = TransformerConfidenceHead(
                n_emb=n_emb,
                hidden_dim=hidden_dim,
                obs_encoder_n_layers=confidence_obs_encoder_n_layers,
                obs_encoder_hidden_dim=confidence_obs_encoder_hidden_dim,
                n_heads=confidence_transformer_n_heads,
                dropout=confidence_transformer_dropout,
                log_sigma_min=confidence_log_sigma_min,
                log_sigma_max=confidence_log_sigma_max,
                clamp_log_sigma=confidence_clamp_log_sigma,
            )

        self.set_training_stage(training_stage)
        self.apply_stage_freeze()

        logger.info(
            "FMMultiFreqPMFConfidencePolicy initialized "
            f"(stage={training_stage}, data_proportion={data_proportion}, "
            f"sample_pred_weight={sample_pred_weight}, "
            f"confidence_head={self.confidence_head is not None}, "
            f"inference_head={inference_head})"
        )

    def set_training_stage(self, stage: int):
        assert stage in (1, 2), f"Invalid stage {stage}, must be 1 or 2"
        self.training_stage = stage

    def apply_stage_freeze(self):
        if self.training_stage == 1:
            self.requires_grad_(True)
            if self.confidence_head is not None:
                self.confidence_head.requires_grad_(False)
                self.confidence_head.eval()
        elif self.training_stage == 2:
            self.requires_grad_(False)
            if self.confidence_head is not None:
                self.confidence_head.requires_grad_(True)
                self.confidence_head.train()

    def _sample_tr(self, B: int, device: torch.device):
        t = torch.distributions.Beta(self.beta_a, self.beta_b).sample((B,)).to(device)
        r = torch.distributions.Beta(self.beta_a, self.beta_b).sample((B,)).to(device)
        t = t * 0.999 + 0.001
        r = r * 0.999 + 0.001
        t, r = torch.max(t, r), torch.min(t, r)

        data_size = int(B * self.data_proportion)
        fm_mask = torch.arange(B, device=device) < data_size
        r = torch.where(fm_mask, t, r)
        return t, r

    @staticmethod
    def _adp_wt_fn(loss, norm_eps: float = 0.01, norm_p: float = 1.0):
        adp_wt = (loss + norm_eps) ** norm_p
        return loss / adp_wt.detach()

    def _compute_pmf_jvp(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond: Optional[torch.Tensor],
        freq_mode: str,
        v_tangent: torch.Tensor,
    ):
        def u_fn(z, t_val, r_val):
            h_val = t_val - r_val
            u_out, v_out, x0_u_out, x0_v_out = self.model(
                z,
                t_val,
                h_val,
                cond=obs_tokens,
                temporal_cond=temporal_cond,
                freq_mode=freq_mode,
            )
            return u_out, (v_out, x0_u_out, x0_v_out)

        u, du_combined, aux = jvp(
            u_fn,
            (z_t, t, r),
            (v_tangent, torch.ones_like(t), torch.zeros_like(r)),
            has_aux=True,
        )
        v, x0_u, x0_v = aux
        return u, v, x0_u, x0_v, du_combined

    def _compute_pmf_loss_one_freq(
        self,
        x0: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond: Optional[torch.Tensor],
        freq_mode: str,
        loss_mask=None,
    ):
        B = x0.shape[0]
        device = x0.device
        t, r = self._sample_tr(B, device)

        noise = torch.randn_like(x0)
        t_exp = t.view(B, 1, 1)
        z_t = t_exp * noise + (1.0 - t_exp) * x0
        denom = torch.clamp(t, min=self.model.sample_prediction_t_clip).view(B, 1, 1)
        v_target = ((z_t - x0) / denom).detach()

        u_pred, v_pred, x0_u, x0_v, du_combined = self._compute_pmf_jvp(
            z_t,
            t,
            r,
            obs_tokens,
            temporal_cond,
            freq_mode,
            v_tangent=v_target,
        )

        h_exp = (t - r).view(B, 1, 1)
        V = u_pred + h_exp * du_combined.detach()

        sq_u = (V - v_target) ** 2
        sq_v = (v_pred - v_target) ** 2
        sq_x0_u = (x0_u - x0) ** 2
        sq_x0_v = (x0_v - x0) ** 2

        raw_loss_u = sq_u.mean(dim=(1, 2))
        raw_loss_v = sq_v.mean(dim=(1, 2))
        raw_loss_x0_u = sq_x0_u.mean(dim=(1, 2))
        raw_loss_x0_v = sq_x0_v.mean(dim=(1, 2))

        loss_u = self._adp_wt_fn(sq_u.sum(dim=(1, 2)), self.norm_eps, self.norm_p)
        loss_v = self._adp_wt_fn(sq_v.sum(dim=(1, 2)), self.norm_eps, self.norm_p)
        loss_x0 = sq_x0_u.mean(dim=(1, 2)) + sq_x0_v.mean(dim=(1, 2))
        per_sample_loss = loss_u + loss_v + self.sample_pred_weight * loss_x0

        if loss_mask is not None:
            per_sample_loss = per_sample_loss[loss_mask]

        loss = per_sample_loss.mean()
        loss_dict = {
            f"pmf_loss_u_{freq_mode}": loss_u.mean().item(),
            f"pmf_loss_v_{freq_mode}": loss_v.mean().item(),
            f"pmf_loss_x0_{freq_mode}": loss_x0.mean().item(),
            f"raw_pmf_loss_u_{freq_mode}": raw_loss_u.mean().item(),
            f"raw_pmf_loss_v_{freq_mode}": raw_loss_v.mean().item(),
            f"raw_pmf_loss_x0_u_{freq_mode}": raw_loss_x0_u.mean().item(),
            f"raw_pmf_loss_x0_v_{freq_mode}": raw_loss_x0_v.mean().item(),
            f"raw_pmf_loss_total_{freq_mode}": (
                raw_loss_u.mean()
                + raw_loss_v.mean()
                + self.sample_pred_weight * (raw_loss_x0_u.mean() + raw_loss_x0_v.mean())
            ).item(),
        }
        return loss, loss_dict

    def _encode_normalized_obs(self, nobs: Dict[str, torch.Tensor], B: int):
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])
        return obs_tokens

    def _compute_confidence_logits(self, obs_tokens: torch.Tensor):
        if self.confidence_head is None:
            return None
        return self.confidence_head(obs_tokens)

    def _compute_confidence_loss(self, obs_tokens, pmf_mse_vec_conf, loss_mask=None):
        if self.confidence_head is None:
            return None, None, None, None

        log_sigma_raw = self._compute_confidence_logits(obs_tokens)
        log_sigma = log_sigma_raw[loss_mask] if loss_mask is not None else log_sigma_raw
        pmf_loss = pmf_mse_vec_conf[loss_mask] if loss_mask is not None else pmf_mse_vec_conf

        sigma = 1.0 + torch.exp(log_sigma)
        log_sigma_term = torch.log(sigma)
        conf_loss = (
            self.confidence_gamma * pmf_loss * sigma
            - self.confidence_alpha * log_sigma_term
        ).mean()
        return conf_loss, log_sigma, sigma, pmf_loss

    def _compute_pmf_10hz_sample_mse(
        self,
        nactions_10: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond_10hz: Optional[torch.Tensor],
        loss_mask=None,
        n_samples: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
    ):
        B = nactions_10.shape[0]
        n_samples = self.confidence_n_samples if n_samples is None else n_samples
        steps = (
            self.confidence_num_inference_steps
            if num_inference_steps is None
            else num_inference_steps
        )

        per_sample_loss_all_accum = torch.zeros_like(nactions_10)
        with torch.no_grad():
            for _ in range(n_samples):
                x0_pred = self.pmf_sample(
                    cond=obs_tokens,
                    temporal_cond=temporal_cond_10hz,
                    freq_mode="10hz",
                    num_inference_steps=steps,
                )
                per_sample_loss_all_accum += F.mse_loss(
                    x0_pred, nactions_10, reduction="none"
                )

        per_sample_loss_all = per_sample_loss_all_accum / max(1, n_samples)
        per_sample_loss = reduce(per_sample_loss_all, "b t a -> b", "mean")

        conf_dims = min(self.confidence_action_dims, per_sample_loss_all.shape[-1])
        sel_loss = per_sample_loss_all[..., :conf_dims]
        if self.normalize_confidence_loss:
            gt_sq = nactions_10[..., :conf_dims] ** 2
            sel_loss = sel_loss / (gt_sq + 1e-8)
        per_sample_loss_conf = reduce(sel_loss, "b t a -> b", "mean")

        if loss_mask is None:
            loss_mask = torch.ones(B, dtype=torch.bool, device=nactions_10.device)
        loss = per_sample_loss[loss_mask].mean()
        return loss, per_sample_loss, per_sample_loss_conf

    def compute_loss(self, batch, loss_mask=None):
        nobs = self.normalizer.normalize(batch["obs"])
        nactions_10 = self.normalizer["action"].normalize(batch["action"])
        B = nactions_10.shape[0]

        if self.rnn_lowfreq is not None or self.rnn_highfreq is not None:
            assert "extended_obs" in batch, "extended_obs is required for RNNObsEncoder"

        if self.training_stage == 1:
            nactions_50 = self.normalizer["action_stack"].normalize(batch["action_stack"])
            temporal_cond_10hz = (
                self.get_temporal_cond(batch["extended_obs"], head="10hz")
                if self.rnn_lowfreq is not None
                else None
            )
            temporal_cond_50hz = (
                self.get_temporal_cond(batch["extended_obs"], head="50hz")
                if self.rnn_highfreq is not None
                else None
            )
            obs_tokens = self._encode_normalized_obs(nobs, B)

            loss_10, dict_10 = self._compute_pmf_loss_one_freq(
                nactions_10, obs_tokens, temporal_cond_10hz, "10hz", loss_mask
            )
            loss_50, dict_50 = self._compute_pmf_loss_one_freq(
                nactions_50, obs_tokens, temporal_cond_50hz, "50hz", loss_mask
            )
            total_loss = loss_10 + loss_50
            loss_dict = {
                "total_loss": total_loss.item(),
                "training_stage": 1.0,
                **dict_10,
                **dict_50,
            }
            loss_dict["raw_total_loss"] = (
                loss_dict["raw_pmf_loss_total_10hz"]
                + loss_dict["raw_pmf_loss_total_50hz"]
            )
            return total_loss, loss_dict

        if self.training_stage == 2:
            with torch.no_grad():
                temporal_cond_10hz = (
                    self.get_temporal_cond(batch["extended_obs"], head="10hz")
                    if self.rnn_lowfreq is not None
                    else None
                )
                obs_tokens = self._encode_normalized_obs(nobs, B)
                pmf_mse, pmf_mse_vec, pmf_mse_vec_conf = (
                    self._compute_pmf_10hz_sample_mse(
                        nactions_10,
                        obs_tokens,
                        temporal_cond_10hz,
                        loss_mask=loss_mask,
                    )
                )
            obs_tokens = obs_tokens.detach()
            conf_loss, log_sigma, sigma_stats, pmf_loss_normalized = (
                self._compute_confidence_loss(obs_tokens, pmf_mse_vec_conf, loss_mask)
            )
            total_loss = conf_loss if conf_loss is not None else pmf_mse
            loss_dict = {
                "pmf_10hz_sample_mse": pmf_mse.item(),
                "total_loss": total_loss.item(),
                "training_stage": 2.0,
            }
            if log_sigma is not None:
                loss_dict.update(
                    {
                        "confidence/log_sigma": log_sigma.mean().item(),
                        "confidence/log_sigma_min": log_sigma.min().item(),
                        "confidence/log_sigma_max": log_sigma.max().item(),
                        "confidence/sigma": sigma_stats.mean().item(),
                        "confidence/pmf_origin_loss": pmf_loss_normalized.mean().item(),
                        "confidence/conf_loss": conf_loss.item(),
                    }
                )
            return total_loss, loss_dict

        raise ValueError(f"Invalid training_stage: {self.training_stage}")

    def forward(self, batch, loss_mask=None):
        return self.compute_loss(batch, loss_mask)

    def pmf_sample(
        self,
        cond=None,
        temporal_cond=None,
        freq_mode: str = "10hz",
        generator=None,
        num_inference_steps: Optional[int] = None,
        noisy_trajectory=None,
        **kwargs,
    ):
        steps = num_inference_steps if num_inference_steps is not None else self.num_inference_steps
        if noisy_trajectory is None:
            x = torch.randn(
                size=(cond.shape[0], temporal_cond.shape[1], self.action_dim),
                dtype=cond.dtype,
                device=cond.device,
                generator=generator,
            )
        else:
            x = noisy_trajectory

        B = x.shape[0]
        t_schedule = torch.linspace(1.0, 0.0, steps + 1, device=x.device)
        for i in range(steps):
            t = t_schedule[i]
            h = t - t_schedule[i + 1]
            t_tensor = torch.full((B,), t.item(), dtype=x.dtype, device=x.device)
            h_tensor = torch.full((B,), h.item(), dtype=x.dtype, device=x.device)
            u, _, _, _ = self.model(
                x,
                t_tensor,
                h_tensor,
                cond=cond,
                temporal_cond=temporal_cond,
                freq_mode=freq_mode,
            )
            x = x - h * u
        return x

    # Compatibility alias for existing reactive runners.
    imf_sample = pmf_sample

    def _select_head_from_confidence(self, obs_tokens: torch.Tensor):
        confidence_value = None
        selected_freq = self.inference_head
        if self.use_confidence_based_head_selection and self.confidence_head is not None:
            log_sigma = self._compute_confidence_logits(obs_tokens)
            if log_sigma is not None:
                sigma = 1.0 + torch.exp(log_sigma)
                confidence_value = torch.log(sigma).mean().item()
                selected_freq = (
                    "50hz"
                    if confidence_value > self.confidence_threshold_for_head_selection
                    else "10hz"
                )
        return selected_freq, confidence_value

    def _get_inference_head(self, obs_tokens=None):
        if obs_tokens is None:
            return self.inference_head
        if not (self.use_confidence_based_head_selection and self.confidence_head is not None):
            return self.inference_head
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(obs_tokens.shape[0], -1, self.n_emb)
        selected_freq, _ = self._select_head_from_confidence(obs_tokens)
        return selected_freq

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
            assert extended_obs_dict is not None, "extended_obs is required for RNN temporal condition"

        temporal_cond_10hz = self.get_temporal_cond(extended_obs_dict, head="10hz")
        temporal_cond_50hz = self.get_temporal_cond(extended_obs_dict, head="50hz")
        steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps

        nsample_10 = self.pmf_sample(
            cond=obs_tokens,
            temporal_cond=temporal_cond_10hz,
            freq_mode="10hz",
            num_inference_steps=steps,
        )
        nsample_50 = self.pmf_sample(
            cond=obs_tokens,
            temporal_cond=temporal_cond_50hz,
            freq_mode="50hz",
            num_inference_steps=steps,
        )

        action_pred_10hz = self.normalizer["action"].unnormalize(nsample_10)
        action_pred_50hz = self.normalizer["action_stack"].unnormalize(nsample_50)
        action_10hz = action_pred_10hz[
            :, self.action_start_10hz : self.action_start_10hz + self.n_action_steps
        ]
        action_50hz = action_pred_50hz[:, self.action_start_50hz : self.horizon_50hz]

        selected_freq, confidence_value = self._select_head_from_confidence(obs_tokens)
        if selected_freq == "50hz":
            action = action_50hz
            action_pred = action_pred_50hz
        else:
            action = action_10hz
            action_pred = action_pred_10hz

        result = {
            "action": action,
            "action_pred": action_pred,
            "action_10hz": action_10hz,
            "action_50hz": action_50hz,
            "action_pred_10hz": action_pred_10hz,
            "action_pred_50hz": action_pred_50hz,
            "selected_freq": selected_freq,
        }
        if confidence_value is not None:
            result["confidence"] = confidence_value
        return result

    def predict_action(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_action(*args, **kwargs)

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

        obs_tokens = obs_tokens_and_noisy_trajectory[:, : -target_horizon * self.action_dim]
        noisy_trajectory = obs_tokens_and_noisy_trajectory[:, -target_horizon * self.action_dim :]
        obs_tokens = obs_tokens.to(self.device).reshape(B, -1, self.n_emb)
        noisy_trajectory = noisy_trajectory.to(self.device).reshape(
            B, target_horizon, self.action_dim
        )

        max_extended_obs_last_step = max(extended_obs_last_step_list)
        assert extended_obs_dict is not None, "extended_obs is required for RNNObsEncoder"
        extend_obs_pad_after_n = (
            target_horizon - max_extended_obs_last_step if extend_obs_pad_after else None
        )

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
                    1, max_extended_obs_last_step - extended_obs_last_step_list[i], 1
                )
                temporal_cond_list.append(
                    torch.cat(
                        [source[:, -extended_obs_last_step_list[i] :, :], padding_obs],
                        dim=-2,
                    )
                )
            else:
                temporal_cond_list.append(source)
        temporal_cond = torch.cat(temporal_cond_list, dim=0)
        noisy_trajectory = noisy_trajectory[:, : temporal_cond.shape[1], :]

        nsample = self.pmf_sample(
            cond=obs_tokens,
            temporal_cond=temporal_cond,
            freq_mode=head,
            noisy_trajectory=noisy_trajectory,
        )
        normalizer_key = "action_stack" if head == "50hz" else "action"
        action_pred = self.normalizer[normalizer_key].unnormalize(nsample)
        if head == "50hz":
            start, end = self.action_start_50hz, self.horizon_50hz
        else:
            start, end = self.action_start_10hz, self.action_start_10hz + self.n_action_steps
        action = action_pred[:, start:end]

        results = []
        for i in range(B):
            if extend_obs_pad_after:
                results.append({"action": action[i : i + 1], "action_pred": action_pred[i : i + 1]})
            else:
                valid_len = max(0, min(extended_obs_last_step_list[i] - start, end - start))
                results.append(
                    {
                        "action": action[i : i + 1, :valid_len, :],
                        "action_pred": action_pred[i : i + 1, : extended_obs_last_step_list[i], :],
                    }
                )
        return results

    def get_optimizer(
        self,
        lr: float,
        weight_decay: float,
        obs_encoder_lr: Optional[float] = None,
        obs_encoder_weight_decay: Optional[float] = None,
        betas: Tuple[float, float] = (0.9, 0.95),
        **kwargs,
    ) -> torch.optim.Optimizer:
        if self.training_stage == 1:
            return super().get_optimizer(
                lr=lr,
                weight_decay=weight_decay,
                obs_encoder_lr=obs_encoder_lr,
                obs_encoder_weight_decay=obs_encoder_weight_decay,
                betas=betas,
            )

        if self.training_stage == 2:
            optim_groups = []
            if self.confidence_head is not None:
                if hasattr(self.confidence_head, "get_optim_groups"):
                    optim_groups += self.confidence_head.get_optim_groups(
                        weight_decay=weight_decay
                    )
                else:
                    optim_groups.append(
                        {
                            "params": list(self.confidence_head.parameters()),
                            "weight_decay": weight_decay,
                        }
                    )
            return torch.optim.AdamW(optim_groups, lr=lr, betas=betas)

        raise ValueError(f"Invalid training_stage: {self.training_stage}")
