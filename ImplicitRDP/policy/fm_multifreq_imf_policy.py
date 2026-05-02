"""
Multi-frequency Flow Matching Policy with improved MeanFlow (iMF).

Single-stage training that directly learns average velocity u and auxiliary
instantaneous velocity v. Enables 1-step inference without any distillation.

Key idea (MeanFlow equation):
    V = u + (t - r) * du/dt = v_target

Training mixes two sample types each batch:
    - data_proportion fraction: r = t (standard FM velocity matching)
    - remaining fraction:       r < t (MeanFlow, learns large-step jumps)

Reference: "Improved Mean Flows" (arXiv:2512.02012)
"""
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.func import jvp

from loguru import logger

from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.policy.fm_multifreq_policy import FMMultiFreqPolicy
from ImplicitRDP.model.diffusion.transformer_for_imf_multifreq import (
    TransformerForIMFMultiFreq,
)


class FMMultiFreqIMFPolicy(FMMultiFreqPolicy):
    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: torch.nn.Module,
        model: TransformerForIMFMultiFreq,
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
        # iMF specific
        data_proportion: float = 0.5,
        norm_p: float = 1.0,
        norm_eps: float = 0.01,
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
            inference_head=inference_head,
            training_stage=1,  # always stage 1, single-stage
            **kwargs,
        )

        self.data_proportion = data_proportion
        self.norm_p = norm_p
        self.norm_eps = norm_eps

        logger.info(
            "FMMultiFreqIMFPolicy initialized "
            f"(data_proportion={data_proportion}, "
            f"norm_p={norm_p}, norm_eps={norm_eps}, "
            f"inference_head={inference_head}, "
            f"num_inference_steps={num_inference_steps})"
        )

    # ------------------------------------------------------------------
    # iMF time sampling
    # ------------------------------------------------------------------

    def _sample_tr(self, B: int, device: torch.device):
        """Sample t and r for iMF training.

        Returns:
            t: (B,) time values in [0.001, 1.0], t >= r
            r: (B,) time values in [0.001, 1.0], r <= t
               For first data_proportion fraction of batch, r = t (FM mode)
        """
        t = torch.distributions.Beta(self.beta_a, self.beta_b).sample(
            (B,)
        ).to(device)
        t = t * 0.999 + 0.001

        r = torch.distributions.Beta(self.beta_a, self.beta_b).sample(
            (B,)
        ).to(device)
        r = r * 0.999 + 0.001

        # Ensure t >= r
        t_new = torch.max(t, r)
        r_new = torch.min(t, r)
        t, r = t_new, r_new

        # First data_proportion samples: force r = t (standard FM)
        data_size = int(B * self.data_proportion)
        fm_mask = torch.arange(B, device=device) < data_size
        r = torch.where(fm_mask, t, r)

        return t, r

    # ------------------------------------------------------------------
    # Adaptive weighting (from iMF paper)
    # ------------------------------------------------------------------

    @staticmethod
    def _adp_wt_fn(loss, norm_eps: float = 0.01, norm_p: float = 1.0):
        """Per-sample adaptive weighting.

        Normalizes loss so all samples contribute more equally:
        - High-loss (hard) samples get dampened
        - Low-loss (easy) samples get preserved but reduced
        """
        adp_wt = (loss + norm_eps) ** norm_p
        return loss / adp_wt.detach()

    # ------------------------------------------------------------------
    # Exact JVP via forward-mode AD
    # ------------------------------------------------------------------

    def _compute_jvp(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond: Optional[torch.Tensor],
        freq_mode: str,
        v_tangent: torch.Tensor,
    ):
        """Compute du/dz·v + du/dt with exact forward-mode AD.

        The MeanFlow equation needs: V = u + h * (du/dz·v + du/dt)
        where the directional derivative du/dz·v + du/dt is the JVP with
        tangents (v, 1, 0) for inputs (z_t, t, r).

        Args:
            z_t: (B, T, D) noisy actions
            t: (B,) time values (float, NOT scaled by 1000)
            r: (B,) time values (float, NOT scaled by 1000)
            obs_tokens: (B, N, n_emb) observation tokens
            temporal_cond: (B, T, n_emb) or None
            freq_mode: "10hz" or "50hz"
            v_tangent: (B, T, D) tangent vector for z_t

        Returns:
            u: (B, T, D) average velocity prediction (from base point)
            v: (B, T, D) auxiliary instantaneous velocity prediction
            du_combined: (B, T, D) exact du/dz·v + du/dt
        """
        def u_fn(z, t_val, r_val):
            h_val = t_val - r_val
            u_out, v_out = self.model(
                z, t_val, h_val,
                cond=obs_tokens,
                temporal_cond=temporal_cond,
                freq_mode=freq_mode,
            )
            return u_out, v_out

        u, du_combined, v = jvp(
            u_fn,
            (z_t, t, r),
            (v_tangent, torch.ones_like(t), torch.zeros_like(r)),
            has_aux=True,
        )

        return u, v, du_combined

    # ------------------------------------------------------------------
    # iMF loss for one frequency
    # ------------------------------------------------------------------

    def _compute_imf_loss_one_freq(
        self,
        x0: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond: Optional[torch.Tensor],
        freq_mode: str,
    ):
        """Compute improved MeanFlow loss for one frequency head.

        Args:
            x0: (B, T, D) clean action sequences
            obs_tokens: (B, N, n_emb) observation tokens
            temporal_cond: (B, T, n_emb) or None
            freq_mode: "10hz" or "50hz"

        Returns:
            loss: scalar loss
            loss_dict: dict of loss components for logging
        """
        B = x0.shape[0]
        device = x0.device

        # 1. Sample t, r
        t, r = self._sample_tr(B, device)

        # 2. Construct noisy sample
        noise = torch.randn_like(x0)
        t_exp = t.view(B, 1, 1)
        z_t = t_exp * noise + (1.0 - t_exp) * x0
        v_true = noise - x0  # true instantaneous velocity

        # 3. Exact JVP forward pass → get u, v, and du/dz·v_true + du/dt
        h = t - r
        u_pred, v_pred, du_combined = self._compute_jvp(
            z_t, t, r,
            obs_tokens, temporal_cond, freq_mode,
            v_tangent=v_true.detach(),
        )

        # 4. Compound function: V = u + h * du/dt
        h_exp = h.view(B, 1, 1)
        V = u_pred + h_exp * du_combined.detach()

        # 5. Targets (stop gradient)
        v_target = v_true.detach()

        # 6. Per-sample losses with adaptive weighting.
        # Keep raw MSE metrics for logging; the adaptive loss is often near
        # a constant when norm_p=1.0.
        sq_u = (V - v_target) ** 2
        sq_v = (v_pred - v_target) ** 2
        raw_loss_u = sq_u.mean(dim=(1, 2))
        raw_loss_v = sq_v.mean(dim=(1, 2))

        loss_u = sq_u.sum(dim=(1, 2))
        loss_u = self._adp_wt_fn(loss_u, self.norm_eps, self.norm_p)

        loss_v = sq_v.sum(dim=(1, 2))
        loss_v = self._adp_wt_fn(loss_v, self.norm_eps, self.norm_p)

        loss = (loss_u + loss_v).mean()

        loss_dict = {
            f"imf_loss_u_{freq_mode}": loss_u.mean().item(),
            f"imf_loss_v_{freq_mode}": loss_v.mean().item(),
            f"raw_imf_loss_u_{freq_mode}": raw_loss_u.mean().item(),
            f"raw_imf_loss_v_{freq_mode}": raw_loss_v.mean().item(),
            f"raw_imf_loss_total_{freq_mode}": (
                raw_loss_u.mean() + raw_loss_v.mean()
            ).item(),
        }

        return loss, loss_dict

    # ------------------------------------------------------------------
    # compute_loss (single-stage, replaces 2-stage CTM)
    # ------------------------------------------------------------------

    def compute_loss(self, batch, loss_mask=None):
        nobs = self.normalizer.normalize(batch["obs"])
        nactions_10 = self.normalizer["action"].normalize(batch["action"])
        nactions_50 = self.normalizer["action_stack"].normalize(batch["action_stack"])
        B = nactions_10.shape[0]

        # Temporal conditions
        if self.rnn_lowfreq is not None and self.rnn_highfreq is not None:
            assert "extended_obs" in batch, "extended_obs is required for RNNObsEncoder"
            temporal_cond_10hz = self.get_temporal_cond(batch["extended_obs"], head="10hz")
            temporal_cond_50hz = self.get_temporal_cond(batch["extended_obs"], head="50hz")
        else:
            temporal_cond_10hz = None
            temporal_cond_50hz = None

        # Encode observations
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])

        # iMF loss for both frequencies
        loss_10, dict_10 = self._compute_imf_loss_one_freq(
            nactions_10, obs_tokens, temporal_cond_10hz, "10hz"
        )
        loss_50, dict_50 = self._compute_imf_loss_one_freq(
            nactions_50, obs_tokens, temporal_cond_50hz, "50hz"
        )

        total_loss = loss_10 + loss_50

        loss_dict = {
            "total_loss": total_loss.item(),
            **dict_10,
            **dict_50,
        }
        loss_dict["raw_total_loss"] = (
            loss_dict["raw_imf_loss_total_10hz"]
            + loss_dict["raw_imf_loss_total_50hz"]
        )

        return total_loss, loss_dict

    def forward(self, batch, loss_mask=None):
        return self.compute_loss(batch, loss_mask)

    # ------------------------------------------------------------------
    # iMF Euler sampler (replaces both conditional_sample & consistency_sample)
    # ------------------------------------------------------------------

    def imf_sample(
        self,
        cond=None,
        temporal_cond=None,
        freq_mode: str = "10hz",
        generator=None,
        num_inference_steps: Optional[int] = None,
        noisy_trajectory=None,
        **kwargs,
    ):
        """iMF Euler sampler using average velocity u.

        1-step: x_0 = x_1 - 1.0 * u(x_1, t=1.0, h=1.0)
        N-step: Euler integration with u at each step
        """
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

            t_tensor = torch.full(
                (B,), t.item(), dtype=x.dtype, device=x.device
            )
            h_tensor = torch.full(
                (B,), h.item(), dtype=x.dtype, device=x.device
            )

            u, _ = self.model(
                x,
                t_tensor,
                h_tensor,
                cond=cond,
                temporal_cond=temporal_cond,
                freq_mode=freq_mode,
            )

            x = x - h * u

        return x

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

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

        # Both frequencies use the same iMF sampler
        nsample_10 = self.imf_sample(
            cond=obs_tokens,
            temporal_cond=temporal_cond_10hz,
            freq_mode="10hz",
            num_inference_steps=steps,
        )
        nsample_50 = self.imf_sample(
            cond=obs_tokens,
            temporal_cond=temporal_cond_50hz,
            freq_mode="50hz",
            num_inference_steps=steps,
        )

        # Unnormalize
        assert nsample_10.shape == (B, temporal_cond_10hz.shape[1], self.action_dim)
        assert nsample_50.shape == (B, temporal_cond_50hz.shape[1], self.action_dim)
        action_pred_10hz = self.normalizer["action"].unnormalize(nsample_10)
        action_pred_50hz = self.normalizer["action_stack"].unnormalize(nsample_50)

        # Extract action windows
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

    # ------------------------------------------------------------------
    # Reactive inference (obs_tokens + noisy_trajectory interface)
    # ------------------------------------------------------------------

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
                    1, max_extended_obs_last_step - extended_obs_last_step_list[i], 1
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

        # Both frequencies use the same iMF sampler
        nsample = self.imf_sample(
            cond=obs_tokens,
            temporal_cond=temporal_cond,
            freq_mode=head,
            noisy_trajectory=noisy_trajectory,
        )

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
                    0, min(extended_obs_last_step_list[i] - start, end - start)
                )
                results.append({
                    "action": action[i: i + 1, :valid_len, :],
                    "action_pred": action_pred[
                        i: i + 1, :extended_obs_last_step_list[i], :
                    ],
                })
        return results
