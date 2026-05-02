"""
Two-stage multi-frequency FM policy with CTM-style distillation.

Stage 1 is standard multi-frequency rectified-flow velocity matching. Stage 2
uses the Stage-1 model as a frozen FM teacher and trains a stop-time-conditioned
student transition:

    G_theta(x_t, t, s) = x_t + (s - t) * v_theta(x_t, t, s)

The CTM loss follows the consistency-policy pattern, adapted to flow matching:
teacher moves x_t from t to u, student maps x_t -> s, EMA target maps x_u -> s,
then both are optionally chained to 0 for the final consistency target.
"""
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from loguru import logger

from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.policy.fm_multifreq_policy import FMMultiFreqPolicy
from ImplicitRDP.model.diffusion.transformer_for_ctm_multifreq import (
    TransformerForCTMMultiFreq,
)


class FMMultiFreqCTMPolicy(FMMultiFreqPolicy):
    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: torch.nn.Module,
        model: TransformerForCTMMultiFreq,
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
        ctm_num_scales: int = 40,
        ctm_beta_a: Optional[float] = None,
        ctm_beta_b: Optional[float] = None,
        ctm_loss_weight: float = 1.0,
        dsm_loss_weight: float = 0.5,
        ctm_chain_to_zero: bool = True,
        stop_time_sample_mode: str = "mixed",
        stop_time_zero_prob: float = 0.5,
        ctm_delta: float = 0.0,
        ctm_freq: str = "50hz",
        inference_head: str = "50hz",
        training_stage: int = 1,
        fm_preserve_loss_weight: float = 1.0,
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
            cd_num_scales=ctm_num_scales,
            cd_beta_a=beta_a if ctm_beta_a is None else ctm_beta_a,
            cd_beta_b=beta_b if ctm_beta_b is None else ctm_beta_b,
            inference_head=inference_head,
            training_stage=training_stage,
            **kwargs,
        )

        self.ctm_num_scales = ctm_num_scales
        self.ctm_beta_a = beta_a if ctm_beta_a is None else ctm_beta_a
        self.ctm_beta_b = beta_b if ctm_beta_b is None else ctm_beta_b
        self.ctm_loss_weight = ctm_loss_weight
        self.dsm_loss_weight = dsm_loss_weight
        self.ctm_chain_to_zero = ctm_chain_to_zero
        self.stop_time_sample_mode = stop_time_sample_mode
        self.stop_time_zero_prob = stop_time_zero_prob
        self.ctm_delta = ctm_delta
        self.ctm_freq = ctm_freq
        self.fm_preserve_loss_weight = fm_preserve_loss_weight

        assert self.ctm_num_scales >= 2
        assert self.ctm_freq in ("10hz", "50hz", "both")
        assert self.stop_time_sample_mode in ("zero", "uniform", "mixed")
        assert 0.0 <= self.stop_time_zero_prob <= 1.0
        assert self.ctm_loss_weight >= 0.0
        assert self.dsm_loss_weight >= 0.0
        assert self.fm_preserve_loss_weight >= 0.0

        logger.info(
            "FMMultiFreqCTMPolicy initialized "
            f"(stage={training_stage}, ctm_num_scales={ctm_num_scales}, "
            f"ctm_beta=({self.ctm_beta_a}, {self.ctm_beta_b}), "
            f"ctm_freq={ctm_freq}, ctm_chain_to_zero={ctm_chain_to_zero}, "
            f"loss_weights=({ctm_loss_weight}, {dsm_loss_weight}), "
            f"fm_preserve_loss_weight={fm_preserve_loss_weight})"
        )

    def set_teacher_model(self, teacher):
        # Store as a buffer-like attribute that follows .to(device) but does
        # NOT appear in state_dict() (we don't want to save the teacher).
        # Using register_module would include it in state_dict; using
        # __dict__ would skip .to(device).  Solution: register as a module
        # but exclude it explicitly during save via exclude_keys.
        if hasattr(self, "_frozen_teacher_ref"):
            del self._frozen_teacher_ref
        # Keep a plain reference that we access in forward code
        self._frozen_teacher_ref = [teacher]  # wrap in list to avoid nn.Module registration
        self.__dict__["frozen_teacher"] = teacher

    def set_ema_model_ref(self, ema_model):
        self.ema_model_ref = ema_model

    def model_forward(self, batch):
        """Stage-1 FM velocity matching with stop_time omitted."""
        assert "valid_mask" not in batch

        nobs = self.normalizer.normalize(batch["obs"])
        nactions_10 = self.normalizer["action"].normalize(batch["action"])
        nactions_50 = self.normalizer["action_stack"].normalize(batch["action_stack"])
        B = nactions_10.shape[0]

        if self.rnn_lowfreq is not None and self.rnn_highfreq is not None:
            assert "extended_obs" in batch, "extended_obs is required for RNNObsEncoder"
            temporal_cond_10hz = self.get_temporal_cond(batch["extended_obs"], head="10hz")
            temporal_cond_50hz = self.get_temporal_cond(batch["extended_obs"], head="50hz")
        else:
            temporal_cond_10hz = None
            temporal_cond_50hz = None

        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])

        t = torch.distributions.Beta(self.beta_a, self.beta_b).sample(
            (B,)
        ).to(nactions_10.device)
        t = t * 0.999 + 0.001
        timesteps = (t * 1000).long()
        stop_timesteps = torch.zeros_like(timesteps)
        t_exp = t.view(B, 1, 1)

        noise_10 = torch.randn_like(nactions_10)
        x_t_10 = t_exp * noise_10 + (1.0 - t_exp) * nactions_10
        pred_v_10 = self.model(
            x_t_10,
            timesteps,
            stop_timestep=stop_timesteps,
            cond=obs_tokens,
            temporal_cond=temporal_cond_10hz,
            freq_mode="10hz",
        )

        noise_50 = torch.randn_like(nactions_50)
        x_t_50 = t_exp * noise_50 + (1.0 - t_exp) * nactions_50
        pred_v_50 = self.model(
            x_t_50,
            timesteps,
            stop_timestep=stop_timesteps,
            cond=obs_tokens,
            temporal_cond=temporal_cond_50hz,
            freq_mode="50hz",
        )

        return pred_v_10, noise_10 - nactions_10, pred_v_50, noise_50 - nactions_50

    def _sample_t_u_s(self, B: int, device: torch.device):
        """Sample (t, u, s) on the discrete grid, matching the original CTM.

        All three values live on the grid  t_RF = 1 - index / N  so that
        training timesteps are perfectly aligned with the inference grid.

        In bin-index space  (index 0 = pure noise,  index N = clean data):
            t_idx  ∈ [0, N-2]            Beta-distributed
            s_idx  ∈ [t_idx, N]          uniform, optionally biased to N (s=0)
            u_idx  = min(t_idx + 1, s_idx)   one teacher Heun step
        """
        N = self.ctm_num_scales

        # ---- t: Beta → snap to grid index [0, N-2] ----
        noise_level = torch.distributions.Beta(
            self.ctm_beta_a, self.ctm_beta_b
        ).sample((B,)).to(device)
        t_indices = ((1.0 - noise_level) * (N - 1)).long().clamp(0, N - 2)

        # ---- s: uniform over [t_idx, N] on the grid (original CTM style) ----
        range_sizes = (N - t_indices + 1).float()          # number of valid bins
        s_offsets = torch.floor(torch.rand(B, device=device) * range_sizes).long()
        s_indices = (t_indices + s_offsets).clamp(max=N)

        if self.stop_time_sample_mode == "zero":
            s_indices = torch.full_like(t_indices, N)
        elif self.stop_time_sample_mode == "mixed":
            use_zero = torch.rand(B, device=device) < self.stop_time_zero_prob
            s_indices = torch.where(
                use_zero, torch.full_like(t_indices, N), s_indices
            )
        # "uniform" → keep s as sampled above

        # ---- u: one Heun step toward clean, never past s (original CTM) ----
        u_indices = torch.min(t_indices + 1, s_indices)

        # ---- convert bin indices to RF time ----
        t = 1.0 - t_indices.float() / N
        u = 1.0 - u_indices.float() / N
        s = (1.0 - s_indices.float() / N).clamp(min=0.0)

        return t, u, s

    def _transition(
        self,
        model,
        x: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond: torch.Tensor,
        freq_mode: str,
    ):
        t_steps = (t * 1000).long()
        s_steps = (s * 1000).long()
        v = model(
            x,
            t_steps,
            stop_timestep=s_steps,
            cond=obs_tokens,
            temporal_cond=temporal_cond,
            freq_mode=freq_mode,
        )
        return x + (s - t).view(x.shape[0], 1, 1) * v

    @staticmethod
    def _huber_or_mse(pred: torch.Tensor, target: torch.Tensor, delta: float):
        if delta <= 0:
            return F.mse_loss(pred, target)
        # Pseudo-Huber loss: sqrt((pred - target)² + δ²) - δ
        # F.mse_loss with reduction="none" already gives (pred - target)²
        sq_err = F.mse_loss(pred, target, reduction="none")
        return (torch.sqrt(sq_err + delta * delta) - delta).mean()

    def _teacher_heun_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        u: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond: torch.Tensor,
        freq_mode: str,
    ):
        t_steps = (t * 1000).long()
        u_steps = (u * 1000).long()
        stop_steps = torch.zeros_like(t_steps)
        dt = (u - t).view(x_t.shape[0], 1, 1)
        with torch.no_grad():
            v_t = self.frozen_teacher(
                x_t,
                t_steps,
                stop_timestep=stop_steps,
                cond=obs_tokens,
                temporal_cond=temporal_cond,
                freq_mode=freq_mode,
            )
            x_euler = x_t + dt * v_t
            v_u = self.frozen_teacher(
                x_euler,
                u_steps,
                stop_timestep=stop_steps,
                cond=obs_tokens,
                temporal_cond=temporal_cond,
                freq_mode=freq_mode,
            )
            return x_t + 0.5 * dt * (v_t + v_u)

    def _compute_ctm_loss_one_freq(
        self,
        x0: torch.Tensor,
        obs_tokens: torch.Tensor,
        temporal_cond: torch.Tensor,
        freq_mode: str,
    ):
        assert self.frozen_teacher is not None, "frozen_teacher must be set for Stage 2"
        assert self.ema_model_ref is not None, "ema_model_ref must be set for Stage 2"

        B = x0.shape[0]
        device = x0.device
        noise = torch.randn_like(x0)
        t, u, s = self._sample_t_u_s(B, device)

        x_t = t.view(B, 1, 1) * noise + (1.0 - t.view(B, 1, 1)) * x0
        x_u = self._teacher_heun_step(x_t, t, u, obs_tokens, temporal_cond, freq_mode)

        pred_s = self._transition(
            self.model, x_t, t, s, obs_tokens, temporal_cond, freq_mode
        )
        with torch.no_grad():
            target_s = self._transition(
                self.ema_model_ref,
                x_u,
                u,
                s,
                obs_tokens,
                temporal_cond,
                freq_mode,
            )

        if self.ctm_chain_to_zero:
            # Only chain samples where s > 0; when s == 0 the transition
            # G(x, 0, 0) = x (identity), so the two extra ema_model_ref
            # forward passes would be pure waste.
            zero = torch.zeros_like(s)
            needs_chain = s > 1e-6  # per-sample mask

            if needs_chain.any():
                # Run chain-to-zero only on the subset that actually needs it
                idx = needs_chain.nonzero(as_tuple=True)[0]
                pred_s_sub = pred_s[idx]
                target_s_sub = target_s[idx]
                s_sub = s[idx]
                zero_sub = zero[idx]
                obs_sub = obs_tokens[idx]
                tc_sub = temporal_cond[idx] if temporal_cond is not None else None

                pred_chain_sub = self._transition(
                    self.ema_model_ref, pred_s_sub, s_sub, zero_sub,
                    obs_sub, tc_sub, freq_mode,
                )
                with torch.no_grad():
                    target_chain_sub = self._transition(
                        self.ema_model_ref, target_s_sub, s_sub, zero_sub,
                        obs_sub, tc_sub, freq_mode,
                    )

                # Assemble full tensors: chain result where s>0, direct where s==0
                pred_cmp = pred_s.clone()
                pred_cmp[idx] = pred_chain_sub
                target_cmp = target_s.clone()
                target_cmp[idx] = target_chain_sub
            else:
                # All s == 0 → chain is identity, skip entirely
                pred_cmp = pred_s
                target_cmp = target_s
        else:
            pred_cmp = pred_s
            target_cmp = target_s

        ctm_loss = self._huber_or_mse(pred_cmp, target_cmp.detach(), self.ctm_delta)

        # ---- DSM loss with independent continuous t (matches original CTM) ----
        # Original uses ctm_dsm sampler = 50% uniform + 50% log-normal,
        # covering continuous time range (not just the discrete grid).
        # We adapt this for Rectified Flow: t ∈ [0.001, 1.0] continuous.
        B_half = B // 2
        B_rest = B - B_half

        # 50%: uniform over [0.001, 1.0]
        t_uniform = torch.rand(B_half, device=device) * 0.999 + 0.001
        # 50%: Beta distribution (our analog of log-normal, biased toward 1.0)
        t_beta = torch.distributions.Beta(
            self.ctm_beta_a, self.ctm_beta_b
        ).sample((B_rest,)).to(device)
        t_beta = t_beta * 0.999 + 0.001

        t_dsm = torch.cat([t_uniform, t_beta], dim=0)
        # Shuffle to mix the two distributions
        perm = torch.randperm(B, device=device)
        t_dsm = t_dsm[perm]

        noise_dsm = torch.randn_like(x0)
        x_t_dsm = t_dsm.view(B, 1, 1) * noise_dsm + (1.0 - t_dsm.view(B, 1, 1)) * x0
        zero_dsm = torch.zeros_like(t_dsm)

        pred_x0 = self._transition(
            self.model, x_t_dsm, t_dsm, zero_dsm, obs_tokens, temporal_cond, freq_mode
        )

        # Rectified Flow weighting: 1/t to balance noise levels
        # (analog of Karras weighting (σ²+σ_d²)/(σ·σ_d)² for RF)
        dsm_weights = (1.0 / t_dsm).view(B, 1, 1)
        dsm_weights = dsm_weights / dsm_weights.mean()  # normalize to keep loss scale

        dsm_loss_raw = F.mse_loss(pred_x0, x0, reduction="none")
        dsm_loss = (dsm_loss_raw * dsm_weights).mean()

        total = (
            self.ctm_loss_weight * ctm_loss
            + self.dsm_loss_weight * dsm_loss
        )
        return total, ctm_loss.detach(), dsm_loss.detach()

    def _compute_ctm_loss(self, batch):
        nobs = self.normalizer.normalize(batch["obs"])
        nactions_10 = self.normalizer["action"].normalize(batch["action"])
        nactions_50 = self.normalizer["action_stack"].normalize(batch["action_stack"])
        B = nactions_10.shape[0]

        if self.rnn_lowfreq is not None and self.rnn_highfreq is not None:
            assert "extended_obs" in batch, "extended_obs is required for RNNObsEncoder"
            temporal_cond_10hz = self.get_temporal_cond(batch["extended_obs"], head="10hz")
            temporal_cond_50hz = self.get_temporal_cond(batch["extended_obs"], head="50hz")
        else:
            temporal_cond_10hz = None
            temporal_cond_50hz = None

        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])

        total_loss = nactions_10.new_tensor(0.0)
        metrics = {}

        if self.ctm_freq in ("10hz", "both"):
            loss_10, ctm_10, dsm_10 = self._compute_ctm_loss_one_freq(
                nactions_10, obs_tokens, temporal_cond_10hz, "10hz"
            )
            total_loss = total_loss + loss_10
            metrics.update({
                "ctm_loss_10hz": ctm_10.item(),
                "dsm_loss_10hz": dsm_10.item(),
            })

        if self.ctm_freq in ("50hz", "both"):
            loss_50, ctm_50, dsm_50 = self._compute_ctm_loss_one_freq(
                nactions_50, obs_tokens, temporal_cond_50hz, "50hz"
            )
            total_loss = total_loss + loss_50
            metrics.update({
                "ctm_loss_50hz": ctm_50.item(),
                "dsm_loss_50hz": dsm_50.item(),
            })

        # ---- Fix: 10hz FM velocity-matching preservation loss ----
        # When ctm_freq is "50hz" (default), the 10hz head receives no
        # gradient signal.  Since 10hz and 50hz share the backbone, 50hz
        # CTM gradients corrupt the 10hz head (catastrophic forgetting).
        # This preservation loss keeps 10hz aligned with its Stage-1
        # velocity field through the shared backbone.
        if self.fm_preserve_loss_weight > 0.0 and self.ctm_freq != "both":
            # Only needed when ctm_freq does NOT already cover 10hz
            t = torch.distributions.Beta(self.beta_a, self.beta_b).sample(
                (B,)
            ).to(nactions_10.device)
            t = t * 0.999 + 0.001
            timesteps = (t * 1000).long()
            stop_timesteps = torch.zeros_like(timesteps)
            t_exp = t.view(B, 1, 1)

            noise_10 = torch.randn_like(nactions_10)
            x_t_10 = t_exp * noise_10 + (1.0 - t_exp) * nactions_10
            pred_v_10 = self.model(
                x_t_10,
                timesteps,
                stop_timestep=stop_timesteps,
                cond=obs_tokens,
                temporal_cond=temporal_cond_10hz,
                freq_mode="10hz",
            )
            target_v_10 = noise_10 - nactions_10
            fm_preserve_loss = F.mse_loss(pred_v_10, target_v_10)
            total_loss = total_loss + self.fm_preserve_loss_weight * fm_preserve_loss
            metrics["fm_preserve_loss_10hz"] = fm_preserve_loss.item()

        metrics["total_loss"] = total_loss.item()
        metrics["training_stage"] = 2.0
        return total_loss, metrics

    def compute_loss(self, batch, loss_mask=None):
        if self.training_stage == 1:
            return super().compute_loss(batch, loss_mask)
        if self.training_stage == 2:
            return self._compute_ctm_loss(batch)
        raise ValueError(f"Invalid training_stage={self.training_stage}")

    def conditional_sample(
        self,
        cond=None,
        temporal_cond=None,
        freq_mode: str = "10hz",
        generator=None,
        num_inference_steps: Optional[int] = None,
        noisy_trajectory=None,
        **kwargs,
    ):
        """Rectified-flow Euler sampler with fixed stop_time=0."""
        steps = num_inference_steps if num_inference_steps is not None else self.num_inference_steps
        if noisy_trajectory is None:
            trajectory = torch.randn(
                size=(cond.shape[0], temporal_cond.shape[1], self.action_dim),
                dtype=cond.dtype,
                device=cond.device,
                generator=generator,
            )
        else:
            trajectory = noisy_trajectory

        B = trajectory.shape[0]
        t_schedule = torch.linspace(1.0, 0.0, steps + 1, device=trajectory.device)
        for i in range(steps):
            t_cur = t_schedule[i]
            dt = t_schedule[i + 1] - t_schedule[i]
            t_tensor = torch.full((B,), t_cur * 1000, device=trajectory.device).long()
            stop_tensor = torch.zeros_like(t_tensor)

            model_output = self.model(
                trajectory,
                t_tensor,
                stop_timestep=stop_tensor,
                cond=cond,
                temporal_cond=temporal_cond,
                freq_mode=freq_mode,
            )
            trajectory = trajectory + dt * model_output

        return trajectory

    def consistency_sample(
        self,
        cond=None,
        temporal_cond=None,
        freq_mode: str = "50hz",
        generator=None,
        num_inference_steps: Optional[int] = None,
        noisy_trajectory=None,
        **kwargs,
    ):
        """Chaining-style CTM sampler (matches consistency-policy design).

        Every step jumps all the way to s=0 (the most-trained target),
        then re-noises to a grid-aligned intermediate t for the next step.
        This avoids using (t, s) pairs that were rarely seen during training.

        1-step:  G(noise, t=1.0, s=0) → clean
        2-step:  G(noise, t=1.0, s=0) → clean
                 re-noise to t_mid → G(x, t_mid, s=0) → clean_refined
        """
        steps = num_inference_steps if num_inference_steps is not None else 1
        if noisy_trajectory is None:
            x_t = torch.randn(
                size=(cond.shape[0], temporal_cond.shape[1], self.action_dim),
                dtype=cond.dtype,
                device=cond.device,
                generator=generator,
            )
        else:
            x_t = noisy_trajectory

        B = x_t.shape[0]
        N = self.ctm_num_scales
        zero = torch.zeros(B, device=x_t.device)

        if steps == 1:
            # Single step: G(noise, t=1.0, s=0) → clean
            t = torch.ones(B, device=x_t.device)
            x_t = self._transition(
                self.model, x_t, t, zero, cond, temporal_cond, freq_mode,
            )
        else:
            # Multi-step chaining: pick re-noise points on the training grid.
            # E.g. 2-step with N=120: chain at grid index 60 → t = 1-60/120 = 0.5
            # E.g. 3-step with N=120: chain at indices 40,80 → t = 0.667, 0.333
            chain_indices = torch.linspace(0, N, steps + 1)[1:-1].long()
            chain_times = 1.0 - chain_indices.float() / N  # on the training grid

            for i in range(steps):
                if i == 0:
                    t = torch.ones(B, device=x_t.device)
                else:
                    t = torch.full((B,), chain_times[i - 1].item(), device=x_t.device)

                # Jump all the way to s=0
                x_clean = self._transition(
                    self.model, x_t, t, zero, cond, temporal_cond, freq_mode,
                )

                if i < steps - 1:
                    # Re-noise to the next grid-aligned chaining point
                    t_next = chain_times[i].item()
                    eps = torch.randn_like(x_clean)
                    x_t = t_next * eps + (1.0 - t_next) * x_clean
                else:
                    x_t = x_clean

        return x_t

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
        if head == "50hz":
            nsample = self.consistency_sample(
                cond=obs_tokens,
                temporal_cond=temporal_cond,
                freq_mode="50hz",
                noisy_trajectory=noisy_trajectory,
            )
        else:
            nsample = self.conditional_sample(
                cond=obs_tokens,
                temporal_cond=temporal_cond,
                freq_mode="10hz",
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

    def _predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        dataset_obs_temporal_downsample_ratio: int = 1,
        extended_obs_dict: Dict[str, torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.training_stage != 2:
            return super()._predict_action(
                obs_dict,
                dataset_obs_temporal_downsample_ratio=dataset_obs_temporal_downsample_ratio,
                extended_obs_dict=extended_obs_dict,
                num_inference_steps=num_inference_steps,
            )

        obs_tokens = self.predict_obs_tokens(obs_dict)
        B = obs_tokens.shape[0]
        obs_tokens = obs_tokens.reshape(B, -1, self.n_emb)
        assert extended_obs_dict is not None, "extended_obs is required for RNN temporal condition"

        temporal_cond_10hz = self.get_temporal_cond(extended_obs_dict, head="10hz")
        temporal_cond_50hz = self.get_temporal_cond(extended_obs_dict, head="50hz")
        steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps

        nsample_10 = self.conditional_sample(
            cond=obs_tokens,
            temporal_cond=temporal_cond_10hz,
            freq_mode="10hz",
            num_inference_steps=steps,
        )
        nsample_50 = self.consistency_sample(
            cond=obs_tokens,
            temporal_cond=temporal_cond_50hz,
            freq_mode="50hz",
            num_inference_steps=steps,
        )

        action_pred_10hz = self.normalizer["action"].unnormalize(nsample_10)
        action_pred_50hz = self.normalizer["action_stack"].unnormalize(nsample_50)

        action_10hz = action_pred_10hz[
            :, self.action_start_10hz:self.action_start_10hz + self.n_action_steps
        ]
        action_50hz = action_pred_50hz[:, self.action_start_50hz:self.horizon_50hz]

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
