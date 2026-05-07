"""
Multi-frequency Diffusion Policy with CTM (Consistency Trajectory Model) distillation.

Uses the **Karras / EDM** noise framework (NOT Rectified Flow) to match
the original consistency-policy implementation exactly.

Stage 1:  Karras EDM denoising on both 10hz and 50hz heads.
Stage 2:  CTM distillation on 50hz only  +  10hz EDM preservation loss.

Inference:
  10hz  ->  multi-step Karras Heun ODE  (free-move stage)
  50hz  ->  single-step CTM             (contact-rich stage)

The Transformer backbone (TransformerForCTMMultiFreq) is reused as-is.
All Karras / CTM math is delegated to the schedulers in
ImplicitRDP.model.diffusion.karras_diffusion.
"""

from typing import Dict, List, Optional, Tuple, Union
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from loguru import logger

from ImplicitRDP.model.common.normalizer import LinearNormalizer
from ImplicitRDP.policy.base_image_policy import BaseImagePolicy
from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.model.diffusion.transformer_for_ctm_multifreq import (
    TransformerForCTMMultiFreq,
)
from ImplicitRDP.model.force.rnn import RNN

# ---- Karras / CTM schedulers & loss (local copy, no external dependency) ----
from ImplicitRDP.model.diffusion.karras_diffusion import (
    Karras_Scheduler,
    CTM_Scheduler,
    Huber_Loss,
)


class DPCTMMultiFreqPolicy(BaseImagePolicy):
    """Multi-frequency policy: Karras EDM (Stage 1) + CTM distillation (Stage 2)."""

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: nn.Module,
        model: TransformerForCTMMultiFreq,
        karras_scheduler: Karras_Scheduler,
        ctm_scheduler: CTM_Scheduler,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        n_emb: int = 768,
        # RNN temporal encoder
        use_rnn_obs_encoder: bool = True,
        rnn_obs_encoder_n_layer: int = 1,
        rnn_obs_encoder_hidden_dim: int = 512,
        # Multi-freq
        horizon_10hz: int = 16,
        horizon_50hz: int = 80,
        action_start_10hz: int = 1,
        action_start_50hz: int = 13,
        # CTM / EMA
        initial_ema_decay: float = 0.0,
        delta: float = -1.0,
        losses: list = None,
        dsm_weights: str = "karras",
        chaining_times: list = None,
        # Stage control
        training_stage: int = 1,
        inference_head: str = "50hz",
        # AMP
        use_amp_for_inference: bool = False,
        amp_dtype: str = "bf16",
        **kwargs,
    ):
        super().__init__()

        if losses is None:
            losses = [["ctm", "dsm"], [1, 1]]
        if chaining_times is None:
            chaining_times = ["D", 27, 54]

        # ---- Action shape ----
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        # ---- Extended obs keys ----
        all_extended_obs_keys = (
            list(shape_meta["extended_obs"].keys())
            if "extended_obs" in shape_meta
            else []
        )
        self.extended_obs_keys = sorted(all_extended_obs_keys)

        temporal_cond_dim_low = shape_meta.get("extended_obs", {}).get(
            "right_robot_tcp_wrench", {}
        ).get("shape", [6])[-1]
        temporal_cond_dim_high = shape_meta.get("extended_obs", {}).get(
            "right_robot_tcp_wrench_stack", {}
        ).get("shape", [6])[-1]

        # ---- Two RNN encoders ----
        if use_rnn_obs_encoder:
            self.rnn_lowfreq = RNN(
                input_dim=temporal_cond_dim_low,
                hidden_dim=rnn_obs_encoder_hidden_dim,
                layer_num=rnn_obs_encoder_n_layer,
                n_emb=n_emb,
            )
            self.rnn_highfreq = RNN(
                input_dim=temporal_cond_dim_high,
                hidden_dim=rnn_obs_encoder_hidden_dim,
                layer_num=rnn_obs_encoder_n_layer,
                n_emb=n_emb,
            )
        else:
            self.rnn_lowfreq = None
            self.rnn_highfreq = None

        # ---- Core modules ----
        self.shape_meta = shape_meta
        self.obs_encoder = obs_encoder
        self.model = model

        # ---- Internal EMA (matches consistency-policy design) ----
        self.model_ema = copy.deepcopy(model)
        self.model_ema.requires_grad_(False)

        # ---- Frozen teacher (set by workspace at stage transition) ----
        self.frozen_teacher = None

        # ---- Schedulers ----
        self.karras_scheduler = karras_scheduler   # Stage1 + 10hz inference
        self.ctm_scheduler = ctm_scheduler         # Stage2 + 50hz inference

        # ---- Normalizer ----
        self.normalizer = LinearNormalizer()

        # ---- Scalars ----
        self.action_dim = action_dim
        self.horizon = horizon
        self.horizon_10hz = horizon_10hz
        self.horizon_50hz = horizon_50hz
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.n_emb = n_emb
        self.action_start_10hz = action_start_10hz
        self.action_start_50hz = action_start_50hz

        # ---- CTM ----
        self.ema_decay = initial_ema_decay
        self.delta = delta
        self.dsm_weights = dsm_weights
        self.chaining_times = chaining_times
        self.chain = False

        self.losses = {}
        for loss_name, weight in zip(losses[0], losses[1]):
            self.losses[loss_name] = weight

        # ---- Stage ----
        self.training_stage = training_stage
        self.inference_head = inference_head

        # ---- AMP ----
        self.use_amp_for_inference = use_amp_for_inference
        if use_amp_for_inference:
            if amp_dtype == "bf16" and torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
            elif amp_dtype == "fp16":
                self.amp_dtype = torch.float16
            else:
                self.amp_dtype = torch.float16
        else:
            self.amp_dtype = None

        self.kwargs = kwargs

        logger.info(
            f"DPCTMMultiFreqPolicy initialized "
            f"(stage={training_stage}, delta={delta}, "
            f"losses={self.losses}, dsm_weights={dsm_weights}, "
            f"ema_decay={initial_ema_decay}, "
            f"karras_bins={karras_scheduler.bins}, ctm_bins={ctm_scheduler.bins}, "
            f"inference_head={inference_head})"
        )
        logger.info("Transformer params: %e" % sum(p.numel() for p in self.model.parameters()))
        logger.info("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

    # ==================================================================
    # Stage / teacher control
    # ==================================================================
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def set_training_stage(self, stage: int):
        assert stage in (1, 2)
        self.training_stage = stage

    def set_teacher_model(self, teacher):
        self.frozen_teacher = teacher

    def drop_teacher(self):
        self.frozen_teacher = None

    def enable_chaining(self):
        if self.chaining_times is not None:
            self.chain = True
            logger.info(f"Chaining enabled with times: {self.chaining_times}")

    def disable_chaining(self):
        self.chain = False

    # ==================================================================
    # EMA update (internal, matches consistency-policy)
    # ==================================================================
    @torch.no_grad()
    def ema_update(self):
        param = [p.data for p in self.model.parameters()]
        param_ema = [p.data for p in self.model_ema.parameters()]
        torch._foreach_mul_(param_ema, self.ema_decay)
        torch._foreach_add_(param_ema, param, alpha=1 - self.ema_decay)

    # ==================================================================
    # Bridge helpers: connect Transformer to Karras/CTM schedulers
    # ==================================================================
    def _fixed_stop_rescaled(self, device):
        """Fixed stop-time value for EDM/teacher mode (= rescaled sigma_min)."""
        return (
            1000.0
            * 0.25
            * torch.log(torch.tensor(self.ctm_scheduler.time_min + 1e-44))
        ).to(device)

    def _make_edm_denoise(self, model, obs_tokens, temporal_cond, freq_mode):
        """Create denoise(traj, t) lambda for Karras calc_out / step."""
        fixed_stop = self._fixed_stop_rescaled(obs_tokens.device)

        def denoise(traj, t):
            rescaled_stop = fixed_stop.expand(traj.shape[0])
            return model(
                traj,
                t,
                stop_timestep=rescaled_stop,
                cond=obs_tokens,
                temporal_cond=temporal_cond,
                freq_mode=freq_mode,
            )

        return denoise

    def _make_ctm_denoise(self, model, obs_tokens, temporal_cond, freq_mode):
        """Create denoise(traj, t, s) lambda for CTM CTM_calc_out."""

        def denoise(traj, t, s):
            return model(
                traj,
                t,
                stop_timestep=s,
                cond=obs_tokens,
                temporal_cond=temporal_cond,
                freq_mode=freq_mode,
            )

        return denoise

    # ==================================================================
    # Temporal condition helpers (from FMMultiFreqPolicy)
    # ==================================================================
    def get_temporal_cond(
        self,
        extended_obs_dict,
        extended_obs_last_step: Optional[int] = None,
        extend_obs_pad_after_n: Optional[int] = None,
        head: str = "10hz",
    ):
        if extended_obs_dict is None:
            return None
        if head == "50hz":
            return self.get_temporal_cond_highfreq(
                extended_obs_dict, extended_obs_last_step, extend_obs_pad_after_n
            )
        else:
            return self.get_temporal_cond_lowfreq(
                extended_obs_dict, extended_obs_last_step, extend_obs_pad_after_n
            )

    def get_temporal_cond_highfreq(
        self,
        extended_obs_dict,
        extended_obs_last_step=None,
        extend_obs_pad_after_n=None,
    ):
        key = "right_robot_tcp_wrench_stack"
        obs = extended_obs_dict[key]
        if extended_obs_last_step is not None:
            obs = obs[:, -extended_obs_last_step:, ...]
        if extend_obs_pad_after_n is not None:
            padding_obs = obs[:, -1:, ...].repeat(1, extend_obs_pad_after_n, 1)
            obs = torch.cat([padding_obs, obs], dim=-2)
        obs = self.normalizer[f"extended_{key}"].normalize(obs)
        return self.rnn_highfreq(obs)

    def get_temporal_cond_lowfreq(
        self,
        extended_obs_dict,
        extended_obs_last_step=None,
        extend_obs_pad_after_n=None,
    ):
        key = "right_robot_tcp_wrench"
        obs = extended_obs_dict[key]
        if extended_obs_last_step is not None:
            obs = obs[:, -extended_obs_last_step:, ...]
        if extend_obs_pad_after_n is not None:
            padding_obs = obs[:, -1:, ...].repeat(1, extend_obs_pad_after_n, 1)
            obs = torch.cat([padding_obs, obs], dim=-2)
        obs = self.normalizer[f"extended_{key}"].normalize(obs)
        return self.rnn_lowfreq(obs)

    # ==================================================================
    # Obs encoding
    # ==================================================================
    def _encode_obs(self, batch, nobs, B):
        """Encode observations and temporal conditions."""
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])

        if self.rnn_lowfreq is not None and self.rnn_highfreq is not None:
            assert "extended_obs" in batch, "extended_obs is required for RNNObsEncoder"
            temporal_cond_10hz = self.get_temporal_cond(
                batch["extended_obs"], head="10hz"
            )
            temporal_cond_50hz = self.get_temporal_cond(
                batch["extended_obs"], head="50hz"
            )
        else:
            temporal_cond_10hz = None
            temporal_cond_50hz = None

        return obs_tokens, temporal_cond_10hz, temporal_cond_50hz

    def _predict_obs_tokens(self, obs_dict, return_expanded_obs_tokens=False):
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_tokens = self.obs_encoder(this_nobs)
        if len(obs_tokens.shape) == 2:
            obs_tokens = obs_tokens.reshape(B, -1, *obs_tokens.shape[1:])
        B = obs_tokens.shape[0]
        obs_tokens = obs_tokens.reshape(B, -1)
        return obs_tokens

    def predict_obs_tokens(self, *args, **kwargs):
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_obs_tokens(*args, **kwargs)

    # ==================================================================
    # Stage 1: EDM training (both frequencies)
    # ==================================================================
    def _compute_edm_loss(self, batch):
        """Karras EDM denoising score matching on both 10hz and 50hz."""
        nobs = self.normalizer.normalize(batch["obs"])
        nactions_10 = self.normalizer["action"].normalize(batch["action"])
        nactions_50 = self.normalizer["action_stack"].normalize(
            batch["action_stack"]
        )
        B = nactions_10.shape[0]

        obs_tokens, tc_10, tc_50 = self._encode_obs(batch, nobs, B)

        loss_10hz = nactions_10.new_tensor(0.0)
        loss_50hz = nactions_10.new_tensor(0.0)

        for trajectory, tc, freq in [
            (nactions_10, tc_10, "10hz"),
            (nactions_50, tc_50, "50hz"),
        ]:
            # Sample sigma ~ log_normal
            times, _ = self.karras_scheduler.sample_times(trajectory)
            # Add noise: x_sigma = x_0 + sigma * eps
            noisy_traj = self.karras_scheduler.add_noise(trajectory, times)
            # Model prediction D(x_sigma, sigma) with boundary condition
            denoise = self._make_edm_denoise(self.model, obs_tokens, tc, freq)
            pred = self.karras_scheduler.calc_out(denoise, noisy_traj, times, clamp=False)
            # Karras weighting + Huber loss
            weights = self.karras_scheduler.get_weights(times, None, "karras")
            loss = Huber_Loss(pred, trajectory, delta=self.delta, weights=weights)
            if freq == "10hz":
                loss_10hz = loss
            else:
                loss_50hz = loss

        total_loss = loss_10hz + loss_50hz
        return total_loss, loss_10hz, loss_50hz

    # ==================================================================
    # Stage 2: CTM distillation (50hz) + 10hz preservation
    # ==================================================================
    def _compute_ctm_loss(self, batch):
        """CTM distillation on 50hz + DSM + 10hz EDM preservation."""
        total_loss = {}

        nobs = self.normalizer.normalize(batch["obs"])
        nactions_50 = self.normalizer["action_stack"].normalize(
            batch["action_stack"]
        )
        B = nactions_50.shape[0]
        device = nactions_50.device

        obs_tokens, tc_10, tc_50 = self._encode_obs(batch, nobs, B)

        # ---- CTM loss (matches ctm_real_policy.py exactly) ----
        if "ctm" in self.losses:
            t, s, u = self.ctm_scheduler.sample_times(
                nactions_50, time_sampler="ctm"
            )
            sigma_t = self.ctm_scheduler.timesteps_to_times(t)
            sigma_s = self.ctm_scheduler.timesteps_to_times(s)
            sigma_u = self.ctm_scheduler.timesteps_to_times(u)

            noise_traj = self.ctm_scheduler.add_noise(nactions_50, sigma_t)

            # Teacher ODE: t -> u (Heun solver, stepwise)
            teacher_denoise = self._make_edm_denoise(
                self.frozen_teacher, obs_tokens, tc_50, "50hz"
            )
            u_noise_traj = noise_traj.clone()
            for d in range(self.ctm_scheduler.ode_steps_max):
                ct = torch.stack(
                    [
                        (t_i + d).clamp(int(t_i.item()), int(u_i.item()))
                        for t_i, u_i in zip(t, u)
                    ]
                )
                nt = torch.stack(
                    [
                        (t_i + d + 1).clamp(int(t_i.item()), int(u_i.item()))
                        for t_i, u_i in zip(t, u)
                    ]
                )
                current_times = self.ctm_scheduler.timesteps_to_times(ct)
                next_times = self.ctm_scheduler.timesteps_to_times(nt)
                u_noise_traj = self.ctm_scheduler.step(
                    teacher_denoise, u_noise_traj, current_times, next_times, clamp=False
                )

            # Student: G(x_t, sigma_t -> sigma_s)
            student_ctm = self._make_ctm_denoise(
                self.model, obs_tokens, tc_50, "50hz"
            )
            pred = self.ctm_scheduler.CTM_calc_out(
                student_ctm, noise_traj, sigma_t, sigma_s
            )

            # EMA target: G(x_u, sigma_u -> sigma_s)
            ema_ctm = self._make_ctm_denoise(
                self.model_ema, obs_tokens, tc_50, "50hz"
            )
            target = self.ctm_scheduler.CTM_calc_out(
                ema_ctm, u_noise_traj, sigma_u, sigma_s
            )

            # Chain to 0: project both to sigma_min
            sigma_min = torch.tensor(
                [self.ctm_scheduler.time_min], device=device
            ).expand(sigma_t.shape)
            pred = self.ctm_scheduler.CTM_calc_out(
                ema_ctm, pred, sigma_s, sigma_min
            )
            target = self.ctm_scheduler.CTM_calc_out(
                ema_ctm, target, sigma_s, sigma_min
            )

            loss_ctm = Huber_Loss(pred, target, delta=self.delta, weights=None)
            total_loss["ctm"] = loss_ctm * self.losses["ctm"]

        # ---- DSM loss (matches ctm_real_policy.py exactly) ----
        if "dsm" in self.losses:
            dsm_times, _ = self.ctm_scheduler.sample_times(
                nactions_50, time_sampler="ctm_dsm"
            )
            dsm_weights = self.ctm_scheduler.get_weights(
                dsm_times, None, self.dsm_weights
            )
            noisy_traj = self.ctm_scheduler.add_noise(nactions_50, dsm_times)
            sigma_min = torch.tensor(
                [self.ctm_scheduler.time_min], device=device
            ).expand(dsm_times.shape)

            student_ctm = self._make_ctm_denoise(
                self.model, obs_tokens, tc_50, "50hz"
            )
            pred = self.ctm_scheduler.CTM_calc_out(
                student_ctm, noisy_traj, dsm_times, sigma_min, clamp=False
            )

            loss_dsm = Huber_Loss(
                pred, nactions_50, delta=self.delta, weights=dsm_weights
            )
            total_loss["dsm"] = loss_dsm * self.losses["dsm"]

        # ---- 10hz preservation loss (prevent backbone forgetting) ----
        nactions_10 = self.normalizer["action"].normalize(batch["action"])
        times_10, _ = self.karras_scheduler.sample_times(nactions_10)
        noisy_10 = self.karras_scheduler.add_noise(nactions_10, times_10)
        denoise_10 = self._make_edm_denoise(
            self.model, obs_tokens, tc_10, "10hz"
        )
        pred_10 = self.karras_scheduler.calc_out(
            denoise_10, noisy_10, times_10, clamp=False
        )
        weights_10 = self.karras_scheduler.get_weights(times_10, None, "karras")
        loss_preserve = Huber_Loss(
            pred_10, nactions_10, delta=self.delta, weights=weights_10
        )
        total_loss["preserve_10hz"] = loss_preserve

        return total_loss

    # ==================================================================
    # compute_loss / forward
    # ==================================================================
    def compute_loss(self, batch, loss_mask=None):
        if self.training_stage == 1:
            loss, loss_10hz, loss_50hz = self._compute_edm_loss(batch)
            return loss, {
                "edm_loss": loss.item(),
                "edm_loss_10hz": loss_10hz.item(),
                "edm_loss_50hz": loss_50hz.item(),
                "training_stage": 1.0,
            }
        elif self.training_stage == 2:
            loss_dict = self._compute_ctm_loss(batch)
            total = sum(loss_dict.values())
            metrics = {k: v.item() for k, v in loss_dict.items()}
            metrics["total_loss"] = total.item()
            metrics["training_stage"] = 2.0
            return total, metrics
        else:
            raise ValueError(f"Invalid training_stage={self.training_stage}")

    def forward(self, batch, loss_mask=None):
        return self.compute_loss(batch, loss_mask)

    # ==================================================================
    # Inference: 10hz multi-step Karras ODE
    # ==================================================================
    def conditional_sample_10hz(
        self,
        obs_tokens,
        temporal_cond,
        generator=None,
    ):
        """Multi-step Karras Heun ODE for 10hz (matches EDM inference)."""
        B = obs_tokens.shape[0]
        device = obs_tokens.device
        dtype = obs_tokens.dtype

        cond_data = torch.zeros(
            B, self.horizon_10hz, self.action_dim, device=device, dtype=dtype
        )
        trajectory = self.karras_scheduler.sample_inital_position(
            cond_data, generator=generator
        )

        denoise = self._make_edm_denoise(
            self.model, obs_tokens, temporal_cond, "10hz"
        )

        timesteps = torch.arange(
            0, self.karras_scheduler.bins, device=device
        )
        for b, next_b in zip(timesteps[:-1], timesteps[1:]):
            sigma = self.karras_scheduler.timesteps_to_times(b)
            sigma_next = self.karras_scheduler.timesteps_to_times(next_b)
            trajectory = self.karras_scheduler.step(
                denoise, trajectory, sigma, sigma_next, clamp=True
            )

        return trajectory

    # ==================================================================
    # Inference: 50hz multi-step Karras ODE (EDM mode, for Stage-1 ckpt)
    # ==================================================================
    def conditional_sample_50hz(
        self,
        obs_tokens,
        temporal_cond,
        generator=None,
        num_inference_steps=None,
    ):
        """Multi-step Karras Heun ODE for 50hz (EDM inference, no CTM)."""
        B = obs_tokens.shape[0]
        device = obs_tokens.device
        dtype = obs_tokens.dtype

        cond_data = torch.zeros(
            B, self.horizon_50hz, self.action_dim, device=device, dtype=dtype
        )
        # Use ctm_scheduler for noise scale (same sigma range), but do ODE stepping
        trajectory = self.ctm_scheduler.sample_inital_position(
            cond_data, generator=generator
        )

        denoise = self._make_edm_denoise(
            self.model, obs_tokens, temporal_cond, "50hz"
        )

        bins = num_inference_steps if num_inference_steps is not None else self.ctm_scheduler.bins
        timesteps = torch.arange(0, bins, device=device)
        for b, next_b in zip(timesteps[:-1], timesteps[1:]):
            sigma = self.ctm_scheduler.timesteps_to_times(b)
            sigma_next = self.ctm_scheduler.timesteps_to_times(next_b)
            trajectory = self.ctm_scheduler.step(
                denoise, trajectory, sigma, sigma_next, clamp=True
            )

        return trajectory

    # ==================================================================
    # Inference: 50hz single-step CTM (with optional chaining)
    # ==================================================================
    def consistency_sample_50hz(
        self,
        obs_tokens,
        temporal_cond,
        generator=None,
    ):
        """Single-step CTM for 50hz (matches CTM inference)."""
        B = obs_tokens.shape[0]
        device = obs_tokens.device
        dtype = obs_tokens.dtype

        cond_data = torch.zeros(
            B, self.horizon_50hz, self.action_dim, device=device, dtype=dtype
        )
        trajectory = self.ctm_scheduler.sample_inital_position(
            cond_data, generator=generator
        )

        sigma_max = torch.tensor(
            [self.ctm_scheduler.time_max], device=device
        )
        sigma_min = torch.tensor(
            [self.ctm_scheduler.time_min], device=device
        )

        ctm_denoise = self._make_ctm_denoise(
            self.model, obs_tokens, temporal_cond, "50hz"
        )
        out = self.ctm_scheduler.CTM_calc_out(
            ctm_denoise, trajectory, sigma_max, sigma_min, clamp=True
        )

        if not self.chain:
            return out

        # Chaining: re-noise and refine
        for t_val in self.chaining_times[1:]:
            sigma_chain = torch.tensor([float(t_val)], device=device)
            if self.chaining_times[0] == "C":
                sigma_chain = self.ctm_scheduler.timesteps_to_times(sigma_chain)
            noisy = self.ctm_scheduler.add_noise(out, sigma_chain)
            out = self.ctm_scheduler.CTM_calc_out(
                ctm_denoise, noisy, sigma_chain, sigma_min, clamp=True
            )

        return out

    # ==================================================================
    # predict_action
    # ==================================================================
    def _predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        dataset_obs_temporal_downsample_ratio: int = 1,
        extended_obs_dict: Dict[str, torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        use_edm_50hz: bool = False,
    ) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict

        obs_tokens = self.predict_obs_tokens(obs_dict)
        B = obs_tokens.shape[0]
        obs_tokens = obs_tokens.reshape(B, -1, self.n_emb)

        if self.rnn_lowfreq is not None or self.rnn_highfreq is not None:
            assert extended_obs_dict is not None, (
                "extended_obs is required for RNN temporal condition"
            )

        temporal_cond_10hz = self.get_temporal_cond(extended_obs_dict, head="10hz")
        temporal_cond_50hz = self.get_temporal_cond(extended_obs_dict, head="50hz")

        # 10hz: multi-step Karras Heun ODE
        with torch.no_grad():
            nsample_10 = self.conditional_sample_10hz(
                obs_tokens, temporal_cond_10hz
            )

        # 50hz: CTM single-step (default) or EDM multi-step ODE (for Stage-1 ckpt)
        with torch.no_grad():
            if use_edm_50hz:
                nsample_50 = self.conditional_sample_50hz(
                    obs_tokens, temporal_cond_50hz,
                    num_inference_steps=num_inference_steps,
                )
            else:
                nsample_50 = self.consistency_sample_50hz(
                    obs_tokens, temporal_cond_50hz
                )

        # Unnormalize
        action_pred_10hz = self.normalizer["action"].unnormalize(nsample_10)
        action_pred_50hz = self.normalizer["action_stack"].unnormalize(nsample_50)

        # Extract action windows
        action_10hz = action_pred_10hz[
            :, self.action_start_10hz : self.action_start_10hz + self.n_action_steps
        ]
        action_50hz = action_pred_50hz[
            :, self.action_start_50hz : self.horizon_50hz
        ]

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

    @torch.no_grad()
    def predict_action(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_action(*args, **kwargs)

    # ==================================================================
    # Reactive inference (for deployment)
    # ==================================================================
    def sample_noisy_trajectory(
        self,
        B: int = 1,
        generator=None,
        obs_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, str]:
        head = self.inference_head
        horizon = self.horizon_50hz if head == "50hz" else self.horizon_10hz
        cond_data = torch.zeros(
            B, horizon, self.action_dim,
            dtype=torch.float32, device=self.device,
        )
        if head == "50hz":
            noisy = self.ctm_scheduler.sample_inital_position(cond_data, generator)
        else:
            noisy = self.karras_scheduler.sample_inital_position(cond_data, generator)
        noisy = noisy.reshape(B, -1)
        return noisy, head

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

        obs_tokens = obs_tokens_and_noisy_trajectory[
            :, : -target_horizon * self.action_dim
        ]
        noisy_trajectory = obs_tokens_and_noisy_trajectory[
            :, -target_horizon * self.action_dim :
        ]
        obs_tokens = obs_tokens.to(self.device).reshape(B, -1, self.n_emb)
        noisy_trajectory = noisy_trajectory.to(self.device).reshape(
            B, target_horizon, self.action_dim
        )

        max_extended_obs_last_step = max(extended_obs_last_step_list)
        assert extended_obs_dict is not None
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
        noisy_trajectory = noisy_trajectory[:, : temporal_cond.shape[1], :]

        normalizer_key = "action_stack" if head == "50hz" else "action"
        if head == "50hz":
            # CTM single-step
            sigma_max = torch.tensor(
                [self.ctm_scheduler.time_max], device=self.device
            )
            sigma_min = torch.tensor(
                [self.ctm_scheduler.time_min], device=self.device
            )
            ctm_denoise = self._make_ctm_denoise(
                self.model, obs_tokens, temporal_cond, "50hz"
            )
            nsample = self.ctm_scheduler.CTM_calc_out(
                ctm_denoise, noisy_trajectory, sigma_max, sigma_min, clamp=True
            )
        else:
            # 10hz multi-step Karras Heun
            denoise = self._make_edm_denoise(
                self.model, obs_tokens, temporal_cond, "10hz"
            )
            trajectory = noisy_trajectory
            timesteps = torch.arange(
                0, self.karras_scheduler.bins, device=self.device
            )
            for b_idx, next_b in zip(timesteps[:-1], timesteps[1:]):
                sigma = self.karras_scheduler.timesteps_to_times(b_idx)
                sigma_next = self.karras_scheduler.timesteps_to_times(next_b)
                trajectory = self.karras_scheduler.step(
                    denoise, trajectory, sigma, sigma_next, clamp=True
                )
            nsample = trajectory

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
                    "action": action[i : i + 1],
                    "action_pred": action_pred[i : i + 1],
                })
            else:
                valid_len = max(
                    0, min(extended_obs_last_step_list[i] - start, end - start)
                )
                results.append({
                    "action": action[i : i + 1, :valid_len, :],
                    "action_pred": action_pred[
                        i : i + 1, : extended_obs_last_step_list[i], :
                    ],
                })
        return results

    def predict_from_obs_tokens_and_noisy_trajectory(self, *args, **kwargs):
        with autocast(enabled=self.use_amp_for_inference, dtype=self.amp_dtype):
            return self._predict_from_obs_tokens_and_noisy_trajectory(
                *args, **kwargs
            )

    # ==================================================================
    # Optimizer (from FMMultiFreqPolicy)
    # ==================================================================
    def get_optimizer(
        self,
        lr: float,
        weight_decay: float,
        obs_encoder_lr: Optional[float] = None,
        obs_encoder_weight_decay: Optional[float] = None,
        betas: Tuple[float, float] = (0.9, 0.95),
        **kwargs,
    ) -> torch.optim.Optimizer:
        optim_groups = []

        # Transformer model
        optim_groups.extend(
            self.model.get_optim_groups(weight_decay=weight_decay)
        )

        # RNN encoders
        if self.rnn_lowfreq is not None:
            optim_groups += self.rnn_lowfreq.get_optim_groups(
                weight_decay=weight_decay
            )
        if self.rnn_highfreq is not None:
            optim_groups += self.rnn_highfreq.get_optim_groups(
                weight_decay=weight_decay
            )

        # Obs encoder
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

        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        return optimizer
