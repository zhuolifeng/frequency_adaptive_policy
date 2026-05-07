"""
Karras / CTM diffusion schedulers and Huber loss.

Copied from consistency_policy.diffusion to avoid external dependency.
All code is identical to the original consistency-policy implementation.
"""
import torch
import numpy as np
from torch.nn import functional as F
import random
import math


# ==================== HELPERS ====================
@torch.no_grad()
def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


@torch.no_grad()
def reduce_dims(x, target_dims):
    """Reduces dimensions from the end of a tensor until it has target_dims dimensions."""
    dims_to_reduce = x.ndim - target_dims
    if dims_to_reduce < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is greater"
        )
    for _ in range(dims_to_reduce):
        x = x.squeeze(-1)
    return x


# ==================== LOSS ====================
def Huber_Loss(pred, target, delta=0, weights=None):
    """
    Computes pseudo-huber loss of pred and target of shape (batch, time, dim)

    Delta is the boundary between l_1 and l_2 loss. At delta = 0, this is just MSE loss.
    Setting delta = -1 calculates iCT's recommended delta given data size.

    Also supports weighting of loss
    """
    if delta == -1:
        delta = math.sqrt(math.prod(pred.shape[1:])) * 0.00054

    mse = F.mse_loss(pred, target, reduction="none")
    loss = torch.sqrt(mse ** 2 + delta ** 2) - delta

    if weights is not None:
        loss = torch.einsum("b T D, b -> b T D", loss, weights)

    return loss.mean()


# ==================== KARRAS SCHEDULER ====================
class Karras_Scheduler:
    def __init__(
        self,
        time_min,
        time_max,
        rho,
        bins,
        solver,
        time_sampler,
        scaling,
        data_std,
        # log normal sampling
        P_std=1.2,
        P_mean=-1.2,
        weighting="none",
        # euler marayuma corrector step
        alpha=-1,
        friction=-1,
        # time chunking
        beta=0.0,
        name="singular",
        **kwargs,
    ):
        self.time_min = time_min
        self.time_max = time_max
        self.rho = rho
        self.bins = bins
        self.data_std = data_std
        self.solver = solver
        self.time_sampler = time_sampler
        self.scaling = scaling
        self.weighting = weighting
        self.P_std = P_std
        self.P_mean = P_mean
        self.alpha = alpha
        self.friction = friction
        self.beta = beta
        self.bins_min = 0
        self.bins_max = bins
        self.name = name
        print("Using scheduler {}".format(self.name))

        if "corr" in self.solver:
            if self.alpha < 0:
                raise ValueError("alpha must be specified for corr solver")
            if self.friction < 0:
                raise ValueError("friction must be specified for corr solver")

    # ==================== MAIN ====================
    def step(self, model, samples, t, next_t, clamp=False):
        if self.solver == "euler" or self.solver == "first_order":
            return self.euler_solver(model, samples, t, next_t, clamp=clamp)
        elif self.solver == "heun" or self.solver == "second_order":
            return self.heun_solver(model, samples, t, next_t, clamp=clamp)
        else:
            raise ValueError(f"Unknown solver {self.solver}")

    def calc_out(self, model, trajectory, times, clamp=False):
        if self.scaling == "boundary":
            c_skip, c_out, c_in = [
                append_dims(c, trajectory.ndim)
                for c in self.get_scalings_for_boundary_condition(times)
            ]
        elif self.scaling == "no_boundary":
            c_skip, c_out, c_in = [
                append_dims(c, trajectory.ndim)
                for c in self.get_scalings(times)
            ]
        else:
            raise ValueError(f"Unknown scaling {self.scaling}")

        if times.ndim > 1:
            times = reduce_dims(times, 1)

        rescaled_times = 1000 * 0.25 * torch.log(times + 1e-44)
        model_output = model(trajectory * c_in, rescaled_times)

        out = model_output * c_out + trajectory * c_skip
        if clamp:
            out = out.clamp(-1.0, 1.0)
        return out

    def add_noise(self, trajectory, times):
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        return trajectory + self.trajectory_time_product(noise, times)

    def sample_inital_position(self, trajectory, generator):
        traj = torch.randn(
            size=trajectory.shape,
            dtype=trajectory.dtype,
            device=trajectory.device,
            generator=generator,
        )
        return traj

    # ==================== TIME SAMPLERS ====================
    def sample_times(self, trajectory, time_sampler=None):
        time_sampler = time_sampler if time_sampler is not None else self.time_sampler
        batch = trajectory.shape[0]
        device = trajectory.device

        if time_sampler == "uniform":
            return self.uniform_sampler(batch, device)
        elif time_sampler == "log_normal":
            return self.log_normal_sampler(batch, device)
        elif time_sampler == "ctm_dsm":
            return (
                torch.cat(
                    (
                        self.ctm_dsm_sampler(int(math.ceil(batch / 2)), device)[0],
                        self.log_normal_sampler(int(math.floor(batch / 2)), device)[0],
                    ),
                    dim=0,
                ),
                None,
            )
        else:
            raise ValueError(f"Unknown sampler {time_sampler}")

    def uniform_sampler(self, batch, device):
        timesteps = torch.randint(0, self.bins - 1, (batch,), device=device).long()
        return self.timesteps_to_times(timesteps), self.timesteps_to_times(timesteps + 1)

    def log_normal_sampler(self, batch, device):
        sigma = (torch.randn((batch,), device=device) * self.P_std + self.P_mean).exp()
        return sigma, sigma

    def ctm_dsm_sampler(self, batch, device):
        sigma_max = self.time_max
        sigma_min = self.time_min
        ro = self.rho
        xi_samples = torch.rand((batch,), device=device) * 0.7
        transformed_samples = (
            sigma_max ** (1 / ro)
            + xi_samples * (sigma_min ** (1 / ro) - sigma_max ** (1 / ro))
        ) ** ro
        transformed_samples = transformed_samples.clamp(sigma_min, sigma_max)
        return transformed_samples, transformed_samples

    def timesteps_to_times(self, timesteps):
        t = self.time_max ** (1 / self.rho) + timesteps / (self.bins - 1) * (
            self.time_min ** (1 / self.rho) - self.time_max ** (1 / self.rho)
        )
        t = t ** self.rho
        return t.clamp(self.time_min, self.time_max)

    def times_to_timesteps(self, times):
        r = 1 / self.rho
        timesteps = (times ** r - self.time_max ** r) * (self.bins - 1) / (
            self.time_min ** r - self.time_max ** r
        )
        timesteps = torch.round(timesteps)
        return timesteps.long()

    # ==================== WEIGHTINGS ====================
    def get_weights(self, times, next_times, weighting=None):
        weighting = weighting if weighting is not None else self.weighting
        if weighting == "none":
            return None
        elif weighting == "ict":
            return self.get_ict_weightings(times, next_times)
        elif weighting == "karras":
            return self.get_karras_weightings(times)
        else:
            raise ValueError(f"Unknown weighting {weighting}")

    def get_ict_weightings(self, times, next_times):
        return 1 / (times - next_times)

    def get_karras_weightings(self, times, **kwargs):
        return (times ** 2 + self.data_std ** 2) / ((times * self.data_std) ** 2)

    # ==================== PARAMETERIZATIONS ====================
    def get_scalings(self, time):
        c_skip = self.data_std ** 2 / (time ** 2 + self.data_std ** 2)
        c_out = time * self.data_std / ((time ** 2 + self.data_std ** 2) ** 0.5)
        c_in = 1 / (time ** 2 + self.data_std ** 2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, time):
        c_skip = self.data_std ** 2 / (
            (time - self.time_min) ** 2 + self.data_std ** 2
        )
        c_out = (
            (time - self.time_min)
            * self.data_std
            / (time ** 2 + self.data_std ** 2) ** 0.5
        )
        c_in = 1 / (time ** 2 + self.data_std ** 2) ** 0.5
        return c_skip, c_out, c_in

    # ==================== SOLVERS ====================
    @torch.no_grad()
    def euler_solver(self, model, samples, t, next_t, clamp=False):
        dims = samples.ndim
        step = append_dims((next_t - t), dims)
        denoisedy = self.calc_out(model, samples, t, clamp=clamp)
        dy = (samples - denoisedy) / append_dims(t, dims)
        y_next = samples + step * dy
        return y_next

    @torch.no_grad()
    def heun_solver(self, model, samples, t, next_t, clamp=False):
        dims = samples.ndim
        step = append_dims((next_t - t), dims)
        denoisedy = self.calc_out(model, samples, t, clamp=clamp)
        dy = (samples - denoisedy) / append_dims(t, dims)
        y_next = samples + step * dy
        denoisedy_next = self.calc_out(model, y_next, next_t, clamp=clamp)
        dy_next = (y_next - denoisedy_next) / append_dims(next_t, dims)
        y_next = samples + step * (dy + dy_next) / 2
        return y_next

    # ==================== HELPERS ====================
    @staticmethod
    def trajectory_time_product(traj, times):
        return torch.einsum("b T D, b -> b T D ", traj, times)


# ==================== CTM SCHEDULER ====================
class CTM_Scheduler(Karras_Scheduler):
    def __init__(
        self,
        time_min,
        time_max,
        rho,
        bins,
        solver,
        time_sampler,
        scaling,
        data_std,
        P_std=1.2,
        P_mean=-1.2,
        weighting="none",
        alpha=-1,
        friction=-1,
        beta=0.0,
        ode_steps_max=-1,
        name="singular",
        **kwargs,
    ):
        if ode_steps_max == -1:
            raise ValueError("ode_steps_max must be specified for CTM scheduler")

        self.ode_steps_max = ode_steps_max

        super().__init__(
            time_min=time_min,
            time_max=time_max,
            rho=rho,
            bins=bins,
            solver=solver,
            time_sampler=time_sampler,
            scaling=scaling,
            data_std=data_std,
            P_std=P_std,
            P_mean=P_mean,
            weighting=weighting,
            alpha=alpha,
            friction=friction,
            beta=beta,
            name=name,
        )
        print("Using CTM scheduler")

    def CTM_calc_out(self, model, trajectory, times, stops, clamp=False):
        if self.scaling == "boundary":
            c_skip, c_out, c_in = [
                append_dims(c, trajectory.ndim)
                for c in self.get_scalings_for_boundary_condition(times)
            ]
        elif self.scaling == "no_boundary":
            c_skip, c_out, c_in = [
                append_dims(c, trajectory.ndim)
                for c in self.get_scalings(times)
            ]
        else:
            raise ValueError(f"Unknown scaling {self.scaling}")

        if times.ndim > 1:
            times = reduce_dims(times, 1)
        if stops.ndim > 1:
            stops = reduce_dims(stops, 1)

        rescaled_times = (1000 * 0.25 * torch.log(times + 1e-44)).expand(
            trajectory.shape[0]
        )
        rescaled_stops = (1000 * 0.25 * torch.log(stops + 1e-44)).expand(
            trajectory.shape[0]
        )

        model_output = model(trajectory * c_in, rescaled_times, rescaled_stops)

        out = model_output * c_out + trajectory * c_skip  # g_theta
        ratio = (stops / times).unsqueeze(-1).unsqueeze(-1).expand(*out.shape)
        out = trajectory * ratio + out * (1 - ratio)  # G_theta

        if clamp:
            out = out.clamp(-1.0, 1.0)
        return out

    def sample_times(self, trajectory, time_sampler=None):
        time_sampler = time_sampler if time_sampler is not None else self.time_sampler
        batch = trajectory.shape[0]
        device = trajectory.device

        if time_sampler == "ctm":
            t = torch.randint(0, self.bins, (batch,), device=device).long()
            s = torch.cat(
                [torch.randint(int(t_i.item()), self.bins + 1, (1,)) for t_i in t]
            ).to(device)
            u = torch.cat(
                [
                    torch.randint(int(t_i.item()), int((s_i + 1).item()), (1,))
                    for t_i, s_i in zip(t, s)
                ]
            ).to(device)
            maxes = t + self.ode_steps_max
            mask = (u > maxes).float()
            u = u * (1 - mask) + maxes * mask
            return t, s, u

        else:
            return super().sample_times(trajectory, time_sampler=time_sampler)

    # CTM needs to handle step size 0
    @torch.no_grad()
    def heun_solver(self, model, samples, t, next_t, clamp=False):
        dims = samples.ndim
        step = append_dims((next_t - t), dims)
        mask = (step == 0).float()

        denoisedy = self.calc_out(model, samples, t, clamp=clamp)
        dy = (samples - denoisedy) / (append_dims(t, dims) + mask)
        y_next = samples + step * dy

        denoisedy_next = self.calc_out(model, y_next, next_t, clamp=clamp)
        dy_next = (y_next - denoisedy_next) / (append_dims(next_t, dims) + mask)
        y_next = samples + step * (dy + dy_next) / 2
        y_next = y_next * (1 - mask) + samples * mask

        return y_next
