import math

import pytest
import torch
from torch import nn

from ImplicitRDP.policy.dp_30hz_consistency_policy import DP30HzConsistencyPolicy


class RecordingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_levels = []
        self.stop_levels = []

    def forward(self, sample, timestep, *, cond, temporal_cond, time_cond=None):
        self.noise_levels.append((timestep[0] / 250).exp().item())
        self.stop_levels.append(None if time_cond is None else (time_cond[0] / 250).exp().item())
        return torch.zeros_like(sample)


def make_policy(**kwargs):
    return DP30HzConsistencyPolicy(
        shape_meta={"action": {"shape": [2]}}, obs_encoder=nn.Identity(), model=RecordingBackbone(),
        horizon=2, n_obs_steps=1, n_action_steps=2, action_start_30hz=0, n_emb=2,
        use_rnn_obs_encoder=False, sigma_min=1, sigma_max=3, sigma_data=1,
        rho=1, num_scales=3, **kwargs,
    )


def test_pseudo_huber_uses_one_norm_per_trajectory():
    policy = make_policy(delta=1)
    pred = torch.tensor([[[3.0, 0.0], [0.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]], requires_grad=True)
    loss = policy._pseudo_huber_per_sample(pred, torch.zeros_like(pred))
    torch.testing.assert_close(loss, torch.tensor([math.sqrt(26) - 1, 0.0]))
    loss.sum().backward()
    torch.testing.assert_close(pred.grad, pred.detach() / math.sqrt(26))


@pytest.mark.parametrize("mode, delta, expected", [
    ("legacy", 1, (math.sqrt(82) + math.sqrt(257)) / 2 - 1),
    ("trajectory", 0, 12.5),
    ("trajectory", -1, math.sqrt(25 + 2 * 0.00054 ** 2) - math.sqrt(2) * 0.00054),
])
def test_pseudo_huber_options(mode, delta, expected):
    policy = make_policy(delta=delta, pseudo_huber_mode=mode)
    pred = torch.tensor([[[3.0, 4.0]]])
    assert policy._pseudo_huber_per_sample(pred, torch.zeros_like(pred)).item() == pytest.approx(expected)


@pytest.mark.parametrize("clip", [False, True])
def test_teacher_solves_edm_and_student_uses_stop_time(clip):
    policy = make_policy(clip_sample=clip)
    cond = torch.zeros(1, 1, 2)
    initial = torch.full((1, 2, 2), 8.0)
    teacher_result = policy.conditional_sample(cond, noisy_trajectory=initial)

    # A zero-output backbone makes the preconditioned denoiser c_skip * x.
    def drift(x, sigma):
        estimate = x / ((sigma - 1) ** 2 + 1)
        if clip:
            estimate = min(1, max(-1, estimate))
        return (x - estimate) / sigma

    x = 8.0
    for sigma, next_sigma in [(3, 2), (2, 1)]:
        dt = next_sigma - sigma
        k1 = drift(x, sigma)
        k2 = drift(x + dt * k1, next_sigma)
        x += dt * (k1 + k2) / 2

    torch.testing.assert_close(teacher_result, torch.full_like(initial, x))
    assert policy.model.noise_levels == pytest.approx([3, 2, 2, 1])
    assert policy.model.stop_levels == [None] * 4

    policy.set_training_stage(2)
    student_result = policy.conditional_sample(cond, noisy_trajectory=initial)
    expected_student = 8.0 / 3 + (1 - 1 / 3) * 8 / 5
    torch.testing.assert_close(student_result, torch.full_like(initial, min(1, expected_student) if clip else expected_student))
    assert policy.model.stop_levels[-1] == pytest.approx(1)
    assert len(policy.model.stop_levels) == 5


@pytest.mark.parametrize("stage", [1, 2])
def test_direct_and_cached_initial_noise_agree(stage):
    policy = make_policy(training_stage=stage, initial_noise_scale=0.25, clip_sample=False)
    cond = torch.zeros(1, 1, 2)
    cached = policy.sample_noisy_trajectory(
        generator=torch.Generator().manual_seed(43), return_expanded_noisy_trajectory=True,
    )
    direct = policy.conditional_sample(cond, generator=torch.Generator().manual_seed(43), num_inference_steps=1)
    from_cache = policy.conditional_sample(cond, noisy_trajectory=cached, num_inference_steps=1)
    torch.testing.assert_close(direct, from_cache)
    assert len(policy.model.stop_levels) == (4 if stage == 1 else 2)
