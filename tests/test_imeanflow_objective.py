import torch
from torch import nn

from ImplicitRDP.policy.dp_30hz_imeanflow_policy import DP30HziMeanFlowPolicy


class LinearVelocityField(nn.Module):
    def __init__(self):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(2.0))
        self.velocity = nn.Parameter(torch.tensor(4.0))

    def forward(self, sample, t, *, time_cond, cond, temporal_cond):
        h = time_cond[:, None, None]
        u = self.slope * sample + 2 * t[:, None, None] + 3 * h
        v = (self.velocity + h).expand_as(sample)
        return torch.cat([u, v], dim=-1)


def test_imeanflow_objective_gradients_and_one_step_sampler(monkeypatch):
    field = LinearVelocityField()
    policy = DP30HziMeanFlowPolicy(
        shape_meta={"action": {"shape": [1]}}, obs_encoder=nn.Identity(), model=field,
        horizon=1, n_obs_steps=1, n_action_steps=1, use_rnn_obs_encoder=False,
        action_start_30hz=0, norm_eps=1, norm_p=1,
    )
    x = torch.ones(1, 1, 1)
    cond = torch.zeros_like(x)
    monkeypatch.setattr(policy, "_prepare_inputs", lambda batch: (cond, x, None))
    monkeypatch.setattr(policy, "_sample_tr", lambda *args: (torch.tensor([0.75]), torch.tensor([0.25]), None))
    monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.full_like(tensor, 3.0))
    loss, _ = policy.compute_loss({})

    # At z=2.5, the tangent uses v(t,t)=4, so du/dt=2*4+2+3=13.
    # V=8+0.5*13 and v(t,r)=4.5 are both regressed against noise-x=2.
    error_u = 12.5
    error_v = 2.5
    weight_u = 1 / (error_u**2 + 1)
    weight_v = 1 / (error_v**2 + 1)
    torch.testing.assert_close(loss, torch.tensor(error_u**2 * weight_u + error_v**2 * weight_v))
    loss.backward()
    # Both adaptive weights and the JVP contribution are stop-gradient.
    torch.testing.assert_close(field.slope.grad, torch.tensor(2 * error_u * 2.5 * weight_u))
    torch.testing.assert_close(field.velocity.grad, torch.tensor(2 * error_v * weight_v))

    initial = torch.full_like(x, 3.0)
    sample = policy.conditional_sample(cond, noisy_trajectory=initial)
    torch.testing.assert_close(sample, torch.full_like(x, -8.0))  # z - u(z, 1, 0)
