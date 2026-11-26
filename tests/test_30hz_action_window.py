import pytest
import torch
from torch import nn

from ImplicitRDP.model.common.normalizer import SingleFieldLinearNormalizer
from ImplicitRDP.model.diffusion.transformer_for_dp_30hz import TransformerForDP30Hz
from ImplicitRDP.policy.dp_30hz_consistency_policy import DP30HzConsistencyPolicy
from ImplicitRDP.policy.dp_30hz_imeanflow_policy import DP30HziMeanFlowPolicy


class PoseEncoder(nn.Module):
    def forward(self, obs):
        return obs["pose"]


@pytest.fixture(params=[DP30HziMeanFlowPolicy, DP30HzConsistencyPolicy])
def policy(request):
    torch.manual_seed(12)
    shape = {"action": {"shape": [1]}, "extended_obs": {"wrench": {"shape": [1]}}}
    consistency = request.param is DP30HzConsistencyPolicy
    backbone = TransformerForDP30Hz(
        shape, n_obs_steps=1, cond_dim=8, n_emb=8, n_layer=1, n_head=2,
        horizon_30hz=6, out_dim_multiplier=1 if consistency else 2,
    )
    policy = request.param(
        shape, PoseEncoder(), backbone, horizon=6, n_obs_steps=1, n_action_steps=2,
        n_emb=8, rnn_obs_encoder_hidden_dim=4, highfreq_extended_obs_key="wrench",
        action_start_30hz=1, **({"training_stage": 2} if consistency else {}),
    )
    for field in ("pose", "action_stack", "extended_wrench"):
        policy.normalizer[field] = SingleFieldLinearNormalizer.create_identity()
    return policy.eval()


def test_short_history_training_and_prediction(policy):
    obs = {"pose": torch.zeros(1, 1, 8)}
    wrench = {"wrench": torch.tensor([[[1.0], [2.0], [3.0]]])}
    padded = torch.tensor([1., 2., 3., 3., 3., 3.]).view(1, 6, 1)
    expected = policy.rnn_obs_encoder(padded)
    _, _, train_condition = policy._prepare_inputs({
        "obs": obs, "extended_obs": wrench, "action_stack": torch.zeros(1, 6, 1),
    })
    torch.testing.assert_close(train_condition, expected)
    result = policy.predict_action(obs, wrench)
    assert result["action_pred"].shape == (1, 6, 1)
    torch.testing.assert_close(result["action"], result["action_pred"][:, 1:3])


def test_live_update_interval_must_fit_executable_window(policy):
    policy.validate_reactive_window("TCP", update_interval=1, latency=1)
    with pytest.raises(ValueError, match="gripper needs 3 action steps.*window is 2"):
        policy.validate_reactive_window("gripper", update_interval=2, latency=1)

    # The horizon can be the tighter bound even with a larger configured window.
    policy.n_action_steps = 20
    with pytest.raises(ValueError, match="TCP needs 6 action steps.*window is 5"):
        policy.validate_reactive_window("TCP", update_interval=3, latency=3)

    policy.action_horizon, policy.action_start_30hz, policy.n_action_steps = 48, 7, 41
    policy.validate_reactive_window("TCP", update_interval=30, latency=6)
    policy.validate_reactive_window("gripper", update_interval=35, latency=6)


@pytest.mark.parametrize("pad_to_horizon", [False, True])
def test_reactive_batch_matches_individual_history_windows(policy, pad_to_horizon):
    tokens = torch.randn(2, 8)
    requests = torch.cat([tokens, policy.sample_noisy_trajectory(B=2)], dim=-1).split(1)
    lengths = [3, 5]
    force = torch.tensor([[9., -8., 1., 2., 3.], [-4., -3., -2., -1., 0.]]).unsqueeze(-1)

    def predict(inputs, histories, lengths):
        return policy.predict_from_obs_tokens_and_noisy_trajectory(
            inputs, histories, lengths, dataset_obs_temporal_downsample_ratio=1,
            extend_obs_pad_after=pad_to_horizon,
        )

    # Check per-stream histories, a batched dictionary, and one shared sensor stream.
    for histories, streams in [
        ([{"wrench": sample} for sample in force.split(1)], force),
        ({"wrench": force}, force),
        ({"wrench": force[:1]}, force[:1].expand(2, -1, -1)),
    ]:
        batched = predict(requests, histories, lengths)
        for i, length in enumerate(lengths):
            single = predict([requests[i]], {"wrench": streams[i:i+1, -length:]}, [length])[0]
            torch.testing.assert_close(batched[i]["action_pred"], single["action_pred"])
            torch.testing.assert_close(batched[i]["action"], single["action"])
            assert single["action_pred"].shape[1] == (6 if pad_to_horizon else length)
            torch.testing.assert_close(single["action"], single["action_pred"][:, 1:3])
