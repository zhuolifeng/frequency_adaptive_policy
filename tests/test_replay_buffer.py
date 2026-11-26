import numpy as np
import pytest
import zarr

from ImplicitRDP.common.replay_buffer import ReplayBuffer


@pytest.mark.parametrize("create", [ReplayBuffer.create_empty_numpy, ReplayBuffer.create_empty_zarr])
def test_episode_round_trip(create, tmp_path):
    buffer = create()
    first_obs = np.zeros((6, 3), dtype=np.float32)
    second_obs = np.ones((4, 3), dtype=np.float32)
    buffer.add_episode({"obs": first_obs})
    buffer.add_episode({"obs": second_obs, "action": np.ones((4, 2))})
    if buffer.backend == "zarr":
        buffer.set_chunks({"obs": (4, 3), "action": (4, 2)})
        assert buffer.data["obs"].chunks == (4, 3)

    path = str(tmp_path / "episodes.zarr")
    buffer.save_to_path(path)
    restored = ReplayBuffer.copy_from_path(path, store=zarr.MemoryStore())
    np.testing.assert_array_equal(restored.episode_ends[:], [6, 10])
    np.testing.assert_array_equal(restored.pop_episode()["obs"], second_obs)
    np.testing.assert_array_equal(restored.get_episode(0)["obs"], first_obs)
    assert restored.n_episodes == 1
