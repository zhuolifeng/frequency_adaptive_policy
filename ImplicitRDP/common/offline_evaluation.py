"""Load offline evaluation samples one frame at a time."""

import torch

from ImplicitRDP.common.pytorch_util import dict_apply


class EpisodeBatches:
    def __init__(self, dataset, indices, device):
        self.dataset = dataset
        self.indices = indices
        self.device = device

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, frame):
        sample = self.dataset[self.indices[frame]]
        return dict_apply(
            sample,
            lambda x: x.unsqueeze(0).to(self.device) if isinstance(x, torch.Tensor) else x,
        )


def group_episode_batches(dataset, device):
    indices = [[] for _ in range(dataset.replay_buffer.n_episodes)]
    for i, episode in enumerate(dataset.sampler.indices_to_episode_idx):
        indices[episode].append(i)
    return [EpisodeBatches(dataset, episode, device) for episode in indices]
