from typing import Optional

import torch

from ImplicitRDP.model.act.act_transformer import ACTTransformer


class ACT50HzTransformer(ACTTransformer):
    """50Hz ACT transformer that never consumes action history.

    Passing action_history=None makes ACTTransformer build the decoder input as
    zeros with length action_chunk_size. This class ignores any accidental
    action_history argument so training and inference stay identical.
    """

    def forward(
        self,
        obs_tokens: torch.Tensor,
        action_history: Optional[torch.Tensor] = None,
        temporal_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return super().forward(
            obs_tokens=obs_tokens,
            action_history=None,
            temporal_cond=temporal_cond,
        )
