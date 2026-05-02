from typing import Optional, Union, Callable
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

VISUALIZE_ATTENTION = False

class CustomizedTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                    activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first,
                    norm_first=norm_first, bias=bias, device=device, dtype=dtype)

    @staticmethod
    def _has_forward_ad_tensor(*tensors: Tensor) -> bool:
        return any(
            torch.autograd.forward_ad.unpack_dual(tensor).tangent is not None
            for tensor in tensors
        )

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        need_weights = self._has_forward_ad_tensor(x)
        x, _ = self.self_attn(x, x, x,
                              attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask,
                              is_causal=is_causal,
                              need_weights=need_weights)
        return self.dropout1(x)

    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        need_weights = VISUALIZE_ATTENTION or self._has_forward_ad_tensor(x, mem)
        x, weights = self.multihead_attn(x, mem, mem,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            is_causal=is_causal,
                            need_weights=need_weights)
        if VISUALIZE_ATTENTION:
            self.last_attn_weights = weights
        return self.dropout2(x)

    def forward(self, tgt: Tensor,
                memory: Tensor,
                temporal_memory: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                temporal_memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                temporal_memory_key_padding_mask: Optional[Tensor] = None,
                tgt_is_causal: bool = False,
                memory_is_causal: bool = False,
                temporal_memory_is_causal: bool = False
                ) -> Tensor:

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x
