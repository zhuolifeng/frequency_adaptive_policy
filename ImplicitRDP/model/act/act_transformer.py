from typing import Optional, Tuple
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from ImplicitRDP.model.diffusion.positional_embedding import SinusoidalPosEmb
from ImplicitRDP.model.diffusion.transformer_utils import CustomizedTransformerDecoderLayer
from ImplicitRDP.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)


class ACTTransformer(ModuleAttrMixin):
    def __init__(self,
                 shape_meta: dict,
                 action_chunk_size: int = 8,
                 n_layer: int = 6,
                 n_head: int = 8,
                 n_emb: int = 768,
                 max_cond_tokens: int = 800,
                 max_action_history: int = 32,
                 p_drop_attn: float = 0.1,
                 use_second_cross_attention: bool = False,
                 ) -> None:
        super().__init__()
        
        action_dim = shape_meta['action']['shape'][0]
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.max_action_history = max_action_history
        self.n_emb = n_emb
        
        self.action_emb = nn.Linear(action_dim, n_emb)
        self.action_pos_emb = nn.Parameter(torch.zeros(1, max_action_history + action_chunk_size, n_emb))
        
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, max_cond_tokens, n_emb))
        
        self.decoder_blocks = nn.ModuleList([
            CustomizedTransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True,
                use_second_cross_attention=use_second_cross_attention
            )
            for _ in range(n_layer)
        ])
        self.use_second_cross_attention = use_second_cross_attention
        
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, action_dim)
        
        self.apply(self._init_weights)
        logger.info(
            "ACT Transformer initialized with %e parameters",
            sum(p.numel() for p in self.parameters())
        )
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, ACTTransformer):
            torch.nn.init.normal_(module.action_pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
    
    def _forward(self,
                       obs_tokens: torch.Tensor,
                       action_history: Optional[torch.Tensor] = None,
                       temporal_cond: Optional[torch.Tensor] = None,
                       ) -> torch.Tensor:
        B = obs_tokens.shape[0]
        device = obs_tokens.device
        
        cond_tokens = obs_tokens
        
        if temporal_cond is not None:
            cond_tokens = torch.cat([obs_tokens, temporal_cond], dim=1)
        
        N_cond = cond_tokens.shape[1]
        cond_pos_emb = self.cond_pos_emb[:, :N_cond, :]
        cond_tokens = cond_tokens + cond_pos_emb
        
        if action_history is not None:
            T_hist = action_history.shape[1]
            if T_hist > self.max_action_history:
                action_history = action_history[:, -self.max_action_history:, :]
                T_hist = self.max_action_history
        else:
            T_hist = 0
        
        if T_hist > 0:
            action_seq = torch.cat([
                action_history,
                torch.zeros(B, self.action_chunk_size, self.action_dim, device=device)
            ], dim=1)
        else:
            action_seq = torch.zeros(B, self.action_chunk_size, self.action_dim, device=device)
        
        action_emb = self.action_emb(action_seq)
        T_action = action_emb.shape[1]
        action_pos_emb = self.action_pos_emb[:, :T_action, :]
        action_emb = action_emb + action_pos_emb
        
        causal_mask = self._generate_causal_mask(T_action, device)
        
        tgt = action_emb
        memory = cond_tokens
        
        for block in self.decoder_blocks:
            tgt = block(
                tgt=tgt,
                memory=memory,
                tgt_mask=causal_mask,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None
            )
        
        tgt = self.ln_f(tgt)
        if T_hist > 0:
            actions = self.head(tgt[:, T_hist:, :])
        else:
            actions = self.head(tgt)
        
        return actions
    
    def forward(self,
                obs_tokens: torch.Tensor,  # (B, N_obs, n_emb)
                action_history: Optional[torch.Tensor] = None,  # (B, T_hist, action_dim)
                temporal_cond: Optional[torch.Tensor] = None,  # (B, T_temp, cond_dim)
                ) -> torch.Tensor:
        action_chunk = self._forward(obs_tokens, action_history, temporal_cond)
        return action_chunk
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask
    
    def get_optim_groups(self, weight_decay: float = 1e-3):
        optim_groups = [
            {
                "params": [p for n, p in self.named_parameters() if p.requires_grad],
                "weight_decay": weight_decay,
            },
        ]
        return optim_groups

