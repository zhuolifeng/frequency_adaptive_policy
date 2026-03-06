"""
Modified from HunyuanDiT https://github.com/Tencent/HunyuanDiT
"""
from typing import Union, Optional
import logging
import torch
import torch.nn as nn
from ImplicitRDP.model.diffusion.positional_embedding import SinusoidalPosEmb
from ImplicitRDP.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class ControlNetForActionDiffusion(ModuleAttrMixin):
    def __init__(self,
                 shape_meta: dict,
                 action_horizon: int,
                 n_layer: int = 7,
                 n_head: int = 8,
                 n_emb: int = 768,
                 max_cond_tokens: int=800,
                 p_drop_attn: float = 0.1,
                 ) -> None:
        super().__init__()

        input_dim = shape_meta['input_dim']
        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.randn((1, action_horizon, n_emb)))
        self.time_emb = SinusoidalPosEmb(n_emb)
        # learnable position embedding
        self.cond_pos_emb = nn.Parameter(torch.randn((1, max_cond_tokens, n_emb)))

        # decoder, the same as transformer for action diffusion
        self.decoder_blocks = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True  # important for stability
                )
                for _ in range(n_layer - 1) # first layer is before_proj
            ]
        )
        self.n_layer = n_layer

        # init
        self.apply(self._init_weights)

        # Input zero linear for the first block
        self.before_proj = zero_module(nn.Linear(n_emb, n_emb))

        # Output zero linear for the every block
        self.after_proj_list = nn.ModuleList(
            [
                zero_module(nn.Linear(n_emb, n_emb))
                for _ in range(n_layer)
            ]
        )

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (nn.Dropout,
                        SinusoidalPosEmb,
                        nn.TransformerEncoderLayer,
                        nn.TransformerDecoderLayer,
                        nn.TransformerEncoder,
                        nn.TransformerDecoder,
                        nn.ModuleList,
                        nn.Mish,
                        nn.Sequential,
                        nn.Embedding)
        if isinstance(module, (nn.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, ControlNetForActionDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                cond: Optional[torch.Tensor] = None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,N,n_emb)
        output: (B,T,input_dim)
        """

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)

        # 2. process conditions
        cond_emb = torch.cat([cond, time_emb], dim=1)
        tc = cond_emb.shape[1]
        cond_pos_emb = self.cond_pos_emb[
                       :, :tc, :
                       ]  # each position maps to a (learnable) vector
        cond_emb = cond_emb + cond_pos_emb

        # 3. process input
        input_emb = self.input_emb(sample)
        t = input_emb.shape[1]
        pos_emb = self.pos_emb[
                  :, :t, :
                  ]  # each position maps to a (learnable) vector
        input_emb = input_emb + pos_emb

        # 4. transformer
        controls = []
        tgt = self.before_proj(input_emb)
        memory = cond_emb
        controls.append(tgt)
        for layer, block in enumerate(self.decoder_blocks):
            tgt = block(tgt,
                      memory,
                      tgt_mask=None,
                      memory_mask=None,
                      tgt_key_padding_mask=None,
                      memory_key_padding_mask=None)
            controls.append(self.after_proj_list[layer](tgt)) # zero linear for output

        return controls