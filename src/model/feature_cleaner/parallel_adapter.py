"""
Copyright (C) 2025 Yukara Ikemiya

Adapted from the following repo's code under Apache-2.0 License.
https://github.com/jxhe/unify-parameter-efficient-tuning/

-----------------------------------------------------
Parallel adapter for LLMs.
"""
import math
import typing as tp

import torch
import torch.nn as nn


def init_bert_weights(module):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # std defaults to 0.02, this might need to be changed
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class AdapterLayer(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_bottleneck: int,
        dropout: float = 0.0,
        init_option: str = "bert",
        adapter_scalar: tp.Union[float, str] = 1.0,
        pre_ln_class=None
    ):
        super().__init__()

        self.n_embd = dim_in
        self.down_size = dim_bottleneck

        # Layer normalization options
        self.use_pre_ln = pre_ln_class is not None
        if self.use_pre_ln:
            self.pre_ln = pre_ln_class(dim_in)

        # PA modules
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.scale = nn.Parameter(torch.ones(1)) if adapter_scalar == "learnable_scalar" else float(adapter_scalar)
        self.dropout = dropout

        # Initialization options
        if init_option == "bert":
            self.apply(init_bert_weights)
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)
        else:
            raise ValueError(f"Unknown initialization option: {init_option}")

    def forward(self, x):
        if self.use_pre_ln:
            x = self.pre_ln(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * self.scale

        return up
