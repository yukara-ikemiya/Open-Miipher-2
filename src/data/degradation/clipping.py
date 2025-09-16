"""
Copyright (C) 2025 Yukara Ikemiya
"""
import typing as tp
import random

import torch

from .base import Degradation


class AudioClipping(Degradation):
    def __init__(
        self,
        amp_range: tp.Tuple[float, float] = (1.2, 2.0),
        prob_soft_clip: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.amp_range = amp_range
        self.prob_soft_clip = prob_soft_clip

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        amp_org = x.abs().max().item()
        assert amp_org <= 1.0
        amp = random.uniform(self.amp_range[0], self.amp_range[1])
        scale = amp / (amp_org + 1e-9)

        # clipping
        x = x * scale
        if random.random() < self.prob_soft_clip:
            x = torch.tanh(x)
        else:
            x = x.clamp(-1.0, 1.0)

        # rescale
        x = x / scale

        return x
