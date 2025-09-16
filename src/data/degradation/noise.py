"""
Copyright (C) 2025 Yukara Ikemiya
"""
import random

import torch

from .base import Degradation


class NoiseAddition(Degradation):
    def __init__(
        self,
        snr_range: tuple = (5.0, 30.0),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.snr_range = snr_range

    def __call__(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        assert x.shape == noise.shape, f"Shapes of input and noise must be the same: {x.shape} != {noise.shape}"
        snr = random.uniform(self.snr_range[0], self.snr_range[1])
        p_x = x.pow(2.0).mean()
        p_n = noise.pow(2.0).mean()
        P_new = p_x / (10 ** (snr / 10))
        scale_n = (P_new / (p_n + 1e-9)).sqrt()

        # noise addition
        x = x + noise * scale_n

        # avoid clipping
        amp = x.abs().max().item()
        if amp > 1.0:
            x = x / amp

        return x
