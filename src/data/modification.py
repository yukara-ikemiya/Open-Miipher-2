"""
Copyright (C) 2024 Yukara Ikemiya
"""

import random
import math

import torch
from torch import nn


# Channels

class Mono(nn.Module):
    def __call__(self, x: torch.Tensor):
        assert len(x.shape) <= 2
        return torch.mean(x, dim=0, keepdims=True) if len(x.shape) > 1 else x


class Stereo(nn.Module):
    def __call__(self, x: torch.Tensor):
        x_shape = x.shape
        assert len(x_shape) <= 2
        # Check if it's mono
        if len(x_shape) == 1:  # s -> 2, s
            x = x.unsqueeze(0).repeat(2, 1)
        elif len(x_shape) == 2:
            if x_shape[0] == 1:  # 1, s -> 2, s
                x = x.repeat(2, 1)
            elif x_shape[0] > 2:  # ?, s -> 2,s
                x = x[:2, :]

        return x


# Augmentation

class PhaseFlipper(nn.Module):
    """Randomly invert the phase of a signal"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, x: torch.Tensor):
        assert len(x.shape) <= 2
        return -x if (random.random() < self.p) else x


class VolumeChanger(nn.Module):
    """Randomly change volume (amplitude) of a signal"""

    def __init__(self, min_amp: float = 0.25, max_amp: float = 1.0):
        super().__init__()
        self.min_amp = min_amp
        self.max_amp = max_amp

    def __call__(self, x: torch.Tensor):
        assert x.ndim <= 2
        amp_x = x.abs().max().item()
        if amp_x < 1e-5:
            return x

        min_db = 20 * math.log10(self.min_amp / amp_x)
        max_db = 20 * math.log10(self.max_amp / amp_x)
        scale_db = random.uniform(min_db, max_db)
        scale = 10 ** (scale_db / 20)
        x = x * scale

        return x
