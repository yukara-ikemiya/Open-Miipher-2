"""
Copyright (C) 2024 Yukara Ikemiya
"""

import random

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

    def __init__(self, min_db: float = -29., max_db: float = -19.):
        super().__init__()
        self.min_db = min_db
        self.max_db = max_db

    def __call__(self, x: torch.Tensor):
        assert x.ndim <= 2
        current_power = (x ** 2).mean()
        target_db = random.uniform(self.min_db, self.max_db)
        target_power = 10 ** (target_db / 10)
        gain = (target_power / current_power).sqrt()
        x = x * gain

        # Amplitude must be in [-1, 1] range
        max_amp = torch.abs(x).max()
        if max_amp > 1.0:
            x = x / max_amp

        return x
