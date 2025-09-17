"""
Copyright (C) 2025 Yukara Ikemiya
"""
import random

import torch
import torchaudio

from .base import Degradation


class AudioLowpass(Degradation):
    def __init__(
        self,
        cutoff_range: tuple = (2000.0, 7000.0),
        sample_rate: int = 16000
    ):
        super().__init__(sample_rate=sample_rate)
        self.cutoff_range = cutoff_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        cutoff = random.uniform(self.cutoff_range[0], self.cutoff_range[1])
        x = torchaudio.functional.lowpass_biquad(x, self.sample_rate, cutoff)

        # avoid clipping
        amp = x.abs().max().item()
        if amp > 1.0:
            x = x / (amp + 1e-8)

        return x
