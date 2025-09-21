"""
Copyright (C) 2025 Yukara Ikemiya
"""
import typing as tp
import random

import torch
import numpy as np
from scipy.signal import butter, filtfilt, fftconvolve
import pyroomacoustics as pra

from .base import Degradation


class RIRReverb(Degradation):
    def __init__(
        self,
        sample_rate: int = 16000,
        rt60_range: tuple = (0.2, 0.5),
        xyz_range: tuple = ((2., 10.), (2., 10.), (2., 5.)),
        mic_margin: float = 0.5,
        src_margin: float = 0.1,
        # cutoff freq for RIR filter (highpass)
        hp_cutoff_hz: float = 20.0
    ):
        super().__init__(sample_rate=sample_rate)
        self.rt60_range = rt60_range
        self.xyz_range = xyz_range
        self.mic_margin = mic_margin
        self.src_margin = src_margin
        self.hp_cutoff_hz = hp_cutoff_hz

        # pre-compute highpass filter
        nyq = sample_rate / 2.
        norm_cutoff = self.hp_cutoff_hz / nyq
        self.hp_b, self.hp_a = butter(4, norm_cutoff, btype="high")

    def _sample_room_params(self):
        rt60 = random.uniform(self.rt60_range[0], self.rt60_range[1])
        Lx = random.uniform(self.xyz_range[0][0], self.xyz_range[0][1])
        Ly = random.uniform(self.xyz_range[1][0], self.xyz_range[1][1])
        Lz = random.uniform(self.xyz_range[2][0], self.xyz_range[2][1])
        mic_pos = [
            random.uniform(self.mic_margin, Lx - self.mic_margin),
            random.uniform(self.mic_margin, Ly - self.mic_margin),
            random.uniform(1.2, min(2.0, Lz - self.mic_margin))
        ]
        src_pos = [
            random.uniform(self.src_margin, Lx - self.src_margin),
            random.uniform(self.src_margin, Ly - self.src_margin),
            random.uniform(self.src_margin, Lz - self.src_margin)
        ]

        return rt60, [Lx, Ly, Lz], mic_pos, src_pos

    def _generate_rir(
        self, rt60: float, room_size: tp.List[float],
        mic_pos: tp.List[float], src_pos: tp.List[float]
    ):
        e_absorption, max_order = pra.inverse_sabine(rt60, room_size)
        room = pra.ShoeBox(
            room_size,
            fs=self.sample_rate,
            materials=pra.Material(e_absorption),
            max_order=max_order,
        )
        room.add_microphone_array(
            pra.MicrophoneArray(np.array(mic_pos).reshape(3, 1), self.sample_rate)
        )
        room.add_source(src_pos)
        room.compute_rir()

        rir = room.rir[0][0]
        L_rir = len(rir)

        # highpass to RIR
        rir_hp = filtfilt(self.hp_b, self.hp_a, rir)
        rir_hp = rir_hp[:L_rir]  # trim to original length

        # scale direct sound to 0 dB
        peak_idx = np.argmax(np.abs(rir_hp))
        if np.abs(rir_hp[peak_idx]) > 1e-9:
            rir_hp /= rir_hp[peak_idx]

        return rir_hp, peak_idx

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 1, f"Input audio must be 1D tensor: {x.shape}"
        dtype = x.dtype
        L = len(x)

        # sample room parameters
        rt60, room_size, mic_pos, src_pos = self._sample_room_params()

        # generate RIR
        rir, peak_idx = self._generate_rir(rt60, room_size, mic_pos, src_pos)

        # convolve
        x = fftconvolve(x.numpy(), rir, mode='full')
        x = torch.from_numpy(x).to(dtype)

        # fix signal shift and length
        offset = min(peak_idx, len(x) - L)
        x = x[offset:offset + L]

        # avoid clipping
        amp = x.abs().max().item()
        if amp > 1.0:
            x = x / (amp + 1e-8)

        return x
