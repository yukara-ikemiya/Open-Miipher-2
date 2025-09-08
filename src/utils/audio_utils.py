"""
Copyright (C) 2024 Yukara Ikemiya
"""

import math

import torch
from torch import nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel

EPS = 1e-10


def get_amplitude_spec(x, n_fft, win_size, hop_size, window, return_power: bool = False):
    stft_spec = torch.stft(
        x, n_fft, hop_length=hop_size, win_length=win_size, window=window,
        center=True, normalized=False, onesided=True, return_complex=True)

    power_spec = torch.view_as_real(stft_spec).pow(2).sum(-1)

    return power_spec if return_power else torch.sqrt(power_spec + EPS)


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        sr: int,
        # STFT setting
        n_fft: int, win_size: int, hop_size: int,
        # MelSpec setting
        n_mels: int, fmin: float, fmax: float,
    ):
        super().__init__()

        self.sr = sr
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        mel_basis = librosa_mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        mel_inv_basis = torch.linalg.pinv(mel_basis)

        self.register_buffer('fft_win', torch.hann_window(win_size))
        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('mel_inv_basis', mel_inv_basis)

    def compute_mel(self, x: torch.Tensor):
        """
        Compute Mel-spectrogram.

        Args:
            x: time_signal, (bs, length)
        Returns:
            mel_spec: Mel spectrogram, (bs, n_mels, num_frame)
        """
        assert x.dim() == 2
        L = x.shape[-1]
        # NOTE : To prevent different signal length in the final frame of the STFT between training and inference time,
        #        input signal length must be a multiple of hop_size.
        assert L % self.hop_size == 0, f"Input signal length must be a multiple of hop_size {self.hop_size}." + \
            f"Input shape -> {x.shape}"

        num_frame = L // self.hop_size

        # STFT
        stft_spec = get_amplitude_spec(x, self.n_fft, self.win_size, self.hop_size, self.fft_win)

        # Mel Spec
        mel_spec = torch.matmul(self.mel_basis, stft_spec)

        # NOTE : The last frame is removed here.
        #   When using center=True setting, output from torch.stft has frame length of (L//hopsize+1).
        #   For training WaveGrad-based architecture, the frame length must be (L//hopsize).
        #   There might be a better way, but I believe this has little to no impact on training
        #   since the whole signal information is contained in the previous frames even when removing the last one.
        mel_spec = mel_spec[..., :num_frame]

        return mel_spec
