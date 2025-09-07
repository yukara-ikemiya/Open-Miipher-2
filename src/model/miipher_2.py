"""
Copyright (C) 2025 Yukara Ikemiya

--------------
Miipher-2 model
"""
import typing as tp
from enum import Enum

import torch
from torch import nn


class MiipherMode(Enum):
    """
    Miipher-2 processing modes.

    CLEAN_INPUT: Clean audio is processed by a non-adaptive feature cleaner (WaveFit pretraining).
    NOISY_INPUT: Noisy audio is processed by an adaptive feature cleaner (WaveFit finetuning / Inference).
    """
    CLEAN_INPUT = 'clean_input'      # Clean input waveform
    NOISY_INPUT = 'noisy_input'      # Noisy input waveform


class Miipher2(nn.Module):
    """
    Miipher-2 model consists of a feature cleaner and a WaveFit vocoder.

    Args:
        feature_cleaner (nn.Module): Feature cleaner model (e.g., Google-USM with adapter layers).
        vocoder (nn.Module): Vocoder model (e.g., WaveFit-5).
        mode (MiipherMode): Processing mode. See MiipherMode for details.
    """

    def __init__(
        self,
        feature_cleaner: nn.Module,
        vocoder: nn.Module,
        mode: MiipherMode = MiipherMode.NOISY_INPUT
    ):
        super().__init__()
        self.feature_cleaner = feature_cleaner
        self.vocoder = vocoder
        self._mode = mode

    def eval(self):
        super().eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        self.feature_cleaner.eval()  # Keep the feature cleaner in eval mode
        return self

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spec (torch.Tensor): Input mel-spectrogram, (B, num_frames, feature_dim)
        Returns:
            vocoder_output (List[torch.Tensor]): Output waveform, (B, L')
        """
        # Mel-spectrogram to audio encoder feature
        encoder_only = (self._mode == MiipherMode.CLEAN_INPUT)
        with torch.no_grad():
            feats = self.feature_cleaner(mel_spec, encoder_only=encoder_only)

        # (bs, num_frames, dim) -> (bs, dim, num_frames)
        feats = feats.transpose(1, 2).contiguous()

        # Audio encoder feature to waveform
        vocoder_output = self.vocoder(feats)

        return vocoder_output

    @torch.no_grad()
    def inference(self, input_waveform: torch.Tensor) -> torch.Tensor:
        """
        Inference with waveform input.

        Args:
            input_waveform (torch.Tensor): Input waveform, (B, L)
        Returns:
            decoded_waveform (torch.Tensor): Output waveform, (B, L')
        """
        # Waveform to audio encoder feature
        feats = self.feature_cleaner.forward_waveform(input_waveform, input_type='waveform', encoder_only=False)
        # Audio encoder feature to waveform
        decoded_waveform = self.vocoder(feats, return_only_last=True)[0]  # (B, L')

        return decoded_waveform

    def set_mode(self, mode: tp.Union[MiipherMode, str]):
        mode = MiipherMode(mode) if isinstance(mode, str) else mode
        self._mode = mode

    @property
    def mode(self) -> MiipherMode:
        return self._mode
