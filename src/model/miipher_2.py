"""
Copyright (C) 2025 Yukara Ikemiya

--------------
Miipher-2 model
"""
import typing as tp
from enum import Enum
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from utils.torch_common import exists


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
        mode: str = 'noisy_input',
        # modules for vocoder training
        discriminator: tp.Optional[nn.Module] = None,
        mrstft: tp.Optional[nn.Module] = None,
        loss_lambdas: dict = {},
        # upsampling before vocoder
        upsample_factor: int = 4,
        upsample_mode: str = 'nearest'
    ):
        super().__init__()
        self.feature_cleaner = feature_cleaner
        self.vocoder = vocoder
        self.discriminator = discriminator
        self.mrstft = mrstft
        self.loss_lambdas = loss_lambdas
        self._mode = MiipherMode(mode)
        self.upsample_factor = upsample_factor
        self.upsample_mode = upsample_mode

        # no gradient for feature cleaner
        for param in self.feature_cleaner.parameters():
            param.requires_grad = False

    def eval(self):
        super().eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        self.feature_cleaner.eval()  # Keep the feature cleaner in eval mode
        return self

    def forward(self, mel_spec: torch.Tensor, initial_noise: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spec (torch.Tensor): Input mel-spectrogram, (B, num_frames, feature_dim)
            initial_noise (torch.Tensor): Initial noise for the vocoder, (B, L)
        Returns:
            vocoder_output (List[torch.Tensor]): Output waveform, (B, L')
        """
        # Mel-spectrogram to audio encoder feature
        encoder_only = (self._mode == MiipherMode.CLEAN_INPUT)
        with torch.no_grad():
            feats = self.feature_cleaner(mel_spec, encoder_only=encoder_only)  # (bs, num_frames, dim)

        # Upsample features to match the vocoder's input frame rate (Sec.2.3)
        feats = feats.transpose(1, 2)  # (bs, dim, num_frames)
        feats = F.interpolate(feats, scale_factor=self.upsample_factor, mode=self.upsample_mode)
        feats = feats.transpose(1, 2)  # (bs, num_frames, dim)

        # Audio encoder feature to waveform
        vocoder_output = self.vocoder(initial_noise, feats)

        return vocoder_output

    @torch.no_grad()
    def inference(
        self,
        input_waveform: torch.Tensor,
        initial_noise: tp.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Inference with waveform input.

        Args:
            input_waveform (torch.Tensor): Input waveform, (B, L)
        Returns:
            decoded_waveform (torch.Tensor): Output waveform, (B, L')
        """
        if exists(initial_noise):
            assert initial_noise.shape == input_waveform.shape
        else:
            initial_noise = torch.randn_like(input_waveform)

        # Waveform to audio encoder feature
        feats = self.feature_cleaner.forward_waveform(input_waveform, encoder_only=False)

        # Upsample features to match the vocoder's input frame rate (Sec.2.3)
        feats = feats.transpose(1, 2)  # (bs, dim, num_frames)
        feats = F.interpolate(feats, scale_factor=self.upsample_factor, mode=self.upsample_mode)
        feats = feats.transpose(1, 2)  # (bs, num_frames, dim)

        # Audio encoder feature to waveform
        decoded_waveform = self.vocoder(initial_noise, feats, return_only_last=True)[0]  # (B, L)

        return decoded_waveform

    def set_mode(self, mode: tp.Union[MiipherMode, str]):
        mode = MiipherMode(mode) if isinstance(mode, str) else mode
        self._mode = mode

    @property
    def mode(self) -> MiipherMode:
        return self._mode

    def train_step(
        self,
        target_audio: torch.Tensor,
        input_mel_spec: torch.Tensor,
        train: bool = True
    ) -> dict:
        """
        Training step for the Miipher-2 model.

        Args:
            target_audio (torch.Tensor): Target audio waveform, (B, L)
            input_mel_spec (torch.Tensor): Input mel-spectrogram, (B, num_frames, feature_dim)
            train (bool): Whether in training mode.

        Returns:
            dict: Losses and metrics for the training step.
        """
        assert exists(self.discriminator) and exists(self.mrstft), "Discriminator and MRSTFT must be provided for training."

        self.train() if train else self.eval()

        # Fix the gain of target audio
        # NOTE: Note that the gain of target audio is specified by the WaveFit model
        #       due to the gain normalization.
        scale = self.vocoder.target_gain / (target_audio.abs().max(dim=1, keepdim=True)[0] + 1e-8)
        target_audio = target_audio * scale

        # initial noise
        initial_noise = torch.randn_like(target_audio)

        target_audio = target_audio.unsqueeze(1)  # (bs, 1, L)
        assert target_audio.dim() == 3 and input_mel_spec.dim() == 3
        assert target_audio.size(0) == input_mel_spec.size(0)

        # Forward pass
        preds = self(input_mel_spec, initial_noise)

        # Vocoder losses
        losses = {}
        for pred in preds:
            pred = pred.unsqueeze(1)  # (bs, 1, L)
            losses_i = {}
            losses_i.update(self.mrstft(pred, target_audio))
            losses_i.update(self.discriminator.compute_G_loss(pred, target_audio))
            for k, v in losses_i.items():
                losses[k] = losses.get(k, 0.) + v / len(preds)

        loss = 0.
        for k in self.loss_lambdas.keys():
            losses[k] = losses[k] * self.loss_lambdas[k]
            loss += losses[k]

        # Discriminator loss
        out_real = self.discriminator.compute_D_loss(target_audio, mode='real')
        loss_d_real = out_real.pop('loss')
        losses.update({f"D/{k}_real": v for k, v in out_real.items()})
        # NOTE: Discriminator loss is also computed for all intermediate predictions (Sec.4.2)
        loss_d_fake = 0.
        out_fake = {}
        for pred in preds:
            pred = pred.unsqueeze(1)
            out_fake_ = self.discriminator.compute_D_loss(pred.detach(), mode='fake')
            loss_d_fake += out_fake_.pop('loss') / len(preds)
            for k, v in out_fake_.items():
                out_fake[f"{k}_fake"] = out_fake.get(f"{k}_fake", 0.) + v / len(preds)
        losses.update({f"D/{k}": v for k, v in out_fake.items()})

        loss_d = loss_d_real + loss_d_fake

        output = {'loss': loss}
        output.update({k: v.detach() for k, v in losses.items()})
        output['D/loss_d'] = loss_d
        output['D/loss_d_real'] = loss_d_real.detach()
        output['D/loss_d_fake'] = loss_d_fake.detach()

        return output

    def save_state_dict(self, dir_save: str):
        state_feature_cleaner = self.feature_cleaner.get_state_dict()
        state_vocoder = self.vocoder.state_dict()
        torch.save(state_feature_cleaner, Path(dir_save) / "feature_cleaner.pth")
        torch.save(state_vocoder, Path(dir_save) / "vocoder.pth")

        if exists(self.discriminator):
            state_discriminator = self.discriminator.state_dict()
            torch.save(state_discriminator, Path(dir_save) / "discriminator.pth")

    def load_state_dict(self, dir_load: str):
        state_feature_cleaner = torch.load(Path(dir_load) / "feature_cleaner.pth")
        state_vocoder = torch.load(Path(dir_load) / "vocoder.pth")
        self.feature_cleaner.load_state_dict(state_feature_cleaner)
        self.vocoder.load_state_dict(state_vocoder)

        if exists(self.discriminator):
            state_discriminator = torch.load(Path(dir_load) / "discriminator.pth")
            self.discriminator.load_state_dict(state_discriminator)
