"""
Copyright (C) 2025 Yukara Ikemiya

-----------------------------------------------------
A base class of audio encoder adapters.
"""
from abc import ABC, abstractmethod

import torch
from torch import nn


class AudioEncoderAdapter(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, encoder_only: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def save_state_dict(self, path: str):
        pass

    @abstractmethod
    def load_state_dict(self, path: str):
        pass

    @abstractmethod
    def get_state_dict(self) -> dict:
        pass

    def train_step(
        self,
        x_tgt: torch.Tensor,
        x_deg: torch.Tensor,
        loss_lambda: dict = {
            'l1': 1.0,
            'l2': 1.0,
            'spectral_convergence': 1.0
        },
        train: bool = True
    ) -> dict:
        """
        Loss computation for feature cleaners defined in the Miipher paper.
        https://arxiv.org/abs/2303.01664
        """
        self.train() if train else self.eval()
        assert x_tgt.shape == x_deg.shape, f"Shapes of target and degraded features must be the same: {x_tgt.shape} != {x_deg.shape}"

        with torch.no_grad():
            feats_tgt = self.forward(x_tgt, encoder_only=True)

        with torch.set_grad_enabled(train):
            feats_deg = self.forward(x_deg, encoder_only=False)

        # loss
        l1_loss = (feats_deg - feats_tgt).abs().mean() * loss_lambda['l1']
        l2_loss = (feats_deg - feats_tgt).pow(2.0).mean() * loss_lambda['l2']
        sc_loss = l2_loss / (feats_tgt.pow(2.0).mean() + 1e-9) * loss_lambda['spectral_convergence']

        loss = l1_loss + l2_loss + sc_loss

        return {
            'loss': loss,
            'l1_loss': l1_loss.detach(),
            'l2_loss': l2_loss.detach(),
            'sc_loss': sc_loss.detach()
        }
