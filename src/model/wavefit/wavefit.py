"""
Copyright (C) 2025 Yukara Ikemiya

-----------------
WaveFit module in Miipher-2.
"""

import torch
import torch.nn as nn

from transformers.models.gemma3n.modeling_gemma3n import Gemma3nAudioConformerBlock

from .generator import WaveFitGenerator
from utils.torch_common import checkpoint


class ConformerConfig:
    def __init__(
        self,
        conf_attention_chunk_size: int = 12,
        conf_attention_context_left: int = 13,
        conf_attention_context_right: int = 0,
        conf_attention_logit_cap: float = 50.0,
        conf_conv_kernel_size: int = 5,
        conf_num_attention_heads: int = 8,
        conf_reduction_factor: int = 4,
        conf_residual_weight: float = 0.5,
        gradient_clipping: float = 10000000000.0,
        hidden_size: int = 1536,
        rms_norm_eps: float = 1e-06
    ):
        self.conf_attention_chunk_size = conf_attention_chunk_size
        self.conf_attention_context_left = conf_attention_context_left
        self.conf_attention_context_right = conf_attention_context_right
        self.conf_attention_logit_cap = conf_attention_logit_cap
        self.conf_conv_kernel_size = conf_conv_kernel_size
        self.conf_num_attention_heads = conf_num_attention_heads
        self.conf_reduction_factor = conf_reduction_factor
        self.conf_residual_weight = conf_residual_weight
        self.gradient_clipping = gradient_clipping
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps


class WaveFit(nn.Module):
    def __init__(
        self,
        num_iteration: int,
        target_gain: float = 0.9,
        # Pre-network Conformer blocks (Sec.3.2)
        num_conformer_blocks: int = 4,
        args_conformer: dict = {
            "conf_attention_chunk_size": 12,
            "conf_attention_context_left": 13,
            "conf_attention_context_right": 0,
            "conf_attention_logit_cap": 50.0,
            "conf_conv_kernel_size": 5,
            "conf_num_attention_heads": 8,
            "conf_reduction_factor": 4,
            "conf_residual_weight": 0.5,
            "gradient_clipping": 10000000000.0,
            "hidden_size": 1536,
            "rms_norm_eps": 1e-06
        },
        # WaveFit generator
        args_generator: dict = {
            "dim_feat": 1536,
            "upsample_factors": [5, 4, 3, 2, 2],
            "upsample_channels": [512, 512, 256, 128, 128],
            "downsample_channels": [128, 128, 256, 512],
        }
    ):
        super().__init__()

        self.T = num_iteration
        self.target_gain = target_gain

        # Conformer blocks
        self.conformer_config = ConformerConfig(**args_conformer)
        self.conformer_blocks = nn.ModuleList(
            [Gemma3nAudioConformerBlock(self.conformer_config) for _ in range(num_conformer_blocks)]
        )

        # Generator
        self.generator = WaveFitGenerator(num_iteration, **args_generator)
        self.EPS = 1e-8

    def forward(
        self,
        initial_noise: torch.Tensor,
        audio_feats: torch.Tensor,
        # You can use this option at inference time
        return_only_last: bool = False,
        # training config
        gradient_checkpointing: bool = False
    ):
        """
        Args:
            initial_noise: Initial noise, (bs, L).
            audio_feats: Audio features, (bs, n_frame, dim).
            return_only_last: If true, only the last output (y_0) is returned.
        Returns:
            preds: List of predictions (y_t)
        """
        initial_noise = initial_noise.unsqueeze(1)  # (bs, 1, L)
        assert initial_noise.dim() == audio_feats.dim() == 3
        assert initial_noise.size(0) == audio_feats.size(0)

        # Pre-network
        mask = torch.zeros(audio_feats.size(0), audio_feats.size(1), dtype=torch.bool, device=audio_feats.device)
        for block in self.conformer_blocks:
            if gradient_checkpointing and self.training:
                audio_feats = checkpoint(block, audio_feats, mask)
            else:
                audio_feats = block(audio_feats, mask)

        # (bs, n_frame, dim) -> (bs, dim, n_frame)
        audio_feats = audio_feats.transpose(1, 2).contiguous()

        preds = []
        y_t = initial_noise
        for t in range(self.T):
            # estimate noise
            if gradient_checkpointing and self.training:
                est = checkpoint(self.generator, y_t, audio_feats, t)
            else:
                est = self.generator(y_t, audio_feats, t)

            y_t = y_t - est

            # gain normalization (Sec.2.3 in the Miipher paper)
            y_t = self.normalize_gain(y_t)

            if (not return_only_last) or (t == self.T - 1):
                preds.append(y_t.squeeze(1))

            # To avoid gradient loop
            y_t = y_t.detach()

        return preds

    def normalize_gain(self, z_t: torch.Tensor):
        # z_t: (bs, 1, L)
        scale = self.target_gain / (z_t.squeeze(1).abs().max(dim=1, keepdim=True)[0][:, :, None] + self.EPS)
        return z_t * scale
