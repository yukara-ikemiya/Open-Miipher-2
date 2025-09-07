"""
Copyright (C) 2025 Yukara Ikemiya

Adapted from the following repo's code under Apache License 2.0.
https://github.com/lmnt-com/wavegrad/
"""

import typing as tp
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        nn.init.zeros_(self.bias)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_iter: int, use_conv: bool = True):
        super().__init__()
        self.dim = dim
        self.max_iter = max_iter
        self.use_conv = use_conv
        assert dim % 2 == 0

        if use_conv:
            # 1x1 conv
            self.conv = nn.Conv1d(dim, dim, 1)
            nn.init.xavier_uniform_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)

        # pre-compute positional embedding
        pos_embs = self.prepare_embedding()  # (max_iter, dim)
        self.register_buffer('pos_embs', pos_embs)

    def forward(self, x, t: int):
        """
        Args:
          x: (bs, dim, T)
          t: Step index

        Returns:
          x_with_pos: (bs, dim, T)
        """
        assert 0 <= t < self.max_iter, f"Invalid step index {t}. It must be 0 <= t < {self.max_iter} = max_iter."
        pos_emb = self.pos_embs[t][None, :, None]
        if self.use_conv:
            pos_emb = self.conv(pos_emb)

        return x + pos_emb

    def prepare_embedding(self, scale: float = 5000.):
        dim_h = self.dim // 2
        pos = torch.linspace(0., scale, self.max_iter)
        div_term = torch.exp(- math.log(10000.0) * torch.arange(dim_h) / dim_h)
        pos = pos[:, None] @ div_term[None, :]  # (max_iter, dim_h)
        pos_embs = torch.cat([torch.sin(pos), torch.cos(pos)], dim=-1)  # (max_iter, dim)
        return pos_embs


class MemEfficientFiLM(nn.Module):
    def __init__(self, input_size: int, output_size: int, max_iter: int):
        super().__init__()
        self.step_condition = SinusoidalPositionalEncoding(input_size, max_iter, use_conv=True)
        self.input_conv = nn.Conv1d(input_size, input_size, 3, padding=1)
        self.output_conv_1 = nn.Conv1d(input_size, output_size, 3, padding=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_conv.weight)
        nn.init.zeros_(self.input_conv.bias)
        nn.init.xavier_uniform_(self.output_conv_1.weight)
        nn.init.zeros_(self.output_conv_1.bias)
        if not self.memory_efficient:
            nn.init.xavier_uniform_(self.output_conv_2.weight)
            nn.init.zeros_(self.output_conv_2.bias)

    def forward(self, x, t: int):
        x = self.input_conv(x)
        x = F.leaky_relu(x, 0.2)
        x = self.step_condition(x, t)
        shift = self.output_conv_1(x)

        return shift, None


class EmptyFiLM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, t):
        return 0, 1


class UBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor, dilation):
        super().__init__()
        assert isinstance(dilation, (list, tuple))
        assert len(dilation) == 4

        self.factor = factor
        self.block1 = Conv1d(input_size, hidden_size, 1)
        self.block2 = nn.ModuleList([
            Conv1d(input_size, hidden_size, 3, dilation=dilation[0], padding=dilation[0]),
            Conv1d(hidden_size, hidden_size, 3, dilation=dilation[1], padding=dilation[1])
        ])
        self.block3 = nn.ModuleList([
            Conv1d(hidden_size, hidden_size, 3, dilation=dilation[2], padding=dilation[2]),
            Conv1d(hidden_size, hidden_size, 3, dilation=dilation[3], padding=dilation[3])
        ])

    def forward(self, x, film_shift, film_scale: tp.Optional[torch.Tensor]):
        if film_scale is None:
            film_scale = 1.0

        block1 = F.interpolate(x, size=x.shape[-1] * self.factor)
        block1 = self.block1(block1)

        block2 = F.leaky_relu(x, 0.2)
        block2 = F.interpolate(block2, size=x.shape[-1] * self.factor)
        block2 = self.block2[0](block2)
        block2 = film_shift + film_scale * block2
        block2 = F.leaky_relu(block2, 0.2)
        block2 = self.block2[1](block2)

        x = block1 + block2

        block3 = film_shift + film_scale * x
        block3 = F.leaky_relu(block3, 0.2)
        block3 = self.block3[0](block3)
        block3 = film_shift + film_scale * block3
        block3 = F.leaky_relu(block3, 0.2)
        block3 = self.block3[1](block3)

        x = x + block3

        return x


class DBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor):
        super().__init__()
        self.factor = factor

        # self.residual_dense = Conv1d(input_size, hidden_size, 1)
        # self.conv = nn.ModuleList([
        #     Conv1d(input_size, hidden_size, 3, dilation=1, padding=1),
        #     Conv1d(hidden_size, hidden_size, 3, dilation=2, padding=2),
        #     Conv1d(hidden_size, hidden_size, 3, dilation=4, padding=4),
        # ])

        # NOTE : This might be the correct architecture rather than the above one
        #   since parameter size is quite closer to the reported size in the WaveGrad paper (15M).
        self.residual_dense = Conv1d(input_size, input_size, 1)
        self.conv = nn.ModuleList([
            Conv1d(input_size, input_size, 3, dilation=1, padding=1),
            Conv1d(input_size, input_size, 3, dilation=2, padding=2),
            Conv1d(input_size, hidden_size, 3, dilation=4, padding=4),
        ])

        # downsampling module using Conv1d
        # NOTE: When using kernel_size=3 for all downsampling factors,
        #       the parameter size of generator is 15.12 millions.
        kernel_size = factor // 2 * 2 + 1
        padding = kernel_size // 2
        self.down1 = Conv1d(input_size, hidden_size, kernel_size, padding=padding, stride=factor)
        self.down2 = Conv1d(input_size, input_size, kernel_size, padding=padding, stride=factor)

    def forward(self, x):
        residual = self.residual_dense(x)
        residual = self.down1(residual)

        x = self.down2(x)
        for layer in self.conv:
            x = F.leaky_relu(x, 0.2)
            x = layer(x)

        return x + residual


class WaveFitGenerator(nn.Module):
    """
    WaveFit generator module based on WaveGrad.
    See https://arxiv.org/abs/2009.00713 for details.
    """

    def __init__(
        self,
        num_iteration: int,
        dim_feat: int = 1536,
        upsample_factors: tp.List[int] = [5, 4, 3, 2, 2],
        upsample_channels: tp.List[int] = [512, 512, 256, 128, 128],
        downsample_channels: tp.List[int] = [128, 128, 256, 512],
    ):
        super().__init__()

        self.dim_feat = dim_feat
        self.upsample_factors = upsample_factors
        self.upsample_channels = upsample_channels
        self.downsample_factors = upsample_factors[1:][::-1]  # e.g. [2, 2, 3, 4]
        self.downsample_channels = downsample_channels
        assert len(upsample_factors) == len(upsample_channels) == len(downsample_channels) + 1
        self.upsample_rate = math.prod(upsample_factors)

        # Downsampling blocks
        ch_first_down = 32
        self.downsample = nn.ModuleList([Conv1d(1, ch_first_down, 5, padding=2)])
        for i, (factor, ch_out) in enumerate(zip(self.downsample_factors, self.downsample_channels)):
            ch_in = ch_first_down if i == 0 else self.downsample_channels[i - 1]
            self.downsample.append(DBlock(ch_in, ch_out, factor))

        # FiLM layers
        self.film = nn.ModuleList([EmptyFiLM()])
        for i in range(len(self.downsample_channels)):
            ch_in, ch_out = self.downsample_channels[i], self.upsample_channels[-(i + 2)]
            self.film.append(MemEfficientFiLM(ch_in, ch_out, num_iteration))

        # Upsampling blocks
        # NOTE: Dilation factors in a 5-block case follows an implementation in the WaveGrad paper.
        #       Cases other than 5-block are not verified.
        self.upsample = nn.ModuleList()
        for i, (factor, ch_out) in enumerate(zip(self.upsample_factors, self.upsample_channels)):
            ch_in = self.dim_feat if i == 0 else self.upsample_channels[i - 1]
            dilations = [1, 2, 4, 8] if i < 3 else [1, 2, 1, 2]
            self.upsample.append(UBlock(ch_in, ch_out, factor, dilations))

        self.last_conv = Conv1d(128, 1, 3, padding=1)

    def forward(self, y_t: torch.Tensor, audio_feats: torch.Tensor, t: int):
        """
        Args:
            y_t: Noisy input, (bs, 1, L)
            audio_feats: Audio features, (bs, dim_feat, num_frame)
            t: Step index
        Returns:
            n_hat: Estimated noise, (bs, 1, L)
        """
        bs, ch, L = y_t.size()
        bs_, dim, num_frame = audio_feats.size()
        assert bs == bs_ and dim == self.dim_feat and ch == 1
        assert L == num_frame * self.upsample_rate, f"Length mismatch: {L} != {num_frame} * {self.upsample_rate}"

        x = y_t

        # downsampling
        downsampled = []
        for film, layer in zip(self.film, self.downsample):
            x = layer(x)
            downsampled.append(film(x, t))

        # upsampling and FiLM
        x = audio_feats
        for layer, (film_shift, film_scale) in zip(self.upsample, reversed(downsampled)):
            x = layer(x, film_shift, film_scale)

        # to monoral
        x = self.last_conv(x)

        return x
