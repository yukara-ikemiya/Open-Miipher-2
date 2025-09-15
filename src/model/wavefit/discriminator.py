"""
Copyright (C) 2025 Yukara Ikemiya

Adapted from the following repos code under MIT License.
https://github.com/descriptinc/melgan-neurips/
https://github.com/descriptinc/descript-audio-codec
"""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from einops import rearrange


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConv2dWithLeakyReLU(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


class MultiDiscriminator(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, return_feature: bool = True):
        pass

    def compute_G_loss(self, x_fake, x_real):
        """
        The eq.(18) loss
        """
        assert x_fake.shape == x_real.shape

        out_f = self(x_fake, return_feature=True)
        with torch.no_grad():
            out_r = self(x_real, return_feature=True)

        num_D = len(self.model)
        losses = {
            'disc_gan_loss': 0.,
            'disc_feat_loss': 0.
        }

        for i_d in range(num_D):
            n_layer = len(out_f[i_d])

            # GAN loss
            losses['disc_gan_loss'] += (1 - out_f[i_d][-1]).relu().mean()

            # Feature-matching loss
            # eq.(8)
            feat_loss = 0.
            for i_l in range(n_layer - 1):
                feat_loss += F.l1_loss(out_f[i_d][i_l], out_r[i_d][i_l])

            losses['disc_feat_loss'] += feat_loss / (n_layer - 1)

        losses['disc_gan_loss'] /= num_D
        losses['disc_feat_loss'] /= num_D

        return losses

    def compute_D_loss(self, x, mode: str):
        """
        The eq.(7) loss
        """
        assert mode in ['fake', 'real']
        sign = 1 if mode == 'fake' else -1

        out = self(x, return_feature=False)

        num_D = len(self.model)
        losses = {'loss': 0.}

        for i_d in range(num_D):
            # Hinge loss
            losses['loss'] += (1 + sign * out[i_d][-1]).relu().mean()

        losses['loss'] /= num_D

        return losses


class MSDBlock(nn.Module):
    def __init__(self, ndf, n_layers, downsampling_factor):
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            )

        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        )

        model["layer_%d" % (n_layers + 2)] = WNConv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        )

        self.model = model

    def forward(self, x: torch.Tensor, return_feature: bool = True):
        """
        Args:
            x: input audio, (bs, 1, L)
        """
        n_layer = len(self.model)
        results = []
        for idx, (key, layer) in enumerate(self.model.items()):
            x = layer(x)
            if return_feature or (idx == n_layer - 1):
                results.append(x)

        return results


class MSD(MultiDiscriminator):
    """ Multi-scale discriminator """

    def __init__(
        self,
        num_D: int = 3,
        ndf: int = 16,
        n_layers: int = 4,
        downsampling_factor: int = 4
    ):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = MSDBlock(
                ndf, n_layers, downsampling_factor
            )

        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor, return_feature: bool = True):
        """
        Args:
            x: input audio, (bs, 1, L)
        """
        results = []
        for key, disc in self.model.items():
            results.append(disc(x, return_feature))
            x = self.downsample(x)

        return results


class MPDBlock(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                WNConv2dWithLeakyReLU(1, 64, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2dWithLeakyReLU(64, 128, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2dWithLeakyReLU(128, 256, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2dWithLeakyReLU(256, 512, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2dWithLeakyReLU(512, 1024, (5, 1), 1, padding=(2, 0)),
            ]
        )
        self.conv_post = WNConv2dWithLeakyReLU(
            1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False
        )

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        return x

    def forward(self, x, return_feature: bool = True):
        results = []

        x = self.pad_to_period(x)
        x = rearrange(x, "b c (l p) -> b c l p", p=self.period)

        for layer in self.convs:
            x = layer(x)
            if return_feature:
                results.append(x)

        x = self.conv_post(x)
        results.append(x)

        return results


class MPD(MultiDiscriminator):
    """ Multi-period discriminator """

    def __init__(self, periods=[2, 3, 5, 7, 11, 13, 17, 19]):
        super().__init__()
        self.model = nn.ModuleDict()
        for p in periods:
            self.model[f"disc_{p}"] = MPDBlock(p)

        self.apply(weights_init)

    def forward(self, x: torch.Tensor, return_feature: bool = True):
        """
        Args:
            x: input audio, (bs, 1, L)
        """
        results = []
        for key, disc in self.model.items():
            results.append(disc(x, return_feature))

        return results


class Discriminator(nn.Module):
    def __init__(
        self,
        msd_kwargs: dict = {
            "num_D": 3,
            "ndf": 16,
            "n_layers": 4,
            "downsampling_factor": 4
        },
        mpd_kwargs: dict = {
            "periods": [2, 3, 5, 7, 11, 13, 17, 19]
        }
    ):
        super().__init__()
        self.msd = MSD(**msd_kwargs)
        self.mpd = MPD(**mpd_kwargs)

    def compute_G_loss(self, x_fake, x_real):
        losses = {}
        losses_msd = self.msd.compute_G_loss(x_fake, x_real)
        losses_mpd = self.mpd.compute_G_loss(x_fake, x_real)
        for k in losses_msd.keys():
            losses[k] = (losses_msd[k] + losses_mpd[k]) / 2
            losses[f"mpd-{k}"] = losses_mpd[k].detach()
            losses[f"msd-{k}"] = losses_msd[k].detach()

        return losses

    def compute_D_loss(self, x, mode: str):
        losses = {}
        losses_msd = self.msd.compute_D_loss(x, mode)
        losses_mpd = self.mpd.compute_D_loss(x, mode)
        for k in losses_msd.keys():
            losses[k] = (losses_msd[k] + losses_mpd[k]) / 2
            losses[f"mpd-{k}"] = losses_mpd[k].detach()
            losses[f"msd-{k}"] = losses_msd[k].detach()

        return losses
