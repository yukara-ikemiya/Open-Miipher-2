"""
Copyright (C) 2025 Yukara Ikemiya

Adapted from the following repo's code under Apache-2.0 License.
https://github.com/huggingface/transformers/blob/a52478253bbe522a420e88ea3940d4d98a935300/src/transformers/models/gemma3n/modular_gemma3n.py

-----------------------------------------------------
Universal Speech Model (USM) from Google.
"""
import typing as tp
import gc

import torch
import torch.nn as nn
from transformers import Gemma3nAudioEncoder, Gemma3nAudioFeatureExtractor
from transformers.models.gemma3n.modeling_gemma3n import Gemma3nRMSNorm

from .parallel_adapter import AdapterLayer
from utils.torch_common import exists


class GoogleUSMAdapter(nn.Module):
    """
    Parallel adapter for Google USM described in Fig.1 of the Miipher-2 paper.

    NOTE: The shared layer norm before the adapter layers seems to be missing in the Gemma3n implementation.
    Instead, I tentatively introduce a pre-layer norm here for each adapter layer.
    """

    def __init__(
        self,
        n_adaptive_layers: tp.Optional[int] = None,
        encoder_id="Atotti/google-usm",
        model_id="google/gemma-3n-e2b-it",
        adapter_config: dict = {
            "dim_in": 1536,
            "dim_bottleneck": 1024,
            "init_option": "bert",
            "adapter_scalar": 1.0,
            "pre_ln_class": Gemma3nRMSNorm
        }
    ):
        super().__init__()
        self.encoder_id = encoder_id
        self.model_id = model_id

        self.feature_extractor = Gemma3nAudioFeatureExtractor.from_pretrained(model_id)
        self.audio_encoder = Gemma3nAudioEncoder.from_pretrained(encoder_id)
        self.n_adaptive_layers = min(n_adaptive_layers, self.n_layers) if exists(n_adaptive_layers) else self.n_layers
        self.dim = self.audio_encoder.config.hidden_size
        assert adapter_config["dim_in"] == self.dim, \
            f"Adapter dim_in {adapter_config['dim_in']} does not match encoder hidden size {self.dim}."
        adapter_config["pre_ln_class"] = globals()[adapter_config["pre_ln_class"]] \
            if isinstance(adapter_config["pre_ln_class"], str) else adapter_config["pre_ln_class"]

        # Remove unused layers
        layers = self.audio_encoder.conformer
        self.audio_encoder.conformer = nn.ModuleList(layers[:self.n_adaptive_layers])
        del layers  # Free memory
        gc.collect()
        torch.cuda.empty_cache()

        # Exclude encoder from learnable modules
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        self.audio_encoder.eval()

        # Adapter layers
        self.adapter_layers = nn.ModuleList()
        self.adapter_norms = nn.ModuleList()
        for _ in range(self.n_adaptive_layers):
            self.adapter_layers.append(AdapterLayer(**adapter_config))
            self.adapter_norms.append(Gemma3nRMSNorm(self.dim))

    @property
    def sampling_rate(self) -> int:
        return self.feature_extractor.sampling_rate

    @property
    def n_layers(self) -> int:
        return len(self.audio_encoder.conformer)

    def eval(self):
        super().eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        self.audio_encoder.eval()  # Keep the encoder in eval mode
        return self

    def forward(self, audio_waveform: torch.Tensor, encoder_only: bool = False):
        # audio to mel spectrogram
        inputs = self.feature_extractor(audio_waveform, return_tensors="pt")

        audio_mel = inputs["input_features"]
        audio_mel_mask = (inputs["input_features_mask"] == 0)

        output = self._encoder_forward(audio_mel, audio_mel_mask, encoder_only)

        return output

    def _encoder_forward(
        self, audio_mel: torch.Tensor, audio_mel_mask: torch.BoolTensor, encoder_only: bool = False
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        """Encodes a batch of MELs.

        Args:
            audio_mel: a torch.Tensor of shape [batch, num_frames, num_channels, mel_bins].
            audio_mel_mask: a torch.BoolTensor of shape [batch, num_frames].

        Returns:
            feats: a torch.Tensor of shape `[batch_size, frame_length, self.dim]`
        """
        audio_encodings = self.audio_encoder.subsample_conv_projection(audio_mel)  # audio_encodings: [B, T_sub, D]

        # Subsample the input audio_mel_mask to match the time dimension of audio_encodings (T_sub)
        t_sub = audio_encodings.shape[1]

        time_stride_product = 1
        for stride_pair_idx in range(len(self.audio_encoder.config.sscp_conv_stride_size)):
            time_stride_product *= self.audio_encoder.config.sscp_conv_stride_size[stride_pair_idx][0]

        # Create indices for gathering from the original mask.
        # These indices map to original time steps corresponding to the start of each
        # receptive field in the subsampled output.
        indices = torch.arange(t_sub, device=audio_mel_mask.device) * time_stride_product
        indices = torch.clamp(indices, max=audio_mel_mask.shape[1] - 1)  # Ensure indices are valid

        # Expand indices for batch compatibility if B > 1 and indices is 1D.
        if audio_mel_mask.ndim > 1 and indices.ndim == 1:
            indices = indices.unsqueeze(0).expand(audio_mel_mask.shape[0], -1)  # [B, T_sub]
        elif (
            audio_mel_mask.ndim == indices.ndim
            and audio_mel_mask.shape[0] == 1
            and indices.shape[0] != 1
            and t_sub == indices.shape[0]
        ):
            # Handle case where B=1 but indices became [T_sub] instead of [1, T_sub]
            indices = indices.unsqueeze(0)

        current_mask = torch.gather(audio_mel_mask, 1, indices)  # [B, T_sub]

        # Adaptation
        feats = audio_encodings
        for i, (conformer_block, adapter_layer, adapter_norm) \
                in enumerate(zip(self.audio_encoder.conformer, self.adapter_layers, self.adapter_norms)):
            feats = self._block_forward(
                conformer_block,
                adapter_layer,
                adapter_norm,
                feats,
                current_mask,
                encoder_only
            )

        return feats

    def _block_forward(
        self,
        c_block,
        adapter_layer,
        adapter_norm,
        feats: torch.Tensor,
        mask: torch.BoolTensor,
        encoder_only: bool = False
    ) -> torch.Tensor:
        """ Gemma3nAudioConformerBlock forward pass with adaptation (Fig.1) """

        feats = c_block.ffw_layer_start(feats)
        feats = c_block.attention(feats, mask)
        validity_mask_for_lconv = ~mask  # True for valid
        feats_for_lconv_input = feats * validity_mask_for_lconv.unsqueeze(-1).to(feats.dtype)
        feats_conv = c_block.lconv1d(feats_for_lconv_input)
        feats = c_block.ffw_layer_end(feats_conv)  # feats_conv + feats_mlp
        feats = torch.clamp(feats, -c_block.gradient_clipping, c_block.gradient_clipping)
        # NOTE: This layer norm doesn't exist in the Fig.1 of the paper.
        feats = c_block.norm(feats)

        if not encoder_only:
            # Adapter layer
            feats_adapt = adapter_layer(feats_conv)
            feats = feats + feats_adapt
            # Post layer norm
            feats = adapter_norm(feats)

        return feats


# class Gemma3nAudioConformerBlock(nn.Module):
#     def __init__(self, config: Gemma3nAudioConfig):
#         super().__init__()
#         self.config = config

#         self.ffw_layer_start = Gemma3nAudioConformerFeedForward(self.config)
#         self.attention = Gemma3nAudioConformerAttention(self.config)
#         self.lconv1d = Gemma3nAudioConformerLightConv1d(self.config)
#         self.ffw_layer_end = Gemma3nAudioConformerFeedForward(self.config)
#         self.register_buffer("gradient_clipping", torch.tensor(self.config.gradient_clipping), persistent=False)
#         self.norm = Gemma3nRMSNorm(self.config.hidden_size)

#     def forward(self, audio_encodings: torch.Tensor, audio_mel_mask: torch.BoolTensor) -> torch.Tensor:
#         audio_encodings = self.ffw_layer_start(audio_encodings)
#         audio_encodings = self.attention(audio_encodings, audio_mel_mask)
#         validity_mask_for_lconv = ~audio_mel_mask  # True for valid
#         audio_encodings_for_lconv_input = audio_encodings * validity_mask_for_lconv.unsqueeze(-1).to(
#             audio_encodings.dtype
#         )
#         audio_encodings = self.lconv1d(audio_encodings_for_lconv_input)

#         audio_encodings = self.ffw_layer_end(audio_encodings)
#         audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
#         output = self.norm(audio_encodings)
#         return output


# class Gemma3nAudioEncoder(PreTrainedModel):
#     """A Universal Speech Encoder -- https://arxiv.org/abs/2303.01037"""

#     config_class = Gemma3nAudioConfig

#     main_input_name = "audio_mel"

#     def __init__(self, config: Gemma3nAudioConfig):
#         super().__init__(config)
#         self.config = config

#         self.subsample_conv_projection = Gemma3nAudioSubSampleConvProjection(config)
#         self.conformer = nn.ModuleList(
#             [Gemma3nAudioConformerBlock(config) for _ in range(config.conf_num_hidden_layers)]
#         )

#     def forward(
#         self, audio_mel: torch.Tensor, audio_mel_mask: torch.BoolTensor
#     ) -> tuple[torch.Tensor, torch.BoolTensor]:
#         """Encodes a batch of MELs.

#         Args:
#             audio_mel: a torch.Tensor of shape [batch, num_frames, num_channels,
#               mel_bins].

#         Returns:
#             audio_encodings: a torch.Tensor of shape
#                 `[batch_size, self.config.audio_soft_tokens_per_image,
#                 self.config.audio_config.hidden_size]`
#             audio_mel_mask: a torch.BoolTensor of shape [batch, num_frames].
#         """
#         audio_encodings = self.subsample_conv_projection(audio_mel)  # audio_encodings: [B, T_sub, D]

#         # Subsample the input audio_mel_mask to match the time dimension of audio_encodings (T_sub)
#         t_sub = audio_encodings.shape[1]

#         time_stride_product = 1
#         for stride_pair_idx in range(len(self.config.sscp_conv_stride_size)):
#             time_stride_product *= self.config.sscp_conv_stride_size[stride_pair_idx][0]

#         # Create indices for gathering from the original mask.
#         # These indices map to original time steps corresponding to the start of each
#         # receptive field in the subsampled output.
#         indices = torch.arange(t_sub, device=audio_mel_mask.device) * time_stride_product
#         indices = torch.clamp(indices, max=audio_mel_mask.shape[1] - 1)  # Ensure indices are valid

#         # Expand indices for batch compatibility if B > 1 and indices is 1D.
#         if audio_mel_mask.ndim > 1 and indices.ndim == 1:
#             indices = indices.unsqueeze(0).expand(audio_mel_mask.shape[0], -1)  # [B, T_sub]
#         elif (
#             audio_mel_mask.ndim == indices.ndim
#             and audio_mel_mask.shape[0] == 1
#             and indices.shape[0] != 1
#             and t_sub == indices.shape[0]
#         ):
#             # Handle case where B=1 but indices became [T_sub] instead of [1, T_sub]
#             indices = indices.unsqueeze(0)

#         current_mask = torch.gather(audio_mel_mask, 1, indices)  # [B, T_sub]

#         for block in self.conformer:
#             audio_encodings = block(audio_encodings, current_mask)  # Pass the processed mask

#         if self.config.conf_reduction_factor > 1:
#             audio_encodings = audio_encodings[:, :: self.config.conf_reduction_factor]
#             # Reduce the mask as well
#             current_mask = current_mask[:, :: self.config.conf_reduction_factor]

#         audio_encodings = audio_encodings.masked_fill(current_mask.unsqueeze(-1), 0.0)
#         return audio_encodings, current_mask
