"""
Copyright (C) 2025 Yukara Ikemiya

Adapted from the following repo's code under Apache-2.0 License.
https://github.com/huggingface/transformers/blob/a52478253bbe522a420e88ea3940d4d98a935300/src/transformers/models/gemma3n/modular_gemma3n.py

-----------------------------------------------------
Universal Speech Model (USM) from Google.
"""
import typing as tp
import gc
from pathlib import Path

import torch
import torch.nn as nn
from transformers import Gemma3nAudioEncoder, Gemma3nAudioFeatureExtractor
from transformers.models.gemma3n.modeling_gemma3n import Gemma3nRMSNorm

from .parallel_adapter import AdapterLayer
from .base import AudioEncoderAdapter
from utils.torch_common import exists


class GoogleUSMAdapter(AudioEncoderAdapter):
    """
    Parallel adapter for Google USM described in Fig.1 of the Miipher-2 paper.

    NOTE: The shared layer norm before the adapter layers seems to be missing in the Gemma3n implementation.
    Instead, I tentatively introduce a pre-layer norm here for each adapter layer.
    """

    def __init__(
        self,
        n_adaptive_layers: tp.Optional[int] = None,
        model_id: str = "google/gemma-3n-e2b-it",
        encoder_id: str = "Atotti/google-usm",
        adapter_config: dict = {
            "dim_bottleneck": 1024,
            "init_option": "bert",
            "adapter_scalar": 1.0,
            "pre_ln_class": Gemma3nRMSNorm
        }
    ):
        super().__init__()
        self.model_id = model_id
        self.encoder_id = encoder_id

        # Feature extractor (mel-spectrogram)
        # NOTE: This feature extractor is not used when an input is a mel-spectrogram (e.g. forward function).
        #       This could be used at an inference time when the input is a waveform.
        self.feature_extractor = Gemma3nAudioFeatureExtractor.from_pretrained(model_id)

        # Main audio encoder
        self.audio_encoder = Gemma3nAudioEncoder.from_pretrained(encoder_id)
        self.n_adaptive_layers = min(n_adaptive_layers, self.n_layers) if exists(n_adaptive_layers) else self.n_layers
        self.dim = self.audio_encoder.config.hidden_size
        self.adapter_config = adapter_config
        self.adapter_config["dim_in"] = self.dim
        self.adapter_config["pre_ln_class"] = globals()[adapter_config["pre_ln_class"]] \
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
            self.adapter_layers.append(AdapterLayer(**self.adapter_config))
            self.adapter_norms.append(Gemma3nRMSNorm(self.dim))

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

    def forward(
        self,
        audio_mel: torch.Tensor,
        encoder_only: bool = False
    ) -> torch.Tensor:
        """
        Args:
            audio_mel (torch.Tensor): (bs, n_frame, mel_bins)
            encoder_only (bool): If True, only the encoder is used without adaptation.
        Returns:
            (torch.Tensor): (bs, n_frame', dim)
        """
        bs, n_frame, _ = audio_mel.shape
        # NOTE: 'False' for valid frames, 'True' for padded frames
        audio_mel_mask = torch.zeros((bs, n_frame), dtype=torch.bool, device=audio_mel.device)

        feats = self._encoder_forward(audio_mel, audio_mel_mask, encoder_only)

        return feats

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

    @torch.no_grad()
    def forward_waveform(
        self,
        audio: torch.Tensor,
        encoder_only: bool = False,
        device: tp.Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Args:
            audio (torch.Tensor): Monoral audio, (bs, n_sample)
            encoder_only (bool): If True, only the encoder is used without adaptation.
        Returns:
            (torch.Tensor): (bs, n_frame', dim)
        """
        # Feature extraction
        audio_np = audio.cpu().numpy()
        output = self.feature_extractor(audio_np, return_tensors="pt")
        audio_mel = output["input_features"]

        device = audio.device if device is None else device
        audio_mel = audio_mel.to(device)

        # Forward
        feats = self(audio_mel, encoder_only)

        return feats

    def save_state_dict(self, dir_save: str):
        """
        Save only the adapter layer and adaptive norm parameters.
        """
        state = {
            'adapter_layers': self.adapter_layers.state_dict(),
            'adapter_norms': self.adapter_norms.state_dict()
        }
        torch.save(state, Path(dir_save) / "model.pth")

    def load_state_dict(self, dir_load: tp.Optional[str] = None, state: tp.Optional[dict] = None):
        """
        Load only the adapter layer and adaptive norm parameters.
        """
        assert exists(dir_load) or exists(state), "Either dir_load or state must be provided."
        state = state if exists(state) else torch.load(Path(dir_load) / "model.pth")
        self.adapter_layers.load_state_dict(state['adapter_layers'])
        self.adapter_norms.load_state_dict(state['adapter_norms'])

    def get_state_dict(self):
        state = {
            'adapter_layers': self.adapter_layers.state_dict(),
            'adapter_norms': self.adapter_norms.state_dict()
        }

        return state
