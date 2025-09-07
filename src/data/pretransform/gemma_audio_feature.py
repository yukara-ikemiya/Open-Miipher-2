"""
Copyright (C) 2025 Yukara Ikemiya

--------------
A wrapper of Gemma audio feature extractor
"""

import numpy as np
import torch
from torchaudio import transforms as T
from transformers import Gemma3nAudioFeatureExtractor


class GemmaAudioFeature:
    """
    A wrapper around the Gemma3nAudioFeatureExtractor.

    NOTE: Feature extraction is executed on CPU.
    """

    def __init__(self, model_id="google/gemma-3n-e2b-it"):
        self.extractor = Gemma3nAudioFeatureExtractor.from_pretrained(model_id)
        self.sr = self.extractor.sampling_rate

    def __call__(self, audio: np.ndarray, sr_in: int) -> torch.Tensor:
        """
        Args:
            audio (np.ndarray): (num_samples)
        Returns:
            (torch.Tensor): (num_frames, feature_dim)
        """
        if sr_in != self.sr:
            resample_tf = T.Resample(sr_in, self.sr)
            audio = resample_tf(audio)

        audio = audio.reshape(1, -1)  # (1, L)
        output = self.extractor(audio, return_tensors="pt")
        audio_mel = output["input_features"]
        # The encoder expects a padding mask (True for padding), while the feature extractor
        # returns an attention mask (True for valid tokens). We must invert it.
        # NOTE: 'False' for valid frames, 'True' for padded frames
        # audio_mel_mask = ~output["input_features_mask"].to(torch.bool) # not used for now

        return audio_mel.squeeze(0)  # (num_frames, feature_dim)
