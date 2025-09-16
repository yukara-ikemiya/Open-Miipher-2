"""
Copyright (C) 2025 Yukara Ikemiya
"""

import os
import random
import typing as tp
import csv

import torch
import numpy as np

from utils.torch_common import print_once, exists
from .audio_io import get_audio_metadata, load_audio_with_pad
from .modification import Mono, PhaseFlipper, VolumeChanger
from .pretransform import GemmaAudioFeature
from .degradation import AudioClipping, NoiseAddition, RIRReverb, AudioLowpass


def fast_scandir(dir: str, ext: tp.List[str]):
    """ Very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243

    fast_scandir implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py

    Args:
        dir (str): top-level directory at which to begin scanning.
        ext (tp.List[str]): list of allowed file extensions.
    """
    subfolders, files = [], []
    # add starting period to extensions if needed
    ext = ['.' + x if x[0] != '.' else x for x in ext]

    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = os.path.basename(f.path).startswith(".")
                    has_ext = os.path.splitext(f.name)[1].lower() in ext

                    if has_ext and (not is_hidden):
                        files.append(f.path)
            except Exception:
                pass
    except Exception:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)

    return subfolders, files


def get_info_from_csv(csv_path: str, filepath_tag: str = 'file_path',
                      other_info_tags: tp.List[str] = ['sample_rate', 'num_frames', 'num_channels']):
    file_paths = []
    meta_dicts = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)  # 各行を dict として読み込む
        for row in reader:
            file_paths.append(row[filepath_tag])
            meta = {k: int(row[k]) for k in other_info_tags}
            meta_dicts.append(meta)

    # sort by file path
    sorted_indices = np.argsort(file_paths)
    file_paths = [file_paths[i] for i in sorted_indices]
    meta_dicts = [meta_dicts[i] for i in sorted_indices]

    return file_paths, meta_dicts


def get_audio_info(
    paths: tp.List[str],  # directories in which to search
    exts: tp.List[str] = ['.wav', '.mp3', '.flac', '.ogg', '.aif', '.opus']
):
    """recursively get a list of audio filenames"""
    if isinstance(paths, str):
        paths = [paths]

    # get a list of relevant filenames
    filepaths = []
    metas = []
    for p in paths:
        metadata_csv_path = f"{p}/metadata.csv"
        if os.path.exists(metadata_csv_path):
            # If metadata.csv exists, it's faster to get info
            filepaths_, metas_ = get_info_from_csv(metadata_csv_path)
        else:
            _, filepaths_ = fast_scandir(p, exts)
            filepaths_.sort()
            metas_ = []
            for filepath in filepaths_:
                info = get_audio_metadata(filepath, cache=True)
                metas_.append(info)

        filepaths_ = [os.path.join(p, f) for f in filepaths_]
        filepaths.extend(filepaths_)
        metas.extend(metas_)

    return filepaths, metas


class DegradedAudioDataset(torch.utils.data.Dataset):
    """
    A dataset class for loading a pair of a target audio and a degraded audio.
    NOTE: This dataset supports only monoral output.
    """

    def __init__(
        self,
        dirs_audio: tp.List[str],
        dirs_noise: tp.Optional[tp.List[str]] = None,
        sample_size: int = 120000,
        sample_rate: int = 24000,
        pretransform: tp.Optional[str] = None,
        exts: tp.List[str] = ['wav', 'flac'],
        # augmentation
        augment_shift: bool = True,
        augment_flip: bool = True,
        augment_volume: bool = True,
        volume_range: tp.Tuple[float, float] = (0.25, 1.0),
        # degradation
        deg_types: tp.List[str] = ['clipping', 'noise', 'reverb', 'lowpass'],
        n_deg_comb: int = 3,  # maximum number of combined degradations
        prob_no_deg: float = 0.05,  # probability of no degradation samples
        clean_only: bool = False,  # If true, only clean audio is returned.
    ):

        super().__init__()
        self.sample_size = sample_size
        self.sr = sample_rate
        self.augment_shift = augment_shift

        self.deg_types = deg_types
        self.n_deg_comb = min(n_deg_comb, len(deg_types))
        self.prob_no_deg = prob_no_deg
        self.clean_only = clean_only

        print_once('[Dataset instantiation]')

        # Degradation modules
        self.degradations = {
            'clipping': AudioClipping(sample_rate=sample_rate),
            'noise': NoiseAddition(sample_rate=sample_rate),
            'reverb': RIRReverb(sample_rate=sample_rate),
            'lowpass': AudioLowpass(sample_rate=sample_rate)
        }

        # Audio augmentations
        self.ch_encoding = torch.nn.Sequential(Mono())
        self.augs = torch.nn.Sequential(
            PhaseFlipper() if augment_flip else torch.nn.Identity(),
            VolumeChanger(*volume_range) if augment_volume else torch.nn.Identity()
        )

        # Pre-transform
        if pretransform == "gemma":
            self.pretransform = GemmaAudioFeature()
        else:
            self.pretransform = None

        # find all audio files
        print_once('\t->-> Searching AUDIO files...')
        self.filepaths, self.metas = get_audio_info(dirs_audio, exts=exts)
        print_once(f'\t->-> Found {len(self.filepaths)} AUDIO files.')

        if exists(dirs_noise) and ('noise' in self.deg_types) and (not self.clean_only):
            print_once('\t->-> Searching NOISE files...')
            self.filepaths_noise, self.metas_noise = get_audio_info(dirs_noise, exts=exts)
            print_once(f'\t->-> Found {len(self.filepaths_noise)} NOISE files.')

    def get_track_info(self, idx):
        filepath = self.filepaths[idx]
        info = self.metas[idx]
        max_ofs = max(0, info['num_frames'] - self.sample_size)
        offset = random.randint(0, max_ofs) if (self.augment_shift and max_ofs) else 0
        return filepath, offset, info

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):

        filename, offset, info = self.get_track_info(idx)
        # Load audio
        audio = load_audio_with_pad(filename, info, self.sr, self.sample_size, offset)
        # To mono
        audio = self.ch_encoding(audio).squeeze(0)  # (L,)
        # Audio augmentations
        audio = self.augs(audio)
        target = self.pretransform(audio, self.sr) if exists(self.pretransform) else audio

        # Degradation
        if self.clean_only:
            deg = deg_audio = 0.
        else:
            num_deg = random.randint(1, self.n_deg_comb) if random.random() > self.prob_no_deg else 0
            # randomly sample degradations
            deg_types = random.sample(self.deg_types, num_deg)
            deg_audio = audio.clone()
            # apply degradations
            for deg_type in deg_types:
                deg_audio = self.degradations[deg_type](deg_audio)

            deg = self.pretransform(deg_audio, self.sr) if exists(self.pretransform) else deg_audio
            info['deg_types'] = deg_types

        return target, deg, audio, deg_audio, info
