"""
Copyright (C) 2024 Yukara Ikemiya
"""

import os
import random
import typing as tp
import csv

import torch
import numpy as np

from utils.torch_common import print_once
from .modification import Stereo, Mono, PhaseFlipper, VolumeChanger
from .audio_io import get_audio_metadata, load_audio_with_pad


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


class MultiAudioSourceDataset(torch.utils.data.Dataset):
    """
    A dataset class for loading multiple audio sources.
    """

    def __init__(
        self,
        dir_list: tp.List[str],
        num_src: int,
        sample_size: int = 120000,
        sample_rate: int = 24000,
        out_channels="mono",
        exts: tp.List[str] = ['wav'],
        # augmentation
        augment_shift: bool = True,
        augment_flip: bool = True,
        augment_volume: bool = True,
        volume_range: tp.Tuple[float, float] = (-29., -19.),
        mixture_clipping: bool = True,  # whether to clip the mixture audio into [-1, 1] range
        # Others
        max_samples: tp.Optional[int] = None
    ):
        assert out_channels in ['mono', 'stereo']

        super().__init__()
        self.num_src = num_src
        self.sample_size = sample_size
        self.sr = sample_rate
        self.augment_shift = augment_shift
        self.mixture_clipping = mixture_clipping
        self.out_channels = out_channels

        print_once('[Dataset instantiation]')

        self.ch_encoding = torch.nn.Sequential(
            Stereo() if self.out_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.out_channels == "mono" else torch.nn.Identity(),
        )

        self.augs = torch.nn.Sequential(
            PhaseFlipper() if augment_flip else torch.nn.Identity(),
            VolumeChanger(*volume_range) if augment_volume else torch.nn.Identity()
        )

        # find all audio files
        print_once('\t->-> Searching audio files...')
        self.filepaths, self.metas = get_audio_info(dir_list, exts=exts)

        assert len(self.filepaths) > self.num_src > 0

        max_samples = max_samples if max_samples else len(self.filepaths)
        self.filepaths = self.filepaths[:max_samples]
        self.metas = self.metas[:max_samples]
        print_once(f'\t->-> Found {len(self.filepaths)} files.')

    def get_track_info(self, idx):
        filepath = self.filepaths[idx]
        info = self.metas[idx]
        max_ofs = max(0, info['num_frames'] - self.sample_size)
        offset = random.randint(0, max_ofs) if (self.augment_shift and max_ofs) else 0
        return filepath, offset, info

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # select other sources
        ids_src = [idx] + self.__select_index(self.num_src - 1, exclude=[idx])

        sources = []
        infos = []
        for idx in ids_src:
            filename, offset, info = self.get_track_info(idx)
            # Load audio
            audio = load_audio_with_pad(filename, info, self.sr, self.sample_size, offset)
            # Fix channel number
            audio = self.ch_encoding(audio)
            # Audio augmentations
            audio = self.augs(audio)

            sources.append(audio)
            infos.append(info)

        audio = torch.stack(sources, dim=0)  # (num_src, ch, sample_size)

        if self.mixture_clipping:
            # The mixture amplitude must be in [-1, 1] range.
            max_amp = audio.sum(dim=0).abs().max()
            if max_amp > 1.0:
                audio = audio / max_amp

        return audio, infos

    def __select_index(self, N, exclude: tp.List[int]):
        """Select random indices with an exclude list."""
        if N < 1:
            return []
        mask = np.ones(len(self), dtype=bool)
        mask[exclude] = False
        valid = np.nonzero(mask)[0]
        selected = random.sample(valid.tolist(), k=N)
        return selected


# debug

class SourceNoisePairDataset(torch.utils.data.Dataset):
    """
    A dataset class for loading an audio source and noises.
    """

    def __init__(
        self,
        dir_list: tp.List[str],
        num_src: int,
        sample_size: int = 120000,
        sample_rate: int = 24000,
        out_channels="mono",
        exts: tp.List[str] = ['wav'],
        # augmentation
        augment_shift: bool = True,
        augment_flip: bool = True,
        augment_volume: bool = True,
        volume_range: tp.Tuple[float, float] = (-29., -19.),
        mixture_clipping: bool = True,  # whether to clip the mixture audio into [-1, 1] range
        # Others
        max_samples: tp.Optional[int] = None
    ):
        assert out_channels in ['mono', 'stereo']

        super().__init__()
        self.num_src = num_src
        self.sample_size = sample_size
        self.sr = sample_rate
        self.augment_shift = augment_shift
        self.mixture_clipping = mixture_clipping
        self.out_channels = out_channels

        print_once('[Dataset instantiation]')

        self.ch_encoding = torch.nn.Sequential(
            Stereo() if self.out_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.out_channels == "mono" else torch.nn.Identity(),
        )

        self.augs = torch.nn.Sequential(
            PhaseFlipper() if augment_flip else torch.nn.Identity(),
            VolumeChanger(*volume_range) if augment_volume else torch.nn.Identity()
        )

        # find all audio files
        print_once('\t->-> Searching audio files...')
        self.filepaths, self.metas = get_audio_info(dir_list, exts=exts)

        assert len(self.filepaths) > self.num_src > 0

        max_samples = max_samples if max_samples else len(self.filepaths)
        self.filepaths = self.filepaths[:max_samples]
        self.metas = self.metas[:max_samples]
        print_once(f'\t->-> Found {len(self.filepaths)} files.')

    def get_track_info(self, idx):
        filepath = self.filepaths[idx]
        info = self.metas[idx]
        max_ofs = max(0, info['num_frames'] - self.sample_size)
        offset = random.randint(0, max_ofs) if (self.augment_shift and max_ofs) else 0
        return filepath, offset, info

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # select other sources
        num_noise = self.num_src - 1

        filename, offset, info = self.get_track_info(idx)
        # Load audio
        audio = load_audio_with_pad(filename, info, self.sr, self.sample_size, offset)
        # Fix channel number
        audio = self.ch_encoding(audio)
        # Audio augmentations
        audio = self.augs(audio)

        sources = [audio]
        infos = [info]

        for _ in range(num_noise):
            noise = torch.randn_like(audio)
            noise = self.ch_encoding(noise)
            noise = self.augs(noise)
            sources.append(noise)

        audio = torch.stack(sources, dim=0)  # (num_src, ch, sample_size)

        if self.mixture_clipping:
            # The mixture amplitude must be in [-1, 1] range.
            max_amp = audio.sum(dim=0).abs().max()
            if max_amp > 1.0:
                audio = audio / max_amp

        return audio, infos
