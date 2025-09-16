import os
import argparse
from tqdm import tqdm

import pandas as pd
import torchaudio


def fast_scandir(dir: str, exts: list = ['wav', 'flac']) -> tuple:
    """ Very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243

    fast_scandir implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py

    Args:
        dir (str): top-level directory at which to begin scanning.
        exts (tp.List[str]): list of allowed file extensions.
    """
    subfolders, files = [], []
    # add starting period to extensions if needed
    exts = ['.' + x if x[0] != '.' else x for x in exts]

    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = os.path.basename(f.path).startswith(".")
                    has_ext = os.path.splitext(f.name)[1].lower() in exts

                    if has_ext and (not is_hidden):
                        files.append(f.path)
            except Exception:
                pass
    except Exception:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, exts)
        subfolders.extend(sf)
        files.extend(f)

    return subfolders, files


def get_audio_metadata(filepath):
    info_ = torchaudio.info(filepath)
    sample_rate = info_.sample_rate
    num_channels = info_.num_channels
    num_frames = info_.num_frames

    info = {
        'sample_rate': sample_rate,
        'num_frames': num_frames,
        'num_channels': num_channels
    }

    return info


def make_metadata_csv(dir: str, exts=['wav', 'flac']):

    csv_path = os.path.join(dir, "metadata.csv")
    rows = []

    _, files = fast_scandir(dir, exts=exts)
    print(f"Found {len(files)} audio files in {dir}")

    for p in tqdm(files):
        info = get_audio_metadata(p)
        rel_path = os.path.relpath(p, dir)
        row = {
            "file_path": rel_path,
            "sample_rate": info["sample_rate"],
            "num_frames": info["num_frames"],
            "num_channels": info["num_channels"]
        }
        rows.append(row)

        # print(row)

    rows.sort(key=lambda x: x["file_path"])
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    print(f"Saved metadata to {csv_path}")


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--root-dir', type=str, required=True, help="A root directory of audio dataset.")
    args = args.parse_args()

    root_dir = args.root_dir

    print(root_dir)
    make_metadata_csv(root_dir)


if __name__ == "__main__":
    main()
