"""
Copyright (C) 2025 Yukara Ikemiya
"""

import os
import sys
sys.dont_write_bytecode = True
import argparse
import math

import hydra
import torch
import torchaudio
from accelerate import Accelerator
from omegaconf import OmegaConf

from model import Miipher2
from utils.torch_common import get_rank, get_world_size, print_once
from data.dataset import get_audio_info


def make_audio_batch(audio, sample_size: int, overlap: int):
    """
    audio : (ch, L)
    """
    assert 0 <= overlap < sample_size
    L = audio.shape[-1]
    shift = sample_size - overlap

    n_split = math.ceil(max(L - sample_size, 0) / shift) + 1
    # to mono
    audio = audio.mean(0)  # (L)
    batch = []
    for n in range(n_split):
        b = audio[n * shift: n * shift + sample_size]
        if n == n_split - 1:
            b = torch.nn.functional.pad(b, (0, sample_size - len(b)))
        batch.append(b)

    batch = torch.stack(batch, dim=0)  # (n_split, sample_size)
    return batch, L


def cross_fade(preds, overlap: int, L: int):
    """
    preds: (bs, sample_size)
    """
    bs, sample_size = preds.shape
    shift = sample_size - overlap
    full_L = sample_size + (bs - 1) * shift
    win = torch.bartlett_window(overlap * 2, device=preds.device)

    buf = torch.zeros(full_L, device=preds.device)
    pre_overlap = None
    for idx in range(bs):
        pred = preds[idx]  # (sample_size)
        ofs = idx * shift
        if idx != 0:
            # Fix volume
            # NOTE: Since volume is changed by gain normalization in WaveFit module,
            #       it have to be adjusted not to be discontinuous.
            cur_overlap = pred[:overlap]
            volume_rescale = (pre_overlap.pow(2).sum() / (cur_overlap.pow(2).sum() + 1e-10)).sqrt()
            pred *= volume_rescale

            pred[:overlap] *= win[:overlap]

        if idx != bs - 1:
            pre_overlap = pred[-overlap:].clone()
            pred[-overlap:] *= win[overlap:]

        buf[ofs:ofs + sample_size] += pred

    buf = buf[:L]

    return buf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', type=str, help="Checkpoint directory.")
    parser.add_argument('--input-audio-dir', type=str, help="Root directory which contains input audio files.")
    parser.add_argument('--output-dir', type=str, help="Output directory.")
    parser.add_argument('--sample-size', type=int, default=160000, help="Input sample size.")
    parser.add_argument('--sr-in', type=int, default=16000, help="Input sample rate.")
    parser.add_argument('--sr-out', type=int, default=24000, help="Output sample rate.")
    parser.add_argument('--max-batch-size', type=int, default=10, help="Max batch size for inference.")
    parser.add_argument('--overlap-rate', type=float, default=0.05, help="Overlap rate for inference.")
    parser.add_argument('--use-original-name', default=True, type=bool, help="Whether to use an original file name as an output name.")
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    input_audio_dir = args.input_audio_dir
    output_dir = args.output_dir
    sample_size = args.sample_size
    sr_in = args.sr_in
    sr_out = args.sr_out
    max_batch_size = args.max_batch_size
    overlap_rate = args.overlap_rate
    use_original_name = args.use_original_name

    # Distributed inference
    accel = Accelerator()
    device = accel.device
    rank = get_rank()
    world_size = get_world_size()

    print_once(f"Checkpoint dir  : {ckpt_dir}")
    print_once(f"Input audio dir : {input_audio_dir}")
    print_once(f"Output dir      : {output_dir}")

    # Load Miipher-2 model
    cfg_ckpt = OmegaConf.load(f'{ckpt_dir}/config.yaml')
    # remove discriminator and MRSTFT modules
    cfg_ckpt.model.discriminator = None
    cfg_ckpt.model.mrstft = None

    model: Miipher2 = hydra.utils.instantiate(cfg_ckpt.model)
    model.load_state_dict(ckpt_dir)
    model.to(device)
    model.eval()
    print_once("->-> Successfully loaded Miipher-2 model from checkpoint.")

    overlap_in = int(sample_size * overlap_rate)
    overlap_out = int(overlap_in * sr_out / sr_in)
    print_once(f"->-> [Sample size] : {sample_size} samples")
    print_once(f"->-> [Overlap size]: {overlap_in} samples ({overlap_in / sample_size * 100:.1f} %)")

    # Get audio files
    files, _ = get_audio_info([input_audio_dir])
    print_once(f"->-> Found {len(files)} audio files from {input_audio_dir}.")
    os.makedirs(output_dir, exist_ok=True)

    # Split files for each process
    files = files[rank::world_size]

    print_once(f"--- Rank-{rank} : Start inference... ---")

    for idx, f_path in enumerate(files):
        # load and split audio
        audio, sr = torchaudio.load(f_path)
        if sr != sr_in:
            audio = torchaudio.functional.resample(audio, sr, sr_in)
            sr = sr_in

        audio_batch, L = make_audio_batch(audio, sample_size, overlap_in)
        n_iter = math.ceil(audio_batch.shape[0] / max_batch_size)

        audio_batch = audio_batch.to(device)

        # execute
        preds = []
        for n in range(n_iter):
            batch_ = audio_batch[n * max_batch_size:(n + 1) * max_batch_size]
            with torch.no_grad():
                pred = model.inference(batch_)  # (bs, L)

            preds.append(pred)

        preds = torch.cat(preds, dim=0)

        # cross-fade
        L_out = int(L * sr_out / sr_in)
        pred_audio = cross_fade(preds, overlap_out, L_out).cpu()

        # rescale volume to avoid clipping
        pred_audio = pred_audio / (pred_audio.abs().max() + 1e-8) * 0.9

        # save audio
        out_name = os.path.splitext(os.path.basename(f_path))[0] if use_original_name else f"sample_{idx}"
        out_path = f"{output_dir}/{out_name}.wav"
        torchaudio.save(out_path, pred_audio.unsqueeze(0), sample_rate=sr_out, encoding="PCM_F")

    print(f"--- Rank-{rank} : Finished. ---")


if __name__ == '__main__':
    main()
