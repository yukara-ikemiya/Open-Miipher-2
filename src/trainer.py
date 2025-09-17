"""
Copyright (C) 2025 Yukara Ikemiya
"""

import os

import torch
import wandb
import hydra
from einops import rearrange

from utils.logging import MetricsLogger
from utils.torch_common import exists, sort_dict, print_once

from model import AudioEncoderAdapter, Miipher2, MiipherMode


class Trainer:
    def __init__(
        self,
        model,              # model
        ema,                # exponential moving average
        optimizer,          # optimizer
        scheduler,          # scheduler
        train_dataloader,
        accel,              # Accelerator object
        cfg,                # Configurations
        # Discriminator options (for WaveFit training)
        optimizer_d=None,
        scheduler_d=None,
        # Resume training from a checkpoint directory
        ckpt_dir=None
    ):
        self.model = accel.unwrap_model(model)
        self.ema = ema
        self.opt = optimizer
        self.sche = scheduler
        self.train_dataloader = train_dataloader
        self.accel = accel
        self.cfg = cfg
        self.cfg_t = cfg.trainer
        self.EPS = 1e-8

        # discriminator
        self.have_disc = exists(optimizer_d) and exists(scheduler_d)
        self.opt_d = optimizer_d
        self.sche_d = scheduler_d

        self.logger = MetricsLogger()           # Logger for WandB
        self.logger_print = MetricsLogger()     # Logger for printing
        self.logger_test = MetricsLogger()      # Logger for test

        self.states = {'global_step': 0, 'best_metrics': float('inf'), 'latest_metrics': float('inf')}

        # time measurement
        self.s_event = torch.cuda.Event(enable_timing=True)
        self.e_event = torch.cuda.Event(enable_timing=True)

        # resume training
        if ckpt_dir is not None:
            self.__load_ckpt(ckpt_dir)

    def start_training(self):
        """
        Start training with infinite loops
        """
        self.model.train()
        self.s_event.record()

        print_once("\n[ Started training ]\n")

        while True:
            for batch in self.train_dataloader:
                # Update
                metrics = self.run_step(batch)

                if self.accel.is_main_process:
                    self.logger.add(metrics)
                    self.logger_print.add(metrics)

                    # Log
                    if self.__its_time(self.cfg_t.logging.n_step_log):
                        self.__log_metrics()

                    # Print
                    if self.__its_time(self.cfg_t.logging.n_step_print):
                        self.__print_metrics()

                    # Save checkpoint
                    if self.__its_time(self.cfg_t.logging.n_step_ckpt):
                        self.__save_ckpt()

                    # Sample
                    if not isinstance(self.model, AudioEncoderAdapter):
                        if self.__its_time(self.cfg_t.logging.n_step_sample):
                            self.__sampling()

                self.states['global_step'] += 1

    def run_step(self, batch, train: bool = True):
        """ One training step """

        # target and degraded audios: (bs, ch, sample_length)
        x_tgt, x_deg, clean_audio, noisy_audio, _ = batch

        # Update

        if train:
            self.opt.zero_grad()
            if self.have_disc:
                self.opt_d.zero_grad()

        if isinstance(self.model, AudioEncoderAdapter):
            output = self.model.train_step(x_tgt, x_deg, train=train)
        elif isinstance(self.model, Miipher2):
            input = x_tgt if self.model.mode == MiipherMode.CLEAN_INPUT else x_deg
            output = self.model.train_step(clean_audio, input, train=train)
        else:
            raise NotImplementedError(f"Model class '{self.model.__class__.__name__}' is not supported.")

        if train:
            self.accel.backward(output['loss'])
            if self.accel.sync_gradients:
                self.accel.clip_grad_norm_(self.model.parameters(), self.cfg_t.max_grad_norm)
            self.opt.step()
            self.sche.step()

            if self.have_disc:
                self.accel.backward(output['D/loss_d'])
                if self.accel.sync_gradients:
                    self.accel.clip_grad_norm_(self.model.discriminator.parameters(), self.cfg_t.max_grad_norm)
                self.opt_d.step()
                self.sche_d.step()

            # EMA
            if exists(self.ema):
                self.ema.update()

        return {k: v.detach() for k, v in output.items()}

    @torch.no_grad()
    def __sampling(self):
        # Restoration / Reconstruction samples from Miipher-2
        self.model.eval()

        n_sample: int = self.cfg_t.logging.n_samples_per_step

        # randomly select samples
        dataset = self.train_dataloader.dataset
        idxs = torch.randint(len(dataset), size=(n_sample,))
        x_tgt, x_deg, clean_audio, noisy_audio = [], [], [], []
        for idx in idxs:
            x_tgt_, x_deg_, clean_audio_, noisy_audio_, _ = dataset[idx]
            x_tgt.append(x_tgt_)
            x_deg.append(x_deg_)
            clean_audio.append(clean_audio_)
            noisy_audio.append(noisy_audio_)

        x_tgt = torch.stack(x_tgt, dim=0).to(self.accel.device)
        x_deg = torch.stack(x_deg, dim=0).to(self.accel.device) if self.model.mode != MiipherMode.CLEAN_INPUT else None
        clean_audio = torch.stack(clean_audio, dim=0).to(self.accel.device)
        noisy_audio = torch.stack(noisy_audio, dim=0).to(self.accel.device) if self.model.mode != MiipherMode.CLEAN_INPUT else None

        columns = ['clean (audio)', 'decoded (audio)'] if self.model.mode == MiipherMode.CLEAN_INPUT \
            else ['clean (audio)', 'degraded (audio)', 'restored (audio)']
        table_audio = wandb.Table(columns=columns)

        # sampling
        x_input = x_tgt if self.model.mode == MiipherMode.CLEAN_INPUT else x_deg
        initial_noise = torch.randn_like(clean_audio)
        with torch.no_grad():
            x_preds = self.model(x_input, initial_noise)
            x_pred = x_preds[-1]  # (n_sample, L)

        for idx in range(n_sample):
            # clean audio
            data = [wandb.Audio(clean_audio[idx].cpu().numpy(), sample_rate=dataset.sr)]

            # degraded audio
            if self.model.mode == MiipherMode.NOISY_INPUT:
                data += [wandb.Audio(noisy_audio[idx].cpu().numpy(), sample_rate=dataset.sr)]

            # decoded audio
            data += [wandb.Audio(x_pred[idx].cpu().numpy().T, sample_rate=dataset.sr)]

            table_audio.add_data(*data)

        self.accel.log({'Samples': table_audio}, step=self.states['global_step'])

        self.model.train()

        print("\t->->-> Sampled.")

    def __save_ckpt(self):
        import shutil
        import json
        from omegaconf import OmegaConf

        out_dir = self.cfg_t.output_dir + '/ckpt'

        # save latest ckpt
        latest_dir = out_dir + '/latest'
        os.makedirs(latest_dir, exist_ok=True)

        # save optimizer/scheduler states
        ckpts = {'optimizer': self.opt, 'scheduler': self.sche}
        for name, m in ckpts.items():
            torch.save(m.state_dict(), f"{latest_dir}/{name}.pth")

        # save model states
        self.model.save_state_dict(latest_dir)

        # save states and configuration
        OmegaConf.save(self.cfg, f"{latest_dir}/config.yaml")
        with open(f"{latest_dir}/states.json", mode="wt", encoding="utf-8") as f:
            json.dump(self.states, f, indent=2)

        # save best ckpt
        if self.states['latest_metrics'] == self.states['best_metrics']:
            shutil.copytree(latest_dir, out_dir + '/best', dirs_exist_ok=True)

        print("\t->->-> Saved checkpoints.")

    def __load_ckpt(self, dir: str):
        import json

        print_once(f"\n[Resuming training from the checkpoint directory] -> {dir}")

        ckpts = {'optimizer': self.opt, 'scheduler': self.sche}
        for k, v in ckpts.items():
            v.load_state_dict(torch.load(f"{dir}/{k}.pth", weights_only=False))

        self.model.load_state_dict(dir)

        with open(f"{dir}/states.json", mode="rt", encoding="utf-8") as f:
            self.states.update(json.load(f))

    def __log_metrics(self, sort_by_key: bool = False):
        metrics = self.logger.pop()
        # learning rate
        metrics['lr'] = self.sche.get_last_lr()[0]
        if sort_by_key:
            metrics = sort_dict(metrics)

        self.accel.log(metrics, step=self.states['global_step'])

        # update states
        m_for_ckpt = self.cfg_t.logging.metrics_for_best_ckpt
        m_latest = float(sum([metrics[k].detach() for k in m_for_ckpt]))
        self.states['latest_metrics'] = m_latest
        if m_latest < self.states['best_metrics']:
            self.states['best_metrics'] = m_latest

    def __print_metrics(self, sort_by_key: bool = False):
        self.e_event.record()
        torch.cuda.synchronize()
        p_time = self.s_event.elapsed_time(self.e_event) / 1000.  # [sec]

        metrics = self.logger_print.pop()
        # tensor to scalar
        metrics = {k: v.item() for k, v in metrics.items()}
        if sort_by_key:
            metrics = sort_dict(metrics)

        step = self.states['global_step']
        s = f"Step {step} ({p_time:.1e} [sec]): " + ' / '.join([f"[{k}] - {v:.3e}" for k, v in metrics.items()])
        print(s)

        self.s_event.record()

    def __its_time(self, itv: int):
        return (self.states['global_step'] - 1) % itv == 0
