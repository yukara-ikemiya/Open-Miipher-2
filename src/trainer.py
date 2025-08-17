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

        # timestep sampler
        self.ts_sampler = hydra.utils.instantiate(self.cfg_t.timestep_sampler)

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
                    if self.__its_time(self.cfg_t.logging.n_step_sample):
                        self.__sampling()

                self.states['global_step'] += 1

    def run_step(self, batch, train: bool = True):
        """ One training step """

        # srouces: (bs, n_src, sample_length)
        sources, _ = batch

        # NOTE: Channels are treated as different samples here.
        # TBD : (Implement multi-channel separation)
        sources = rearrange(sources, 'b s c l -> (b c) s l')

        bs, n_src, sample_length = sources.shape

        # sample timesteps
        t = self.ts_sampler.sample(bs, device=self.accel.device)

        # Update

        if train:
            self.opt.zero_grad()

        output = self.model.train_step(sources, t, debug=self.cfg_t.debug)

        if train:
            self.accel.backward(output['loss'])
            if self.accel.sync_gradients:
                self.accel.clip_grad_norm_(self.model.parameters(), self.cfg_t.max_grad_norm)
            self.opt.step()
            self.sche.step()

            # EMA
            if exists(self.ema):
                self.ema.update()

        return {k: v.detach() for k, v in output.items()}

    @torch.no_grad()
    def __sampling(self):
        self.model.eval()

        steps: list = self.cfg_t.logging.steps
        n_sample: int = self.cfg_t.logging.n_samples_per_step
        n_src: int = self.model.n_src

        # randomly select samples
        dataset = self.train_dataloader.dataset
        idxs = torch.randint(len(dataset), size=(n_sample,))
        audios = torch.stack([dataset[idx][0] for idx in idxs], dim=0).to(self.accel.device)
        audios = audios.mean(2)  # remove channels
        mix = audios.sum(dim=1)  # (n_sample, L)

        # columns = ['mix (audio)', 'mix (spec)']
        # columns += [item for pair in [[f"sep-{i} (audio)", f"sep-{i} (spec)"] for i in range(n_src)] for item in pair]
        columns = ['mix (audio)'] + [f"gt-{i} (audio)" for i in range(n_src)]
        columns += [item for pair in [[f"sep-{i} (audio)"] for i in range(n_src)] for item in pair]
        table_audio = wandb.Table(columns=columns)

        for step in steps:
            # sampling
            gen_sample, info = self.model.sample(sources=audios, n_step=step, debug=True)  # (n_sample, n_src, L)
            sources = info.pop('sources')  # (n_sample, n_src, L)
            for k, v in info.items():
                print(f"\t{k}: {v}")

            for idx in range(n_sample):
                data = [wandb.Audio(mix[idx].cpu().numpy().T, sample_rate=dataset.sr)]
                # ground truth sources
                data += [wandb.Audio(sources[idx, i].cpu().numpy().T, sample_rate=dataset.sr) for i in range(n_src)]
                # separated sources
                data += [wandb.Audio(gen_sample[idx, i].cpu().numpy().T, sample_rate=dataset.sr) for i in range(n_src)]

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
        ckpts = {'model': self.model,
                 'optimizer': self.opt,
                 'scheduler': self.sche}
        for name, m in ckpts.items():
            torch.save(m.state_dict(), f"{latest_dir}/{name}.pth")

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
        ckpts = {'model': self.model,
                 'optimizer': self.opt,
                 'scheduler': self.sche}

        for k, v in ckpts.items():
            v.load_state_dict(torch.load(f"{dir}/{k}.pth", weights_only=False))

        with open(f"{dir}/states.json", mode="rt", encoding="utf-8") as f:
            self.states.update(json.load(f))

    def __log_metrics(self, sort_by_key: bool = True):
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

    def __print_metrics(self, sort_by_key: bool = True):
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
