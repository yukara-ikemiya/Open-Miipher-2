"""
Copyright (C) 2025 Yukara Ikemiya
"""

import sys
sys.dont_write_bytecode = True

# DDP
from accelerate import Accelerator, DistributedDataParallelKwargs, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration

import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from ema_pytorch import EMA

from utils.torch_common import print_once, get_world_size, count_parameters, set_seed
from trainer import Trainer


@hydra.main(version_base=None, config_path='../configs/', config_name="default.yaml")
def main(cfg: DictConfig):

    # Update config if ckpt_dir is specified (training resumption)

    if cfg.trainer.ckpt_dir is not None:
        overrides = HydraConfig.get().overrides.task
        overrides = [e for e in overrides if isinstance(e, str)]
        override_conf = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_conf)

        # Load checkpoint configuration
        cfg_ckpt = OmegaConf.load(f'{cfg.trainer.ckpt_dir}/config.yaml')
        cfg = OmegaConf.merge(cfg_ckpt, override_conf)

    # HuggingFace Accelerate for distributed training

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    dl_config = DataLoaderConfiguration(split_batches=True)
    p_config = ProjectConfiguration(project_dir=cfg.trainer.output_dir)
    accel = Accelerator(
        mixed_precision=cfg.trainer.amp,
        dataloader_config=dl_config,
        project_config=p_config,
        kwargs_handlers=[ddp_kwargs],
        log_with='wandb'
    )

    accel.init_trackers(cfg.trainer.logger.project_name, config=OmegaConf.to_container(cfg),
                        init_kwargs={"wandb": {"name": cfg.trainer.logger.run_name, "dir": cfg.trainer.output_dir}})

    if accel.is_main_process:
        print("->->-> DDP Initialized.")
        print(f"->->-> World size (Number of GPUs): {get_world_size()}")

    set_seed(cfg.trainer.seed)

    # Dataset

    batch_size = cfg.trainer.batch_size
    num_workers = cfg.trainer.num_workers
    train_dataset = hydra.utils.instantiate(cfg.data.train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

    # Model

    model = hydra.utils.instantiate(cfg.model)

    # EMA

    ema = None
    if accel.is_main_process:
        ema = EMA(model, **cfg.trainer.ema)
        ema.to(accel.device)

    # Optimizer

    # check if cfg.optimizer has optimizer_d
    if 'optimizer_d' in cfg.optimizer:
        # discriminator exists
        opt = hydra.utils.instantiate(cfg.optimizer.optimizer)(params=model.vocoder.parameters())
        sche = hydra.utils.instantiate(cfg.optimizer.scheduler)(optimizer=opt)
        opt_d = hydra.utils.instantiate(cfg.optimizer.optimizer_d)(params=model.discriminator.parameters())
        sche_d = hydra.utils.instantiate(cfg.optimizer.scheduler_d)(optimizer=opt_d)
    else:
        opt = hydra.utils.instantiate(cfg.optimizer.optimizer)(params=model.parameters())
        sche = hydra.utils.instantiate(cfg.optimizer.scheduler)(optimizer=opt)
        opt_d = None
        sche_d = None

    # Log

    model.train()
    num_params = count_parameters(model) / 1e6
    if accel.is_main_process:
        print("=== Parameters ===")
        print(f"\tModel:\t{num_params:.2f} [million]")
        print("=== Dataset ===")
        print(f"\tBatch size: {cfg.trainer.batch_size}")
        print("\tTrain data:")
        print(f"\t\tChunks:  {len(train_dataset)}")
        print(f"\t\tBatches: {len(train_dataset)//cfg.trainer.batch_size}")

    # Prepare for DDP

    train_dataloader, model, opt, sche = \
        accel.prepare(train_dataloader, model, opt, sche)

    # Start training

    trainer = Trainer(
        model=model,
        ema=ema,
        optimizer=opt,
        scheduler=sche,
        optimizer_d=opt_d,
        scheduler_d=sche_d,
        train_dataloader=train_dataloader,
        accel=accel,
        cfg=cfg,
        ckpt_dir=cfg.trainer.ckpt_dir
    )

    trainer.start_training()


if __name__ == '__main__':
    main()
    print("[Training finished.]")
