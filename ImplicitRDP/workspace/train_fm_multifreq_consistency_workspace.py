"""
Training workspace for teacher-free multi-frequency consistency policy.
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import copy
import random
import pathlib

import hydra
import torch
import dill
import numpy as np
import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator

from ImplicitRDP.workspace.base_workspace import BaseWorkspace
from ImplicitRDP.policy.fm_multifreq_consistency_policy import (
    FMMultiFreqConsistencyPolicy,
)
from ImplicitRDP.dataset.base_dataset import BaseImageDataset
from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.common.checkpoint_util import TopKCheckpointManager
from ImplicitRDP.common.json_logger import JsonLogger
from ImplicitRDP.model.diffusion.ema_model import EMAModel
from ImplicitRDP.model.common.lr_scheduler import get_scheduler

from loguru import logger

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainFMMultiFreqConsistencyWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: FMMultiFreqConsistencyPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: FMMultiFreqConsistencyPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        else:
            raise ValueError("Consistency training requires training.use_ema=True")

        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        self.optimizer = self.model.get_optimizer(**optimizer_cfg)

        self.global_step = 0
        self.epoch = 0
        self.scaler = None
        if cfg.training.get("use_amp", False):
            self.scaler = GradScaler()

        if not cfg.training.resume:
            self.exclude_keys = ["optimizer", "scaler"]

        logger.info(
            "Teacher-free consistency training: "
            f"{cfg.training.num_epochs} epochs, "
            f"ct_num_scales={cfg.policy.ct_num_scales}, "
            f"ct_beta=({cfg.policy.ct_beta_a}, {cfg.policy.ct_beta_b})"
        )

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        use_amp = cfg.training.get("use_amp", False)
        amp_dtype = cfg.training.get("amp_dtype", "bf16")
        if use_amp:
            if amp_dtype == "bf16" and torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                amp_dtype = torch.float16
        else:
            amp_dtype = None

        tb_log_dir = os.path.join(self.output_dir, "tb_logs")
        accelerator = Accelerator(log_with="tensorboard", project_dir=tb_log_dir)
        cuda_count = accelerator.num_processes
        accelerator.init_trackers(project_name=cfg.logging.project)

        if cfg.training.resume:
            assert cfg.training.pretrained_ckpt_path is None
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                logger.info(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)

        if cfg.training.pretrained_ckpt_path is not None:
            ckpt_path = cfg.training.pretrained_ckpt_path
            payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
            for key in self.model.state_dict():
                if "normalizer" in key:
                    payload["state_dicts"]["model"][key] = self.model.state_dict()[key]
                    payload["state_dicts"]["ema_model"][key] = self.model.state_dict()[key]
            self.load_payload(
                payload,
                exclude_keys=["optimizer", "scaler"],
                include_keys=[],
                strict=False,
            )
            logger.info(f"Loaded model from checkpoint: {ckpt_path}")

        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        dataloader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
        val_dataloader_cfg = OmegaConf.to_container(cfg.val_dataloader, resolve=True)
        dataloader_cfg["batch_size"] = dataloader_cfg["batch_size"] // cuda_count
        val_dataloader_cfg["batch_size"] = val_dataloader_cfg["batch_size"] // cuda_count
        train_dataloader = DataLoader(dataset, **dataloader_cfg)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **val_dataloader_cfg)

        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs
            )
            // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        ema: EMAModel = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = (
            accelerator.prepare(
                train_dataloader,
                val_dataloader,
                self.model,
                self.optimizer,
                lr_scheduler,
            )
        )

        device = accelerator.device
        self.ema_model.to(device)

        unwrapped_model = accelerator.unwrap_model(self.model)
        unwrapped_model.set_ema_model_ref(self.ema_model.model)
        self.ema_model.set_ema_model_ref(self.ema_model.model)

        if cfg.training.debug:
            cfg.training.num_epochs = 3
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        logger.info("Starting teacher-free consistency training...")

        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                unwrapped_model = accelerator.unwrap_model(self.model)
                unwrapped_model.update_progress(local_epoch_idx / cfg.training.num_epochs)
                unwrapped_model.set_ema_model_ref(self.ema_model.model)

                self.model.train()
                if cfg.training.freeze_encoder:
                    unwrapped_model.obs_encoder.eval()
                    unwrapped_model.obs_encoder.requires_grad_(False)

                step_log = dict()
                train_losses = []
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(
                            batch, lambda x: x.to(device, non_blocking=True)
                        )

                        with autocast(enabled=use_amp, dtype=amp_dtype):
                            raw_loss, loss_dict = unwrapped_model.compute_loss(batch)
                            loss = raw_loss / cfg.training.gradient_accumulate_every

                        if not torch.isfinite(loss):
                            logger.warning(
                                f"Step {self.global_step}: loss is {loss.item()}, skipping batch"
                            )
                            self.optimizer.zero_grad(set_to_none=True)
                            self.global_step += 1
                            continue

                        if use_amp:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            if use_amp:
                                self.scaler.unscale_(self.optimizer)

                            bad_grad_name = None
                            for name, p in unwrapped_model.named_parameters():
                                if p.grad is None:
                                    continue
                                if not torch.isfinite(p.grad).all():
                                    bad_grad_name = name
                                    break

                            if bad_grad_name is not None:
                                logger.warning(
                                    f"Step {self.global_step}: non-finite gradient in {bad_grad_name}, skipping optimizer step"
                                )
                                self.optimizer.zero_grad(set_to_none=True)
                                self.global_step += 1
                                continue

                            max_grad_norm = cfg.training.get("max_grad_norm", 1.0)
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                unwrapped_model.parameters(),
                                max_norm=max_grad_norm,
                                error_if_nonfinite=False,
                            )

                            if not torch.isfinite(grad_norm):
                                logger.warning(
                                    f"Step {self.global_step}: grad_norm is {float(grad_norm)}, skipping optimizer step"
                                )
                                self.optimizer.zero_grad(set_to_none=True)
                                self.global_step += 1
                                continue

                            if use_amp:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                            lr_scheduler.step()
                            # Each process keeps a local EMA target for CT loss.
                            ema.step(accelerator.unwrap_model(self.model))

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }
                        for key, value in loss_dict.items():
                            step_log[f"train_{key}"] = value

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (
                            cfg.training.max_train_steps is not None
                            and batch_idx >= cfg.training.max_train_steps - 1
                        ):
                            break

                train_loss = np.nanmean(train_losses) if len(train_losses) > 0 else 0.0
                step_log["train_loss"] = train_loss

                policy = self.ema_model
                policy.set_ema_model_ref(self.ema_model.model)
                policy.eval()

                if (
                    (self.epoch % cfg.training.val_every) == 0
                    and len(val_dataloader) > 0
                    and accelerator.is_main_process
                ):
                    with torch.no_grad():
                        val_losses = []
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(
                                    batch,
                                    lambda x: x.to(device, non_blocking=True),
                                )
                                with autocast(enabled=use_amp, dtype=amp_dtype):
                                    raw_loss, _ = policy.compute_loss(batch)
                                val_losses.append(raw_loss.item())
                                if (
                                    cfg.training.max_val_steps is not None
                                    and batch_idx >= cfg.training.max_val_steps - 1
                                ):
                                    break
                        if len(val_losses) > 0:
                            val_losses_t = torch.tensor(val_losses)
                            valid_mask = ~torch.isnan(val_losses_t)
                            if valid_mask.any():
                                step_log["val_loss"] = torch.mean(
                                    val_losses_t[valid_mask]
                                ).item()
                            else:
                                step_log["val_loss"] = float("nan")

                if (
                    (self.epoch % cfg.training.checkpoint_every) == 0
                    and accelerator.is_main_process
                ):
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                    self.model = model_ddp

                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        if accelerator.is_main_process:
            final_ckpt_path = os.path.join(
                self.output_dir, "checkpoints", "consistency_final.ckpt"
            )
            model_ddp = self.model
            self.model = accelerator.unwrap_model(self.model)
            self.save_checkpoint(path=final_ckpt_path)
            self.model = model_ddp
            logger.info(f"Saved consistency final checkpoint to {final_ckpt_path}")

        accelerator.end_training()
        logger.info("Training completed!")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainFMMultiFreqConsistencyWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
