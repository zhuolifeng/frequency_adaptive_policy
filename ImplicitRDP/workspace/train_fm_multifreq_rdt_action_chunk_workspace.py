"""
Training workspace for RDT-style multi-frequency flow matching distillation.

The model predicts x0/action chunks. Stage 1 trains 10Hz and 50Hz with
velocity losses derived from x0_pred. Stage 2 keeps the 10Hz loss and trains
50Hz against velocity targets derived from a frozen Stage-1 teacher action
chunk.
"""

if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import os
import pathlib
import random

import dill
import hydra
import numpy as np
import torch
import tqdm
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ImplicitRDP.common.checkpoint_util import TopKCheckpointManager
from ImplicitRDP.common.json_logger import JsonLogger
from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.dataset.base_dataset import BaseImageDataset
from ImplicitRDP.model.common.lr_scheduler import get_scheduler
from ImplicitRDP.model.diffusion.ema_model import EMAModel
from ImplicitRDP.policy.fm_multifreq_rdt_action_chunk_policy import (
    FMMultiFreqRDTActionChunkPolicy,
)
from ImplicitRDP.workspace.base_workspace import BaseWorkspace

from loguru import logger

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainFMMultiFreqRDTActionChunkWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: FMMultiFreqRDTActionChunkPolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: FMMultiFreqRDTActionChunkPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop("stage2_lr", None)
        optimizer_cfg.pop("stage2_obs_encoder_lr", None)
        self.optimizer = self.model.get_optimizer(**optimizer_cfg)

        self.global_step = 0
        self.epoch = 0
        self.scaler = GradScaler() if cfg.training.get("use_amp", False) else None

        if not cfg.training.resume:
            self.exclude_keys = ["optimizer", "scaler"]

        self.use_distillation = cfg.training.get("use_distillation", True)
        self.stage1_epochs = int(cfg.training.get("stage1_epochs", 600))
        self.stage2_epochs = int(cfg.training.get("stage2_epochs", 0))

        if self.use_distillation and self.stage2_epochs > 0:
            logger.info(
                "Two-stage RDT-style training: "
                f"Stage 1={self.stage1_epochs} epochs, Stage 2={self.stage2_epochs} epochs"
            )
        else:
            logger.info(f"Single-stage RDT backbone training: {self.stage1_epochs} epochs")

    def _make_accelerator(self, cfg):
        log_backend = cfg.logging.get("backend", "wandb")
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        if log_backend == "tensorboard":
            tb_log_dir = os.path.join(self.output_dir, "tb_logs")
            accelerator = Accelerator(
                log_with="tensorboard",
                project_dir=tb_log_dir,
                kwargs_handlers=[ddp_kwargs],
            )
        elif log_backend == "wandb":
            accelerator = Accelerator(
                log_with="wandb",
                kwargs_handlers=[ddp_kwargs],
            )
        elif log_backend in ("none", "disabled", None):
            accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        else:
            raise ValueError(
                f"Unsupported logging.backend={log_backend}. "
                "Expected 'wandb', 'tensorboard', or 'none'."
            )

        if log_backend == "wandb":
            wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            wandb_cfg.pop("project", None)
            wandb_cfg.pop("backend", None)
            accelerator.init_trackers(
                project_name=cfg.logging.project,
                config=OmegaConf.to_container(cfg, resolve=True),
                init_kwargs={"wandb": wandb_cfg},
            )
        elif log_backend == "tensorboard":
            accelerator.init_trackers(project_name=cfg.logging.project)
        return accelerator

    @staticmethod
    def _set_fixed_ema_decay(ema: EMAModel, decay: float):
        ema.min_value = decay
        ema.max_value = decay
        ema.power = 1.0
        ema.inv_gamma = 1.0
        ema.decay = decay

    def _load_pretrained_if_needed(self, cfg):
        if cfg.training.resume:
            assert cfg.training.pretrained_ckpt_path is None
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                logger.info(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)

        if cfg.training.pretrained_ckpt_path is None:
            return

        ckpt_path = cfg.training.pretrained_ckpt_path
        payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
        for key in self.model.state_dict():
            if "normalizer" in key:
                payload["state_dicts"]["model"][key] = self.model.state_dict()[key]
                if "ema_model" in payload["state_dicts"]:
                    payload["state_dicts"]["ema_model"][key] = self.model.state_dict()[key]
        self.load_payload(
            payload, exclude_keys=["optimizer", "scaler"], include_keys=[], strict=False
        )
        logger.info(f"Loaded model from checkpoint: {ckpt_path}")

    def _build_stage2_teacher(self, unwrapped_model, device):
        teacher_source = self.ema_model.model if self.ema_model is not None else unwrapped_model.model
        teacher_model = copy.deepcopy(teacher_source)
        teacher_model.to(device)
        teacher_model.requires_grad_(False)
        teacher_model.eval()
        unwrapped_model.set_teacher_model(teacher_model)

        if self.ema_model is not None:
            self.ema_model.set_teacher_model(teacher_model)
        logger.info(
            "Created frozen Stage-1 teacher backbone for 50Hz action-chunk distillation"
        )

    def _configure_stage2_modules(self, unwrapped_model, cfg):
        if cfg.training.get("freeze_condition_modules_stage2", True):
            unwrapped_model.freeze_condition_modules()
            if self.ema_model is not None:
                self.ema_model.freeze_condition_modules()
            logger.info("Stage 2 freezes obs_encoder, rnn_lowfreq, and rnn_highfreq")

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

        if self.use_distillation and self.stage2_epochs > 0:
            cfg.training.num_epochs = self.stage1_epochs + self.stage2_epochs
        else:
            cfg.training.num_epochs = self.stage1_epochs

        accelerator = self._make_accelerator(cfg)
        cuda_count = accelerator.num_processes

        self._load_pretrained_if_needed(cfg)

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
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
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs
            ) // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

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

        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        if cfg.training.debug:
            cfg.training.num_epochs = 3
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            self.stage1_epochs = 1
            self.stage2_epochs = 1 if self.use_distillation else 0

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        logger.info(
            "Starting RDT-style multi-frequency training: "
            f"stage1_epochs={self.stage1_epochs}, stage2_epochs={self.stage2_epochs}, "
            f"use_distillation={self.use_distillation}"
        )

        stage2_initialized = False

        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                current_stage = 1
                if (
                    self.use_distillation
                    and self.stage2_epochs > 0
                    and local_epoch_idx >= self.stage1_epochs
                ):
                    current_stage = 2

                unwrapped_model = accelerator.unwrap_model(self.model)
                unwrapped_model.set_training_stage(current_stage)
                unwrapped_model.update_progress(local_epoch_idx / cfg.training.num_epochs)
                if self.ema_model is not None:
                    self.ema_model.set_training_stage(current_stage)

                if current_stage == 2 and not stage2_initialized:
                    if accelerator.is_main_process:
                        stage1_ckpt_path = os.path.join(
                            self.output_dir, "checkpoints", "stage1_final.ckpt"
                        )
                        model_ddp = self.model
                        self.model = unwrapped_model
                        self.save_checkpoint(path=stage1_ckpt_path)
                        self.model = model_ddp
                        logger.info(f"Saved Stage 1 checkpoint to {stage1_ckpt_path}")

                    self._build_stage2_teacher(unwrapped_model, device)
                    self._configure_stage2_modules(unwrapped_model, cfg)

                    optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
                    stage2_lr = optimizer_cfg.pop("stage2_lr", optimizer_cfg["lr"])
                    stage2_obs_encoder_lr = optimizer_cfg.pop(
                        "stage2_obs_encoder_lr", optimizer_cfg.get("obs_encoder_lr", None)
                    )
                    optimizer_cfg["lr"] = stage2_lr
                    optimizer_cfg["obs_encoder_lr"] = stage2_obs_encoder_lr
                    self.optimizer = unwrapped_model.get_optimizer(**optimizer_cfg)
                    if self.scaler is not None:
                        self.scaler = GradScaler()

                    stage2_steps = (
                        len(train_dataloader) * self.stage2_epochs
                    ) // cfg.training.gradient_accumulate_every
                    lr_scheduler_stage2 = get_scheduler(
                        cfg.training.lr_scheduler,
                        optimizer=self.optimizer,
                        num_warmup_steps=cfg.training.get(
                            "stage2_lr_warmup_steps", cfg.training.lr_warmup_steps
                        ),
                        num_training_steps=stage2_steps,
                        last_epoch=-1,
                    )

                    train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = (
                        accelerator.prepare(
                            train_dataloader,
                            val_dataloader,
                            unwrapped_model,
                            self.optimizer,
                            lr_scheduler_stage2,
                        )
                    )

                    if ema is not None:
                        self._set_fixed_ema_decay(
                            ema, cfg.training.get("stage2_ema_rate", 0.9999)
                        )
                    stage2_initialized = True
                    logger.info(
                        "Stage 2 optimizer rebuilt: "
                        f"lr={stage2_lr}, obs_encoder_lr={stage2_obs_encoder_lr}"
                    )

                self.model.train()
                unwrapped_model = accelerator.unwrap_model(self.model)
                if current_stage == 1 and cfg.training.freeze_encoder:
                    unwrapped_model.obs_encoder.eval()
                    unwrapped_model.obs_encoder.requires_grad_(False)
                if current_stage == 2 and cfg.training.get("freeze_condition_modules_stage2", True):
                    unwrapped_model.freeze_condition_modules()

                step_log = {}
                train_losses = []
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                        with autocast(enabled=use_amp, dtype=amp_dtype):
                            raw_loss, loss_dict = unwrapped_model.compute_loss(batch)
                            loss = raw_loss / cfg.training.gradient_accumulate_every

                        if not torch.isfinite(loss):
                            logger.warning(
                                f"Step {self.global_step}: loss={loss.item()}, skipping batch"
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
                                    f"Step {self.global_step}: grad_norm={float(grad_norm)}, skipping optimizer step"
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
                            if ema is not None:
                                ema.step(accelerator.unwrap_model(self.model))

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                            "training_stage": float(current_stage),
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

                step_log["train_loss"] = (
                    np.nanmean(train_losses) if len(train_losses) > 0 else 0.0
                )

                policy = self.ema_model if cfg.training.use_ema else accelerator.unwrap_model(self.model)
                policy.set_training_stage(current_stage)
                if current_stage == 2 and accelerator.unwrap_model(self.model).teacher_model is not None:
                    policy.set_teacher_model(accelerator.unwrap_model(self.model).teacher_model)
                policy.eval()

                if (
                    self.epoch % cfg.training.val_every == 0
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
                                    batch, lambda x: x.to(device, non_blocking=True)
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
                            step_log["val_loss"] = (
                                torch.mean(val_losses_t[valid_mask]).item()
                                if valid_mask.any()
                                else float("nan")
                            )

                if (
                    self.epoch % cfg.training.checkpoint_every == 0
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
            final_name = (
                "stage2_final.ckpt"
                if self.use_distillation and self.stage2_epochs > 0
                else "stage1_final.ckpt"
            )
            final_path = os.path.join(self.output_dir, "checkpoints", final_name)
            model_ddp = self.model
            self.model = accelerator.unwrap_model(self.model)
            self.save_checkpoint(path=final_path)
            self.model = model_ddp
            logger.info(f"Saved final RDT distill checkpoint to {final_path}")

        accelerator.end_training()
        logger.info("RDT-style multi-frequency training completed")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainFMMultiFreqRDTActionChunkWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
