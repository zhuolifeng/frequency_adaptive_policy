"""
Three-stage workspace for DP multi-frequency confidence + 50Hz CD.

Stage 1: standard 10Hz + 50Hz DP training.
Stage 2: freeze policy, train confidence head from 10Hz sample MSE.
Stage 3: train DP backbone with 10Hz DP + 50Hz CD.
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
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ImplicitRDP.common.checkpoint_util import TopKCheckpointManager
from ImplicitRDP.common.json_logger import JsonLogger
from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.dataset.base_dataset import BaseImageDataset
from ImplicitRDP.model.common.lr_scheduler import get_scheduler
from ImplicitRDP.policy.dp_multifreq_cd_50hz_confidence_policy import (
    DPMultiFreqCD50HzConfidencePolicy,
)
from ImplicitRDP.workspace.base_workspace import BaseWorkspace

from loguru import logger

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDPMultiFreqCD50HzConfidenceWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch", "current_stage"]

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: DPMultiFreqCD50HzConfidencePolicy = hydra.utils.instantiate(cfg.policy)
        self.teacher_model: DPMultiFreqCD50HzConfidencePolicy = copy.deepcopy(self.model)
        self.ema_model: DPMultiFreqCD50HzConfidencePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        else:
            raise ValueError("This workspace expects training.use_ema=True")
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)

        self.stage1_epochs = int(cfg.training.get("stage1_epochs", 600))
        self.stage2_epochs = int(cfg.training.get("stage2_epochs", 250))
        self.stage3_epochs = int(cfg.training.get("stage3_epochs", 100))
        self.current_stage = 1
        self.model.set_training_stage(1)
        self.model.apply_stage_freeze()

        optimizer_cfg = self._optimizer_cfg_for_stage(cfg, stage=1)
        self.optimizer = self.model.get_optimizer(**optimizer_cfg)

        self.global_step = 0
        self.epoch = 0
        self.scaler = GradScaler() if cfg.training.get("use_amp", False) else None

        if not cfg.training.resume:
            self.exclude_keys = ["optimizer", "scaler"]

        logger.info("DP MultiFreq CD 50Hz Confidence training:")
        logger.info(f"  Stage 1: {self.stage1_epochs} epochs, lr={cfg.optimizer.stage1_lr}")
        logger.info(f"  Stage 2: {self.stage2_epochs} epochs, lr={cfg.optimizer.stage2_confidence_lr}")
        logger.info(f"  Stage 3: {self.stage3_epochs} epochs, lr={cfg.optimizer.stage3_lr}")

    def _optimizer_cfg_for_stage(self, cfg, stage: int):
        opt = OmegaConf.to_container(cfg.optimizer, resolve=True)
        betas = opt.get("betas", (0.95, 0.999))
        common = {
            "weight_decay": opt.get("weight_decay", 1.0e-6),
            "obs_encoder_weight_decay": opt.get("obs_encoder_weight_decay", 1.0e-6),
            "betas": betas,
        }
        if stage == 1:
            return {
                **common,
                "lr": opt.get("stage1_lr", 1.0e-4),
                "obs_encoder_lr": opt.get("stage1_obs_encoder_lr", 3.0e-5),
            }
        if stage == 2:
            return {
                **common,
                "lr": opt.get("stage2_confidence_lr", 1.0e-4),
                "obs_encoder_lr": opt.get("stage2_confidence_lr", 1.0e-4),
            }
        if stage == 3:
            return {
                **common,
                "lr": opt.get("stage3_lr", 1.0e-5),
                "obs_encoder_lr": opt.get("stage3_lr", 1.0e-5),
            }
        raise ValueError(f"Invalid stage={stage}")

    def _make_accelerator(self, cfg):
        log_backend = cfg.logging.get("backend", "tensorboard")
        if log_backend == "tensorboard":
            return Accelerator(
                log_with="tensorboard",
                project_dir=os.path.join(self.output_dir, "tb_logs"),
            )
        if log_backend == "wandb":
            return Accelerator(log_with="wandb")
        if log_backend in ("none", "disabled", None):
            return Accelerator()
        raise ValueError(f"Unsupported logging.backend={log_backend}")

    def _init_trackers(self, accelerator, cfg):
        log_backend = cfg.logging.get("backend", "tensorboard")
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

    def _rebuild_for_stage(
        self,
        cfg,
        accelerator,
        train_dataloader,
        val_dataloader,
        stage: int,
        stage_epochs: int,
        scheduler_steps_per_epoch: int,
    ):
        unwrapped_model = accelerator.unwrap_model(self.model)
        unwrapped_model.set_training_stage(stage)
        unwrapped_model.apply_stage_freeze()
        self.current_stage = stage

        optimizer_cfg = self._optimizer_cfg_for_stage(cfg, stage=stage)
        self.optimizer = unwrapped_model.get_optimizer(**optimizer_cfg)
        if self.scaler is not None:
            self.scaler = GradScaler()

        num_steps = (
            scheduler_steps_per_epoch * max(1, stage_epochs)
        ) // cfg.training.gradient_accumulate_every
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=max(1, num_steps),
            last_epoch=-1,
        )
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = (
            accelerator.prepare(
                train_dataloader,
                val_dataloader,
                unwrapped_model,
                self.optimizer,
                lr_scheduler,
            )
        )
        if accelerator.state.num_processes > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                accelerator.unwrap_model(self.model),
                device_ids=[self.model.device],
                find_unused_parameters=True,
            )
        logger.info(f"Rebuilt optimizer/scheduler for stage {stage}")
        return train_dataloader, val_dataloader, lr_scheduler

    def _save_named_checkpoint(self, accelerator, name: str):
        if not accelerator.is_main_process:
            return
        model_ddp = self.model
        self.model = accelerator.unwrap_model(self.model)
        path = os.path.join(self.output_dir, "checkpoints", name)
        self.save_checkpoint(path=path, use_thread=False)
        self.model = model_ddp
        logger.info(f"Saved checkpoint: {path}")

    def _refresh_teacher_from_model(self, accelerator):
        unwrapped_model = accelerator.unwrap_model(self.model)
        self.teacher_model.load_state_dict(unwrapped_model.state_dict(), strict=True)
        self.teacher_model.set_training_stage(3)
        self.teacher_model.apply_stage_freeze()
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        logger.info("Refreshed frozen stage-3 teacher from current model")

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        use_amp = cfg.training.get("use_amp", False)
        amp_dtype = cfg.training.get("amp_dtype", "bf16")
        if use_amp:
            amp_dtype = (
                torch.bfloat16
                if amp_dtype == "bf16" and torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            amp_dtype = None

        accelerator = self._make_accelerator(cfg)
        self._init_trackers(accelerator, cfg)
        cuda_count = accelerator.num_processes

        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                logger.info(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path, strict=False)

        if cfg.training.get("pretrained_ckpt_path", None) is not None:
            ckpt_path = cfg.training.pretrained_ckpt_path
            payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
            self.load_payload(
                payload,
                exclude_keys=["optimizer", "scaler"],
                include_keys=[],
                strict=False,
            )
            logger.info(f"Loaded pretrained checkpoint: {ckpt_path}")

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        val_dataset = dataset.get_validation_dataset()

        dataloader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
        val_dataloader_cfg = OmegaConf.to_container(cfg.val_dataloader, resolve=True)
        dataloader_cfg["batch_size"] = dataloader_cfg["batch_size"] // cuda_count
        val_dataloader_cfg["batch_size"] = val_dataloader_cfg["batch_size"] // cuda_count
        train_dataloader = DataLoader(dataset, **dataloader_cfg)
        val_dataloader = DataLoader(val_dataset, **val_dataloader_cfg)
        # Keep the original, unprepared dataloader length for scheduler step
        # counts. Reusing a prepared dataloader at stage transitions can make
        # the cosine scheduler too short, causing LR to oscillate after it
        # passes num_training_steps.
        scheduler_steps_per_epoch = len(train_dataloader)

        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        self.teacher_model.set_normalizer(normalizer)
        self.ema_model.set_normalizer(normalizer)

        num_steps_stage1 = (
            len(train_dataloader) * max(1, self.stage1_epochs)
        ) // cfg.training.gradient_accumulate_every
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=max(1, num_steps_stage1),
            last_epoch=self.global_step - 1,
        )

        ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        topk_stage3_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.stage3_topk,
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
        if accelerator.state.num_processes > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                accelerator.unwrap_model(self.model),
                device_ids=[self.model.device],
                find_unused_parameters=True,
            )

        device = accelerator.device
        self.teacher_model.to(device)
        self.ema_model.to(device)

        if cfg.training.debug:
            self.stage1_epochs = 1
            self.stage2_epochs = 1
            self.stage3_epochs = 1
            cfg.training.num_epochs = 3
            cfg.training.max_train_steps = 2
            cfg.training.max_val_steps = 2
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1

        stage2_start = self.stage1_epochs
        stage3_start = self.stage1_epochs + self.stage2_epochs
        total_epochs = self.stage1_epochs + self.stage2_epochs + self.stage3_epochs
        cfg.training.num_epochs = total_epochs

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(total_epochs):
                if local_epoch_idx < stage2_start:
                    stage = 1
                    stage_name = "Stage 1 DP"
                elif local_epoch_idx < stage3_start:
                    stage = 2
                    stage_name = "Stage 2 Confidence"
                else:
                    stage = 3
                    stage_name = "Stage 3 50Hz CD"

                unwrapped_model: DPMultiFreqCD50HzConfidencePolicy = accelerator.unwrap_model(self.model)
                unwrapped_model.set_training_stage(stage)
                unwrapped_model.update_progress(local_epoch_idx / max(1, total_epochs))

                self.model.train()
                unwrapped_model.apply_stage_freeze()
                if stage == 3:
                    self.teacher_model.eval()
                if self.ema_model is not None:
                    self.ema_model.set_training_stage(stage)
                    self.ema_model.apply_stage_freeze()
                    self.ema_model.eval()

                logger.info(f"Epoch {self.epoch}: {stage_name}")
                step_log = {}
                train_losses = []
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch} ({stage_name})",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        with autocast(enabled=use_amp, dtype=amp_dtype):
                            if stage == 3:
                                raw_loss, loss_dict = unwrapped_model.compute_loss(
                                    batch,
                                    target_policy=self.ema_model,
                                    teacher_policy=self.teacher_model,
                                )
                            else:
                                raw_loss, loss_dict = unwrapped_model.compute_loss(batch)
                            loss = raw_loss / cfg.training.gradient_accumulate_every

                        if not torch.isfinite(loss):
                            logger.warning(
                                f"Step {self.global_step}: non-finite loss={loss.item()}, skipping"
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
                            max_grad_norm = cfg.training.get("max_grad_norm", 1.0)
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                unwrapped_model.parameters(),
                                max_norm=max_grad_norm,
                                error_if_nonfinite=False,
                            )
                            if not torch.isfinite(grad_norm):
                                logger.warning(
                                    f"Step {self.global_step}: non-finite grad_norm={float(grad_norm)}, skipping"
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
                            if accelerator.is_main_process:
                                ema.step(accelerator.unwrap_model(self.model))

                        raw_loss_cpu = raw_loss.item()
                        train_losses.append(raw_loss_cpu)
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                            "training_stage": float(stage),
                        }
                        for key, value in loss_dict.items():
                            step_log[f"train_{key}"] = value

                        is_last_batch = batch_idx == len(train_dataloader) - 1
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
                    float(np.nanmean(train_losses)) if len(train_losses) > 0 else 0.0
                )

                if (
                    self.epoch % cfg.training.val_every == 0
                    and len(val_dataloader) > 0
                    and accelerator.is_main_process
                ):
                    policy = unwrapped_model if stage == 3 else self.ema_model
                    policy.set_training_stage(stage)
                    policy.eval()
                    with torch.no_grad():
                        val_losses = []
                        for batch_idx, batch in enumerate(val_dataloader):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            with autocast(enabled=use_amp, dtype=amp_dtype):
                                if stage == 3:
                                    val_loss, _ = policy.compute_loss(
                                        batch,
                                        target_policy=self.ema_model,
                                        teacher_policy=self.teacher_model,
                                    )
                                else:
                                    val_loss, _ = policy.compute_loss(batch)
                            val_losses.append(val_loss.item())
                            if (
                                cfg.training.max_val_steps is not None
                                and batch_idx >= cfg.training.max_val_steps - 1
                            ):
                                break
                        if len(val_losses) > 0:
                            step_log["val_loss"] = float(np.nanmean(val_losses))

                if stage == 3 and accelerator.is_main_process:
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)
                    if self.epoch % cfg.training.checkpoint_every == 0:
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
                        topk_path = topk_stage3_manager.get_ckpt_path(metric_dict)
                        if topk_path is not None:
                            self.save_checkpoint(path=topk_path)
                    self.model = model_ddp

                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1

                if local_epoch_idx == self.stage1_epochs - 1 and self.stage2_epochs > 0:
                    self._save_named_checkpoint(accelerator, "stage1_final.ckpt")
                    train_dataloader, val_dataloader, lr_scheduler = self._rebuild_for_stage(
                        cfg,
                        accelerator,
                        train_dataloader,
                        val_dataloader,
                        stage=2,
                        stage_epochs=self.stage2_epochs,
                        scheduler_steps_per_epoch=scheduler_steps_per_epoch,
                    )
                elif local_epoch_idx == stage3_start - 1 and self.stage3_epochs > 0:
                    self._save_named_checkpoint(accelerator, "stage2_final.ckpt")
                    self._refresh_teacher_from_model(accelerator)
                    train_dataloader, val_dataloader, lr_scheduler = self._rebuild_for_stage(
                        cfg,
                        accelerator,
                        train_dataloader,
                        val_dataloader,
                        stage=3,
                        stage_epochs=self.stage3_epochs,
                        scheduler_steps_per_epoch=scheduler_steps_per_epoch,
                    )

                self.epoch += 1

        if accelerator.is_main_process:
            model_ddp = self.model
            self.model = accelerator.unwrap_model(self.model)
            self.save_checkpoint(use_thread=False)
            self.model = model_ddp
            logger.info(f"Saved final checkpoint to {self.get_checkpoint_path()}")

        accelerator.end_training()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainDPMultiFreqCD50HzConfidenceWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
