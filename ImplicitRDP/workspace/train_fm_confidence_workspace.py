"""
Two-stage training workspace for FM + Confidence Head.

Stage 1: Train FM model (velocity matching loss).
Stage 2: Freeze FM, train Confidence Head on one-step FM action MSE.

Uses TensorBoard for logging (via Accelerate).
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
import pickle
import pathlib

import hydra
import torch
import torch.nn.functional as F
import dill
import numpy as np
import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator

from ImplicitRDP.workspace.base_workspace import BaseWorkspace
from ImplicitRDP.policy.fm_confidence_policy import FMConfidencePolicy
from ImplicitRDP.dataset.base_dataset import BaseImageDataset
from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.common.checkpoint_util import TopKCheckpointManager
from ImplicitRDP.common.json_logger import JsonLogger
from ImplicitRDP.model.diffusion.ema_model import EMAModel
from ImplicitRDP.model.common.lr_scheduler import get_scheduler

from loguru import logger

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainFMConfidenceWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: FMConfidencePolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: FMConfidencePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        accelerator = Accelerator()
        cuda_count = accelerator.num_processes
        logger.info(f"Number of CUDA devices: {cuda_count}")
        logger.info(f"Learning rate: {optimizer_cfg['lr']}")

        self.optimizer = self.model.get_optimizer(**optimizer_cfg)

        self.global_step = 0
        self.epoch = 0

        # AMP scaler
        self.scaler = None
        if cfg.training.get("use_amp", False):
            self.scaler = GradScaler()

        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ["optimizer", "scaler"]

        # Two-stage config
        self.stage1_epochs = cfg.training.get("stage1_epochs", 600)
        self.stage2_epochs = cfg.training.get("stage2_epochs", 250)
        self.stage2_dataset_path = cfg.training.get("stage2_dataset_path", None)
        self.current_stage = 1

        logger.info("Two-stage training configuration:")
        logger.info(f"  Stage 1: {self.stage1_epochs} epochs (FM velocity matching)")
        logger.info(f"  Stage 2: {self.stage2_epochs} epochs (Confidence on one-step FM MSE)")

        # Stage 1 dataset config
        self._dataset_stage1_cfg = cfg.task.dataset
        # Stage 2 dataset config (may use a different dataset)
        self._dataset_stage2_cfg = OmegaConf.create(
            OmegaConf.to_container(cfg.task.dataset, resolve=True)
        )
        if self.stage2_dataset_path is not None:
            self._dataset_stage2_cfg["dataset_path"] = self.stage2_dataset_path
            logger.info(f"  Stage 2 dataset: {self.stage2_dataset_path}")
        else:
            logger.info(f"  Stage 2 dataset: same as Stage 1")

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # AMP configuration
        use_amp = cfg.training.get("use_amp", False)
        amp_dtype = cfg.training.get("amp_dtype", "bf16")
        if use_amp:
            if amp_dtype == "bf16" and torch.cuda.is_bf16_supported():
                logger.info("Using bf16 for AMP")
                amp_dtype = torch.bfloat16
            elif amp_dtype == "fp16":
                logger.info("Using fp16 for AMP")
                amp_dtype = torch.float16
            else:
                logger.warning("AMP dtype not supported, falling back to fp16")
                amp_dtype = torch.float16
        else:
            amp_dtype = None

        # Accelerator with TensorBoard
        tb_log_dir = os.path.join(self.output_dir, "tb_logs")
        accelerator = Accelerator(log_with="tensorboard", project_dir=tb_log_dir)
        cuda_count = accelerator.num_processes

        accelerator.init_trackers(
            project_name=cfg.logging.project,
        )

        # Resume training
        if cfg.training.resume:
            assert cfg.training.pretrained_ckpt_path is None
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                logger.info(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # Load pretrained checkpoint
        if cfg.training.pretrained_ckpt_path is not None:
            ckpt_path = cfg.training.pretrained_ckpt_path
            payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
            for key in self.model.state_dict():
                if "normalizer" in key:
                    payload["state_dicts"]["model"][key] = self.model.state_dict()[key]
                    if self.ema_model is not None:
                        payload["state_dicts"]["ema_model"][key] = self.model.state_dict()[key]
            self.load_payload(
                payload,
                exclude_keys=["optimizer", "scaler"],
                include_keys=[],
                strict=False,
            )
            logger.info(f"Loaded model from checkpoint: {ckpt_path}")

        # Configure dataset (Stage 1)
        dataset: BaseImageDataset = hydra.utils.instantiate(self._dataset_stage1_cfg)
        assert isinstance(dataset, BaseImageDataset)

        # Pre-build Stage 2 dataset
        dataset_stage2: BaseImageDataset = hydra.utils.instantiate(
            self._dataset_stage2_cfg
        )
        assert isinstance(dataset_stage2, BaseImageDataset)

        dataloader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
        val_dataloader_cfg = OmegaConf.to_container(cfg.val_dataloader, resolve=True)
        logger.info(f"Original batch size: {dataloader_cfg['batch_size']}")
        dataloader_cfg["batch_size"] = dataloader_cfg["batch_size"] // cuda_count
        val_dataloader_cfg["batch_size"] = val_dataloader_cfg["batch_size"] // cuda_count
        logger.info(f"Per-GPU batch size: {dataloader_cfg['batch_size']}")

        train_dataloader = DataLoader(dataset, **dataloader_cfg)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **val_dataloader_cfg)

        # Compute normalizer
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # LR scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        # EMA
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # Checkpoint manager
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        # Accelerator prepare
        (
            train_dataloader,
            val_dataloader,
            self.model,
            self.optimizer,
            lr_scheduler,
        ) = accelerator.prepare(
            train_dataloader,
            val_dataloader,
            self.model,
            self.optimizer,
            lr_scheduler,
        )
        if accelerator.state.num_processes > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                accelerator.unwrap_model(self.model),
                device_ids=[self.model.device],
                find_unused_parameters=True,
            )

        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # ==================== Training loop ====================
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        logger.info("Starting two-stage training...")

        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                # Determine current stage
                if local_epoch_idx < self.stage1_epochs:
                    self.current_stage = 1
                    stage_name = "Stage 1 (FM velocity matching)"
                else:
                    self.current_stage = 2
                    stage_name = "Stage 2 (Confidence training)"

                unwrapped_model = accelerator.unwrap_model(self.model)
                unwrapped_model.update_progress(
                    local_epoch_idx / cfg.training.num_epochs
                )

                if hasattr(unwrapped_model, "set_training_stage"):
                    unwrapped_model.set_training_stage(self.current_stage)

                if (
                    local_epoch_idx % 10 == 0
                    or local_epoch_idx == self.stage1_epochs
                ):
                    logger.info(f"Epoch {local_epoch_idx}: {stage_name}")

                self.model.train()

                if cfg.training.freeze_encoder and self.current_stage == 1:
                    unwrapped_model.obs_encoder.eval()
                    unwrapped_model.obs_encoder.requires_grad_(False)

                step_log = dict()
                train_losses = list()

                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch} ({stage_name})",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(
                            batch, lambda x: x.to(device, non_blocking=True)
                        )
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        train_sampling_batch = batch

                        with autocast(enabled=use_amp, dtype=amp_dtype):
                            raw_loss, loss_dict = unwrapped_model.compute_loss(batch)
                            loss = raw_loss / cfg.training.gradient_accumulate_every

                        if use_amp:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        if (
                            self.global_step
                            % cfg.training.gradient_accumulate_every
                            == 0
                        ):
                            if use_amp:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        if (
                            cfg.training.use_ema
                            and accelerator.is_main_process
                        ):
                            ema.step(accelerator.unwrap_model(self.model))

                        # Logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                            "training_stage": float(self.current_stage),
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

                # Epoch average
                train_loss = np.nanmean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= Validation ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                if (
                    (self.epoch % cfg.training.val_every) == 0
                    and len(val_dataloader) > 0
                    and accelerator.is_main_process
                ):
                    with torch.no_grad():
                        val_losses = list()
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
                                tepoch.set_postfix(
                                    loss=raw_loss.item(), refresh=False
                                )
                                if (
                                    cfg.training.max_val_steps is not None
                                    and batch_idx
                                    >= cfg.training.max_val_steps - 1
                                ):
                                    break
                        if len(val_losses) > 0:
                            val_losses_t = torch.tensor(val_losses)
                            valid_mask = ~torch.isnan(val_losses_t)
                            if valid_mask.any():
                                val_loss = torch.mean(val_losses_t[valid_mask]).item()
                            else:
                                val_loss = float('nan')
                            step_log["val_loss"] = val_loss
                            logger.info(
                                f"Epoch {self.epoch}, val_loss: {val_loss:.6f}"
                            )

                # ========= Sampling ==========
                if (
                    (self.epoch % cfg.training.sample_every) == 0
                    and accelerator.is_main_process
                    and self.current_stage == 1
                ):
                    with torch.no_grad():
                        batch = dict_apply(
                            train_sampling_batch,
                            lambda x: x.to(device, non_blocking=True),
                        )
                        obs_dict = batch["obs"]
                        extended_obs_dict = batch.get("extended_obs", None)
                        gt_action = batch["action"]

                        dataset_obs_temporal_downsample_ratio = cfg.task.dataset.get(
                            "obs_temporal_downsample_ratio", 1
                        )
                        result = policy.predict_action(
                            obs_dict,
                            dataset_obs_temporal_downsample_ratio=dataset_obs_temporal_downsample_ratio,
                            extended_obs_dict=extended_obs_dict,
                        )
                        pred_action = result["action_pred"]

                        # Align shapes
                        min_t = min(pred_action.shape[1], gt_action.shape[1])
                        pred_6 = pred_action[:, :min_t, :6]
                        gt_6 = gt_action[:, :min_t, :6]
                        step_log["train_action_mse_error"] = F.mse_loss(
                            pred_6, gt_6
                        ).item()

                        del batch, gt_action, pred_action

                # ========= Checkpoint ==========
                if (
                    (self.epoch % cfg.training.checkpoint_every) == 0
                    and accelerator.is_main_process
                ):
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.get("save_last_snapshot", False):
                        self.save_snapshot()

                    metric_dict = {
                        k.replace("/", "_"): v for k, v in step_log.items()
                    }
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    self.model = model_ddp

                # ========= Stage transition: 1 -> 2 ==========
                if (
                    self.stage2_epochs > 0
                    and local_epoch_idx == self.stage1_epochs - 1
                ):
                    logger.info(
                        "Stage 1 completed! Transitioning to Stage 2..."
                    )

                    # Save Stage 1 checkpoint
                    if accelerator.is_main_process:
                        stage1_ckpt_path = os.path.join(
                            self.output_dir, "checkpoints", "stage1_final.ckpt"
                        )
                        model_ddp = self.model
                        self.model = accelerator.unwrap_model(self.model)
                        self.save_checkpoint(path=stage1_ckpt_path)
                        self.model = model_ddp
                        logger.info(
                            f"Saved Stage 1 checkpoint to {stage1_ckpt_path}"
                        )

                    # Switch dataset if needed
                    if (
                        self.stage2_dataset_path is not None
                    ):
                        logger.info(
                            f"Switching to Stage 2 dataset: {self.stage2_dataset_path}"
                        )
                        new_train_dataset = dataset_stage2
                        new_val_dataset = dataset_stage2.get_validation_dataset()
                        new_train_dataloader = DataLoader(
                            new_train_dataset, **dataloader_cfg
                        )
                        new_val_dataloader = DataLoader(
                            new_val_dataset, **val_dataloader_cfg
                        )
                    else:
                        new_train_dataloader = train_dataloader
                        new_val_dataloader = val_dataloader

                    # Rebuild optimizer for Stage 2
                    logger.info("Rebuilding optimizer for Stage 2...")
                    optimizer_cfg_dict = OmegaConf.to_container(
                        cfg.optimizer, resolve=True
                    )
                    unwrapped_model = accelerator.unwrap_model(self.model)
                    unwrapped_model.set_training_stage(2)
                    self.optimizer = unwrapped_model.get_optimizer(
                        **optimizer_cfg_dict
                    )

                    # Reset GradScaler
                    if self.scaler is not None:
                        self.scaler = GradScaler()

                    # Rebuild lr_scheduler for Stage 2
                    stage2_train_dl = (
                        new_train_dataloader
                        if self.stage2_dataset_path is not None
                        else train_dataloader
                    )
                    num_steps_stage2 = (
                        len(stage2_train_dl) * self.stage2_epochs
                    ) // cfg.training.gradient_accumulate_every
                    lr_scheduler_stage2 = get_scheduler(
                        cfg.training.lr_scheduler,
                        optimizer=self.optimizer,
                        num_warmup_steps=cfg.training.lr_warmup_steps,
                        num_training_steps=num_steps_stage2,
                        last_epoch=-1,
                    )

                    # Re-prepare with accelerator
                    (
                        train_dataloader,
                        val_dataloader,
                        self.model,
                        self.optimizer,
                        lr_scheduler,
                    ) = accelerator.prepare(
                        stage2_train_dl
                        if self.stage2_dataset_path is not None
                        else train_dataloader,
                        new_val_dataloader
                        if self.stage2_dataset_path is not None
                        else val_dataloader,
                        unwrapped_model,
                        self.optimizer,
                        lr_scheduler_stage2,
                    )

                    if accelerator.state.num_processes > 1:
                        self.model = torch.nn.parallel.DistributedDataParallel(
                            accelerator.unwrap_model(self.model),
                            device_ids=[self.model.device],
                            find_unused_parameters=True,
                        )

                    logger.info(
                        "Optimizer and lr_scheduler rebuilt for Stage 2!"
                    )

                # End of epoch
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()
        logger.info("Training completed!")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainFMConfidenceWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
