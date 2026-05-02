"""
Training workspace for multi-frequency diffusion-policy distillation.

Stage 1:
    Standard multi-frequency diffusion-policy training.

Stage 2:
    One-step DDIM-jump distillation. A frozen copy of the stage-1 DP backbone
    acts as teacher; an EMA copy of the stop-time-conditioned student acts as
    the consistency target.
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
from ImplicitRDP.model.diffusion.ema_model import EMAModel
from ImplicitRDP.policy.dp_multifreq_policy import DPMultiFreqPolicy
from ImplicitRDP.workspace.base_workspace import BaseWorkspace

from loguru import logger

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDPMultiFreqWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: DPMultiFreqPolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: DPMultiFreqPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        else:
            raise ValueError("DP multi-frequency training requires training.use_ema=True")

        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop("stage2_lr", None)
        optimizer_cfg.pop("stage2_obs_encoder_lr", None)
        self.optimizer = self.model.get_optimizer(**optimizer_cfg)

        self.global_step = 0
        self.epoch = 0
        self.scaler = None
        if cfg.training.get("use_amp", False):
            self.scaler = GradScaler()

        if not cfg.training.resume:
            self.exclude_keys = ["optimizer", "scaler"]

        self.current_stage = 1
        self.stage2_ema = None  # Will be created when entering stage 2
        self.stage1_epochs = cfg.training.get("stage1_epochs", 600)
        self.stage2_epochs = cfg.training.get("stage2_epochs", 200)
        logger.info(
            "Two-stage DP multi-frequency training: "
            f"stage1_epochs={self.stage1_epochs}, stage2_epochs={self.stage2_epochs}"
        )

    def _enter_stage2(self, cfg, accelerator, train_dataloader, val_dataloader, device):
        unwrapped_model: DPMultiFreqPolicy = accelerator.unwrap_model(self.model)

        teacher_model = copy.deepcopy(unwrapped_model.model)
        teacher_model.requires_grad_(False)
        teacher_model.eval()
        unwrapped_model.set_teacher_model(teacher_model)
        unwrapped_model.prepare_stage2_student()
        unwrapped_model.model.requires_grad_(False)
        unwrapped_model.model.eval()
        # Freeze obs_encoder in stage 2: visual features are already learned,
        # only train ctm_model (student) and RNN encoders
        unwrapped_model.obs_encoder.requires_grad_(False)
        unwrapped_model.obs_encoder.eval()
        unwrapped_model.set_training_stage(2)

        if self.ema_model is not None:
            self.ema_model.load_state_dict(unwrapped_model.state_dict(), strict=False)
            self.ema_model.to(device)
            self.ema_model.prepare_stage2_student()
            self.ema_model.model.requires_grad_(False)
            self.ema_model.model.eval()
            self.ema_model.obs_encoder.requires_grad_(False)
            self.ema_model.obs_encoder.eval()
            self.ema_model.set_training_stage(2)
            unwrapped_model.set_ema_model_ref(self.ema_model.ctm_model)
            self.ema_model.set_teacher_model(teacher_model)
            self.ema_model.set_ema_model_ref(self.ema_model.ctm_model)

        self.current_stage = 2

        # Reset EMA for stage 2 with CTM-specific initial decay
        # (low initial decay = EMA closely tracks student early on)
        ctm_ema_initial_decay = getattr(unwrapped_model, "ctm_ema_initial_decay", 0.0)
        ema_cfg = OmegaConf.to_container(cfg.ema, resolve=True)
        ema_cfg["min_value"] = ctm_ema_initial_decay
        self.stage2_ema = EMAModel(model=self.ema_model, **{
            k: v for k, v in ema_cfg.items() if k != "_target_"
        })
        # Reset optimization_step so decay ramp starts fresh for stage 2
        self.stage2_ema.optimization_step = 0
        logger.info(
            f"Stage 2 EMA reset: initial_decay={ctm_ema_initial_decay}, "
            f"will ramp up to max_value={ema_cfg.get('max_value', 0.9999)}"
        )

        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        stage2_lr = optimizer_cfg.pop("stage2_lr", 1.0e-5)
        # Remove obs_encoder keys — obs_encoder is frozen in stage 2,
        # get_optimizer skips it, so these would be unused dead config.
        optimizer_cfg.pop("stage2_obs_encoder_lr", None)
        optimizer_cfg.pop("obs_encoder_lr", None)
        optimizer_cfg["lr"] = stage2_lr
        self.optimizer = unwrapped_model.get_optimizer(**optimizer_cfg)

        if self.scaler is not None:
            self.scaler = GradScaler()

        num_training_steps = (
            len(train_dataloader) * self.stage2_epochs
        ) // cfg.training.gradient_accumulate_every
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=max(1, num_training_steps),
            last_epoch=-1,
        )

        prepared = accelerator.prepare(
            train_dataloader,
            val_dataloader,
            unwrapped_model,
            self.optimizer,
            lr_scheduler,
        )
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = prepared

        if accelerator.state.num_processes > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                accelerator.unwrap_model(self.model),
                device_ids=[self.model.device],
                find_unused_parameters=True,
            )

        logger.info(
            f"Entered Stage 2 DDIM distillation with lr={stage2_lr}, "
            f"obs_encoder=frozen"
        )
        return train_dataloader, val_dataloader, lr_scheduler

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
        self.ema_model.set_normalizer(normalizer)

        total_epochs = self.stage1_epochs + self.stage2_epochs
        # Stage 1 LR scheduler: only covers stage1_epochs so cosine decay
        # completes fully before entering stage 2 (which creates its own scheduler)
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * max(1, self.stage1_epochs)
            ) // cfg.training.gradient_accumulate_every,
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
        if accelerator.state.num_processes > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                accelerator.unwrap_model(self.model),
                device_ids=[self.model.device],
                find_unused_parameters=True,
            )

        device = accelerator.device
        self.ema_model.to(device)

        if cfg.training.debug:
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            self.stage1_epochs = min(self.stage1_epochs, 1)
            self.stage2_epochs = min(self.stage2_epochs, 1)
            total_epochs = self.stage1_epochs + self.stage2_epochs

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        logger.info("Starting DP multi-frequency training...")

        stage2_started = False
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(total_epochs):
                if local_epoch_idx < self.stage1_epochs:
                    self.current_stage = 1
                else:
                    if not stage2_started:
                        if accelerator.is_main_process and self.stage1_epochs > 0:
                            stage1_path = os.path.join(
                                self.output_dir, "checkpoints", "stage1_final.ckpt"
                            )
                            model_ddp = self.model
                            self.model = accelerator.unwrap_model(self.model)
                            self.save_checkpoint(path=stage1_path)
                            self.model = model_ddp
                            logger.info(f"Saved Stage-1 checkpoint to {stage1_path}")
                        train_dataloader, val_dataloader, lr_scheduler = self._enter_stage2(
                            cfg, accelerator, train_dataloader, val_dataloader, device
                        )
                        stage2_started = True
                    self.current_stage = 2

                unwrapped_model: DPMultiFreqPolicy = accelerator.unwrap_model(self.model)
                unwrapped_model.set_training_stage(self.current_stage)
                unwrapped_model.update_progress(local_epoch_idx / max(1, total_epochs))
                if self.current_stage == 2:
                    unwrapped_model.set_ema_model_ref(self.ema_model.ctm_model)
                    unwrapped_model.model.eval()
                    unwrapped_model.model.requires_grad_(False)

                self.model.train()
                if cfg.training.freeze_encoder and self.current_stage == 1:
                    unwrapped_model.obs_encoder.eval()
                    unwrapped_model.obs_encoder.requires_grad_(False)
                if self.current_stage == 2:
                    # Stage 2: obs_encoder is frozen (visual features already learned)
                    # Must re-apply after self.model.train() which enables train mode on all submodules
                    unwrapped_model.obs_encoder.eval()
                    unwrapped_model.obs_encoder.requires_grad_(False)

                step_log = {}
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
                            # Use stage2-specific EMA (with low initial decay) during distillation
                            if self.current_stage == 2 and self.stage2_ema is not None:
                                self.stage2_ema.step(accelerator.unwrap_model(self.model))
                            else:
                                ema.step(accelerator.unwrap_model(self.model))

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

                step_log["train_loss"] = (
                    np.nanmean(train_losses) if len(train_losses) > 0 else 0.0
                )

                policy = self.ema_model
                policy.eval()
                if self.current_stage == 2:
                    policy.set_training_stage(2)
                    # NOTE: do NOT call policy.set_ema_model_ref for validation.
                    # Stage 2 validation uses DSM-only loss (pred vs ground truth)
                    # instead of full CTM loss (which would compare EMA vs EMA).
                else:
                    policy.set_training_stage(1)

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
                                    if self.current_stage == 2:
                                        # Stage 2: compute DSM-only loss as validation
                                        # metric. This measures how well the student
                                        # predicts x0 from noisy input vs ground truth,
                                        # avoiding the meaningless CTM(EMA vs EMA) comparison.
                                        raw_loss = policy.compute_val_loss_stage2(batch)
                                    else:
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
            final_path = os.path.join(self.output_dir, "checkpoints", "dp_multifreq_final.ckpt")
            model_ddp = self.model
            self.model = accelerator.unwrap_model(self.model)
            self.save_checkpoint(path=final_path)
            self.model = model_ddp
            logger.info(f"Saved final DP multi-frequency checkpoint to {final_path}")

        accelerator.end_training()
        logger.info("DP multi-frequency training completed")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainDPMultiFreqWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
