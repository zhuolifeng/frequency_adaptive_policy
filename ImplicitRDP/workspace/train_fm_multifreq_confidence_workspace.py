"""
Three-stage training workspace for Multi-frequency FM + Confidence Policy.

Stage 1: FM velocity matching (10hz + 50hz, distill=0)
Stage 2: 10hz FM + 50hz Consistency Distillation
Stage 3: Freeze FM, train Confidence Head on 10hz one-step FM MSE
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
import torch.nn.functional as F
import dill
import numpy as np
import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator

from ImplicitRDP.workspace.base_workspace import BaseWorkspace
from ImplicitRDP.policy.fm_multifreq_confidence_policy import FMMultiFreqConfidencePolicy
from ImplicitRDP.dataset.base_dataset import BaseImageDataset
from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.common.checkpoint_util import TopKCheckpointManager
from ImplicitRDP.common.json_logger import JsonLogger
from ImplicitRDP.model.diffusion.ema_model import EMAModel
from ImplicitRDP.model.common.lr_scheduler import get_scheduler

from loguru import logger

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainFMMultiFreqConfidenceWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: FMMultiFreqConfidencePolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: FMMultiFreqConfidencePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop("stage2_lr", None)
        optimizer_cfg.pop("stage2_obs_encoder_lr", None)
        accelerator = Accelerator()
        cuda_count = accelerator.num_processes
        logger.info(f"Number of CUDA devices: {cuda_count}")

        self.optimizer = self.model.get_optimizer(**optimizer_cfg)

        self.global_step = 0
        self.epoch = 0

        self.scaler = None
        if cfg.training.get("use_amp", False):
            self.scaler = GradScaler()

        if not cfg.training.resume:
            self.exclude_keys = ["optimizer", "scaler"]

        # Three-stage config
        self.stage1_epochs = cfg.training.get("stage1_epochs", 600)
        self.stage2_epochs = cfg.training.get("stage2_epochs", 200)
        self.stage3_epochs = cfg.training.get("stage3_epochs", 250)
        self.stage2_dataset_path = cfg.training.get("stage2_dataset_path", None)
        self.current_stage = 1

        stage2_lr = cfg.optimizer.get("stage2_lr", 1e-5)
        stage2_obs_encoder_lr = cfg.optimizer.get("stage2_obs_encoder_lr", 3e-6)
        stage2_ema_rate = cfg.training.get("stage2_ema_rate", 0.9999)
        logger.info("Three-stage training configuration:")
        logger.info(f"  Stage 1: {self.stage1_epochs} epochs (FM velocity matching)")
        logger.info(f"  Stage 2: {self.stage2_epochs} epochs (CD distillation)")
        logger.info(f"    Stage 2 lr={stage2_lr}, obs_encoder_lr={stage2_obs_encoder_lr}, ema_rate={stage2_ema_rate}")
        logger.info(f"  Stage 3: {self.stage3_epochs} epochs (Confidence Head)")

        self._dataset_stage1_cfg = cfg.task.dataset
        self._dataset_stage2_cfg = OmegaConf.create(
            OmegaConf.to_container(cfg.task.dataset, resolve=True)
        )
        if self.stage2_dataset_path is not None:
            self._dataset_stage2_cfg["dataset_path"] = self.stage2_dataset_path

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
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                logger.info(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        if cfg.training.pretrained_ckpt_path is not None:
            ckpt_path = cfg.training.pretrained_ckpt_path
            payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)

            # Handle loading from FMMultiFreqPolicy ckpt (no confidence head):
            # Filter out missing keys so strict=False works cleanly
            model_state = self.model.state_dict()
            ckpt_model_state = payload["state_dicts"].get("model", {})

            # Check if checkpoint is from a policy without confidence head
            has_confidence_in_ckpt = any(
                "confidence_head" in k for k in ckpt_model_state.keys()
            )
            if not has_confidence_in_ckpt:
                logger.warning(
                    "Checkpoint has NO confidence_head weights! "
                    "Loading FM backbone only, confidence head will be randomly initialized."
                )

            # Override normalizer keys with current model's normalizer
            for key in model_state:
                if "normalizer" in key:
                    payload["state_dicts"]["model"][key] = model_state[key]
                    if self.ema_model is not None and "ema_model" in payload["state_dicts"]:
                        payload["state_dicts"]["ema_model"][key] = model_state[key]

            # If ema_model checkpoint exists but was from a different policy class,
            # also handle missing keys gracefully
            if self.ema_model is not None and "ema_model" in payload["state_dicts"]:
                ema_ckpt_state = payload["state_dicts"]["ema_model"]
                if not any("confidence_head" in k for k in ema_ckpt_state.keys()):
                    logger.warning(
                        "EMA checkpoint also has NO confidence_head weights, "
                        "will be randomly initialized."
                    )

            self.load_payload(payload, exclude_keys=["optimizer", "scaler"], include_keys=[], strict=False)
            logger.info(f"Loaded model from checkpoint: {ckpt_path}")

        # ============================================================
        # Direct Stage 3: skip stage1 + stage2, train confidence only
        # ============================================================
        direct_stage3 = (self.stage1_epochs == 0 and self.stage2_epochs == 0 and self.stage3_epochs > 0)
        if direct_stage3:
            logger.info(
                "=" * 60 + "\n"
                "  stage1_epochs=0, stage2_epochs=0: DIRECT Stage 3 mode\n"
                "  Loading pretrained FM backbone, training confidence head only\n"
                + "=" * 60
            )
            self.current_stage = 3
            self.model.set_training_stage(3)

            # Freeze all non-confidence-head modules
            frozen_params = 0
            for module_name in ['model', 'obs_encoder']:
                module = getattr(self.model, module_name, None)
                if module is not None:
                    module.requires_grad_(False)
                    module.eval()
                    frozen_params += sum(p.numel() for p in module.parameters())
            for rnn_name in ['rnn_lowfreq', 'rnn_highfreq']:
                rnn = getattr(self.model, rnn_name, None)
                if rnn is not None:
                    rnn.requires_grad_(False)
                    rnn.eval()
                    frozen_params += sum(p.numel() for p in rnn.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            logger.info(
                f"Stage 3 direct init: Frozen {frozen_params:,} params, "
                f"trainable {trainable_params:,} params (confidence head only)"
            )

            # Rebuild optimizer for confidence head only
            optimizer_cfg_dict = OmegaConf.to_container(cfg.optimizer, resolve=True)
            # Remove stage2-specific keys if present
            optimizer_cfg_dict.pop("stage2_lr", None)
            optimizer_cfg_dict.pop("stage2_obs_encoder_lr", None)
            self.optimizer = self.model.get_optimizer(**optimizer_cfg_dict)
            if self.scaler is not None:
                self.scaler = GradScaler()
            logger.info("Optimizer rebuilt for Stage 3 (confidence head only)")

            # Sync model weights to ema_model so EMA starts from the
            # same state (especially important for confidence_head which
            # may be randomly initialized and not present in the ckpt)
            if self.ema_model is not None:
                self.ema_model.load_state_dict(self.model.state_dict())
                logger.info("Synced model weights to ema_model for Stage 3")

        # Datasets
        dataset: BaseImageDataset = hydra.utils.instantiate(self._dataset_stage1_cfg)
        assert isinstance(dataset, BaseImageDataset)
        dataset_stage2: BaseImageDataset = hydra.utils.instantiate(self._dataset_stage2_cfg)

        dataloader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
        val_dataloader_cfg = OmegaConf.to_container(cfg.val_dataloader, resolve=True)
        dataloader_cfg["batch_size"] = dataloader_cfg["batch_size"] // cuda_count
        val_dataloader_cfg["batch_size"] = val_dataloader_cfg["batch_size"] // cuda_count

        train_dataloader = DataLoader(dataset, **dataloader_cfg)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **val_dataloader_cfg)

        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        (train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler) = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler,
        )
        if accelerator.state.num_processes > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                accelerator.unwrap_model(self.model), device_ids=[self.model.device], find_unused_parameters=True,
            )

        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 4
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            self.stage1_epochs = 1
            self.stage2_epochs = 1
            self.stage3_epochs = 1
            cfg.training.num_epochs = 4

        # Compute stage boundaries
        stage2_start = self.stage1_epochs
        stage3_start = self.stage1_epochs + self.stage2_epochs

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        logger.info("Starting three-stage training...")

        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                # Determine stage
                if local_epoch_idx < stage2_start:
                    self.current_stage = 1
                    stage_name = "Stage 1 (FM velocity matching)"
                elif local_epoch_idx < stage3_start:
                    self.current_stage = 2
                    stage_name = "Stage 2 (CD distillation)"
                else:
                    self.current_stage = 3
                    stage_name = "Stage 3 (Confidence Head)"

                unwrapped_model = accelerator.unwrap_model(self.model)
                unwrapped_model.update_progress(local_epoch_idx / cfg.training.num_epochs)

                if hasattr(unwrapped_model, "set_training_stage"):
                    unwrapped_model.set_training_stage(self.current_stage)

                if local_epoch_idx % 10 == 0 or local_epoch_idx in (stage2_start, stage3_start):
                    logger.info(f"Epoch {local_epoch_idx}: {stage_name}")

                self.model.train()

                if cfg.training.freeze_encoder and self.current_stage == 1:
                    unwrapped_model.obs_encoder.eval()
                    unwrapped_model.obs_encoder.requires_grad_(False)

                # Stage 3: keep frozen modules in eval mode
                # (self.model.train() above resets all submodules to train)
                if self.current_stage == 3:
                    for module_name in ['model', 'obs_encoder']:
                        module = getattr(unwrapped_model, module_name, None)
                        if module is not None:
                            module.eval()
                    for rnn_name in ['rnn_lowfreq', 'rnn_highfreq']:
                        rnn = getattr(unwrapped_model, rnn_name, None)
                        if rnn is not None:
                            rnn.eval()

                step_log = dict()
                train_losses = list()

                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch} ({stage_name})",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        train_sampling_batch = batch

                        with autocast(enabled=use_amp, dtype=amp_dtype):
                            raw_loss, loss_dict = unwrapped_model.compute_loss(batch)
                            loss = raw_loss / cfg.training.gradient_accumulate_every

                        # Skip batch if loss is NaN/Inf
                        if not torch.isfinite(loss):
                            logger.warning(f"Step {self.global_step}: loss is {loss.item()}, skipping batch")
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

                            if cfg.training.use_ema and accelerator.is_main_process:
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

                        if cfg.training.max_train_steps is not None and batch_idx >= cfg.training.max_train_steps - 1:
                            break

                train_loss = np.nanmean(train_losses) if len(train_losses) > 0 else 0.0
                step_log["train_loss"] = train_loss

                # Validation
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                if (self.epoch % cfg.training.val_every) == 0 and len(val_dataloader) > 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                with autocast(enabled=use_amp, dtype=amp_dtype):
                                    raw_loss, _ = policy.compute_loss(batch)
                                val_losses.append(raw_loss.item())
                                if cfg.training.max_val_steps is not None and batch_idx >= cfg.training.max_val_steps - 1:
                                    break
                        if len(val_losses) > 0:
                            val_losses_t = torch.tensor(val_losses)
                            valid_mask = ~torch.isnan(val_losses_t)
                            if valid_mask.any():
                                val_loss = torch.mean(val_losses_t[valid_mask]).item()
                            else:
                                val_loss = float('nan')
                            step_log["val_loss"] = val_loss

                # Checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and accelerator.is_main_process:
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                    self.model = model_ddp

                # ========= Stage 1 → Stage 2 transition =========
                if local_epoch_idx == stage2_start - 1 and self.stage2_epochs > 0:
                    logger.info("Stage 1 completed! Transitioning to Stage 2 (CD distillation)...")

                    if accelerator.is_main_process:
                        stage1_ckpt_path = os.path.join(self.output_dir, "checkpoints", "stage1_final.ckpt")
                        model_ddp = self.model
                        self.model = accelerator.unwrap_model(self.model)
                        self.save_checkpoint(path=stage1_ckpt_path)
                        self.model = model_ddp
                        logger.info(f"Saved Stage 1 checkpoint to {stage1_ckpt_path}")

                    unwrapped_model = accelerator.unwrap_model(self.model)

                    # Create frozen teacher
                    teacher_model = copy.deepcopy(unwrapped_model.model)
                    teacher_model.requires_grad_(False)
                    teacher_model.eval()
                    unwrapped_model.set_teacher_model(teacher_model)
                    logger.info("Created frozen teacher model for CD")

                    # Set ema_model as CD Target
                    if self.ema_model is not None:
                        unwrapped_model.set_ema_model_ref(self.ema_model.model)
                        logger.info("Set ema_model as CD Target")

                    unwrapped_model.set_training_stage(2)

                    # Rebuild optimizer with lower lr for CD stability
                    optimizer_cfg_dict = OmegaConf.to_container(cfg.optimizer, resolve=True)
                    stage2_lr = optimizer_cfg_dict.pop("stage2_lr", 1e-5)
                    stage2_obs_encoder_lr = optimizer_cfg_dict.pop("stage2_obs_encoder_lr", 3e-6)
                    optimizer_cfg_dict["lr"] = stage2_lr
                    optimizer_cfg_dict["obs_encoder_lr"] = stage2_obs_encoder_lr
                    logger.info(f"Stage 2 optimizer: lr={stage2_lr}, obs_encoder_lr={stage2_obs_encoder_lr}")
                    self.optimizer = unwrapped_model.get_optimizer(**optimizer_cfg_dict)
                    if self.scaler is not None:
                        self.scaler = GradScaler()

                    num_steps_stage2 = (len(train_dataloader) * self.stage2_epochs) // cfg.training.gradient_accumulate_every
                    lr_scheduler_stage2 = get_scheduler(
                        cfg.training.lr_scheduler, optimizer=self.optimizer,
                        num_warmup_steps=cfg.training.lr_warmup_steps,
                        num_training_steps=num_steps_stage2, last_epoch=-1,
                    )

                    (train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler) = accelerator.prepare(
                        train_dataloader, val_dataloader, unwrapped_model, self.optimizer, lr_scheduler_stage2,
                    )
                    if accelerator.state.num_processes > 1:
                        self.model = torch.nn.parallel.DistributedDataParallel(
                            accelerator.unwrap_model(self.model), device_ids=[self.model.device], find_unused_parameters=True,
                        )
                    logger.info("Optimizer and lr_scheduler rebuilt for Stage 2!")

                    # Reset EMA to fixed rate for CD stability
                    stage2_ema_rate = cfg.training.get("stage2_ema_rate", 0.9999)
                    if ema is not None:
                        ema.min_value = stage2_ema_rate
                        ema.max_value = stage2_ema_rate
                        ema.power = 1.0       # disable warmup schedule
                        ema.inv_gamma = 1.0
                        ema.decay = stage2_ema_rate
                        logger.info(f"EMA reset to fixed rate={stage2_ema_rate} for Stage 2 (CD Target)")

                # ========= Stage 2 → Stage 3 transition =========
                if local_epoch_idx == stage3_start - 1 and self.stage3_epochs > 0:
                    logger.info("Stage 2 completed! Transitioning to Stage 3 (Confidence Head)...")

                    if accelerator.is_main_process:
                        stage2_ckpt_path = os.path.join(self.output_dir, "checkpoints", "stage2_final.ckpt")
                        model_ddp = self.model
                        self.model = accelerator.unwrap_model(self.model)
                        self.save_checkpoint(path=stage2_ckpt_path)
                        self.model = model_ddp
                        logger.info(f"Saved Stage 2 checkpoint to {stage2_ckpt_path}")

                    unwrapped_model = accelerator.unwrap_model(self.model)

                    # Free teacher (release GPU memory)
                    unwrapped_model.set_teacher_model(None)
                    unwrapped_model.set_ema_model_ref(None)
                    logger.info("Freed frozen teacher model")

                    unwrapped_model.set_training_stage(3)

                    # Freeze all non-confidence-head modules
                    frozen_params = 0
                    for module_name in ['model', 'obs_encoder']:
                        module = getattr(unwrapped_model, module_name, None)
                        if module is not None:
                            module.requires_grad_(False)
                            module.eval()
                            frozen_params += sum(p.numel() for p in module.parameters())
                    for rnn_name in ['rnn_lowfreq', 'rnn_highfreq']:
                        rnn = getattr(unwrapped_model, rnn_name, None)
                        if rnn is not None:
                            rnn.requires_grad_(False)
                            rnn.eval()
                            frozen_params += sum(p.numel() for p in rnn.parameters())
                    trainable_params = sum(p.numel() for p in unwrapped_model.parameters() if p.requires_grad)
                    logger.info(f"Stage 3: Frozen {frozen_params:,} params, trainable {trainable_params:,} params (confidence head only)")

                    # Rebuild optimizer (only confidence head)
                    optimizer_cfg_dict = OmegaConf.to_container(cfg.optimizer, resolve=True)
                    self.optimizer = unwrapped_model.get_optimizer(**optimizer_cfg_dict)
                    if self.scaler is not None:
                        self.scaler = GradScaler()

                    # Switch dataset if needed
                    if self.stage2_dataset_path is not None:
                        new_train_dataset = dataset_stage2
                        new_val_dataset = dataset_stage2.get_validation_dataset()
                        new_train_dataloader = DataLoader(new_train_dataset, **dataloader_cfg)
                        new_val_dataloader = DataLoader(new_val_dataset, **val_dataloader_cfg)
                    else:
                        new_train_dataloader = train_dataloader
                        new_val_dataloader = val_dataloader

                    num_steps_stage3 = (len(new_train_dataloader if self.stage2_dataset_path else train_dataloader) * self.stage3_epochs) // cfg.training.gradient_accumulate_every
                    lr_scheduler_stage3 = get_scheduler(
                        cfg.training.lr_scheduler, optimizer=self.optimizer,
                        num_warmup_steps=cfg.training.lr_warmup_steps,
                        num_training_steps=num_steps_stage3, last_epoch=-1,
                    )

                    (train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler) = accelerator.prepare(
                        new_train_dataloader if self.stage2_dataset_path else train_dataloader,
                        new_val_dataloader if self.stage2_dataset_path else val_dataloader,
                        unwrapped_model, self.optimizer, lr_scheduler_stage3,
                    )
                    if accelerator.state.num_processes > 1:
                        self.model = torch.nn.parallel.DistributedDataParallel(
                            accelerator.unwrap_model(self.model), device_ids=[self.model.device], find_unused_parameters=True,
                        )
                    logger.info("Optimizer rebuilt for Stage 3!")

                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        # Save final checkpoint
        if accelerator.is_main_process:
            stage3_ckpt_path = os.path.join(self.output_dir, "checkpoints", "stage3_final.ckpt")
            model_ddp = self.model
            self.model = accelerator.unwrap_model(self.model)
            self.save_checkpoint(path=stage3_ckpt_path)
            self.model = model_ddp
            logger.info(f"Saved Stage 3 final checkpoint to {stage3_ckpt_path}")

        accelerator.end_training()
        logger.info("Training completed!")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainFMMultiFreqConfidenceWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
