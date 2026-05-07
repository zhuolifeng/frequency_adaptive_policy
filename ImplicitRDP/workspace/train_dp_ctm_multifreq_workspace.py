"""
Training workspace for multi-frequency DP-CTM policy.

Two-stage training in a single run:
  Stage 1:  Karras EDM denoising on both 10hz and 50hz.
  Stage 2:  CTM distillation on 50hz  +  10hz EDM preservation.

EMA is managed **inside** the policy (policy.ema_update()), matching the
original consistency-policy design.  No external EMAModel is used for Stage 2.

Stage 1 uses an external EMAModel for standard EMA averaging (used for
validation / inference selection).

TensorBoard metrics (matching original consistency-policy):
  Stage 1:
    train/loss          (per step)   - EDM loss
    train/lr            (per step)   - learning rate
    train/epoch_loss    (per epoch)  - epoch average EDM loss
    val/loss            (per epoch)  - validation EDM loss
    val/mse_error       (per epoch)  - predict_action vs gt MSE
    train/action_mse_error (per epoch) - train sample predict_action vs gt MSE

  Stage 2:
    train/loss          (per step)   - total loss (ctm + dsm + preserve)
    train/lr            (per step)   - learning rate
    train/epoch_loss    (per epoch)  - epoch average total loss
    train/ctm_loss      (per epoch)  - epoch average CTM loss
    train/dsm_loss      (per epoch)  - epoch average DSM loss
    train/preserve_10hz_loss (per epoch) - epoch average 10hz preservation loss
    val/loss            (per epoch)  - validation total loss
    val/mse_error       (per epoch)  - predict_action vs gt MSE
    train/action_mse_error (per epoch) - train sample predict_action vs gt MSE
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
from accelerate import Accelerator, DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ImplicitRDP.common.checkpoint_util import TopKCheckpointManager
from ImplicitRDP.common.json_logger import JsonLogger
from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.dataset.base_dataset import BaseImageDataset
from ImplicitRDP.model.common.lr_scheduler import get_scheduler
from ImplicitRDP.model.diffusion.ema_model import EMAModel
from ImplicitRDP.policy.dp_ctm_multifreq_policy import DPCTMMultiFreqPolicy
from ImplicitRDP.workspace.base_workspace import BaseWorkspace

from loguru import logger

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDPCTMMultiFreqWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: DPCTMMultiFreqPolicy = hydra.utils.instantiate(cfg.policy)

        # Stage 1 EMA model (standard EMA for inference/validation)
        self.ema_model: DPCTMMultiFreqPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

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
        self.stage1_epochs = cfg.training.get("stage1_epochs", 600)
        self.stage2_epochs = cfg.training.get("stage2_epochs", 200)

        # Determine which action key to use for validation MSE
        self.action_train_key = getattr(cfg.policy, 'action_train_key', 'action_stack')

        logger.info(
            "Two-stage DP-CTM training: "
            f"stage1_epochs={self.stage1_epochs}, stage2_epochs={self.stage2_epochs}"
        )

    def _enter_stage2(self, cfg, accelerator, train_dataloader, val_dataloader, device):
        """Transition from Stage 1 (EDM) to Stage 2 (CTM distillation)."""
        unwrapped = accelerator.unwrap_model(self.model)

        # 1. Freeze teacher = deepcopy of current model backbone
        teacher = copy.deepcopy(unwrapped.model)
        teacher.requires_grad_(False)
        teacher.eval()
        unwrapped.set_teacher_model(teacher)
        unwrapped.set_training_stage(2)

        # 2. Freeze obs_encoder and RNN encoders
        unwrapped.obs_encoder.eval()
        unwrapped.obs_encoder.requires_grad_(False)
        logger.info("Stage 2: froze obs_encoder")
        if unwrapped.rnn_lowfreq is not None:
            unwrapped.rnn_lowfreq.eval()
            unwrapped.rnn_lowfreq.requires_grad_(False)
            logger.info("Stage 2: froze rnn_lowfreq")
        if unwrapped.rnn_highfreq is not None:
            unwrapped.rnn_highfreq.eval()
            unwrapped.rnn_highfreq.requires_grad_(False)
            logger.info("Stage 2: froze rnn_highfreq")

        # 3. Sync model_ema with current model
        unwrapped.model_ema.load_state_dict(unwrapped.model.state_dict())
        unwrapped.model_ema.to(device)

        # 4. Move teacher to device
        teacher.to(device)

        # 5. Update ema_model (for validation inference)
        if self.ema_model is not None:
            self.ema_model.load_state_dict(unwrapped.state_dict(), strict=False)
            self.ema_model.to(device)
            self.ema_model.set_training_stage(2)
            # frozen_teacher is not part of state_dict (it's a plain attribute),
            # so we must set it explicitly for validation compute_loss to work.
            self.ema_model.set_teacher_model(teacher)
            # Also sync ema_model's internal model_ema for CTM target projection
            self.ema_model.model_ema.load_state_dict(unwrapped.model.state_dict())
            self.ema_model.model_ema.to(device)

        self.current_stage = 2

        # 6. New optimizer for Stage 2 (match consistency-policy: lr=1e-4)
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        stage2_lr = optimizer_cfg.pop("stage2_lr", 1.0e-4)
        optimizer_cfg.pop("stage2_obs_encoder_lr", None)
        optimizer_cfg.pop("obs_encoder_lr", None)
        optimizer_cfg["lr"] = stage2_lr
        # obs_encoder is frozen in Stage 2, but get_optimizer still creates
        # param groups for it. Pass lr=0 and wd=0 so PyTorch doesn't receive
        # None values (which cause TypeError in LR scheduler).
        optimizer_cfg["obs_encoder_lr"] = 0.0
        optimizer_cfg["obs_encoder_weight_decay"] = 0.0
        self.optimizer = unwrapped.get_optimizer(**optimizer_cfg)

        if self.scaler is not None:
            self.scaler = GradScaler()

        # 7. New LR scheduler for Stage 2 (fresh cosine from 0)
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

        # 8. Re-prepare with accelerator
        prepared = accelerator.prepare(
            train_dataloader,
            val_dataloader,
            unwrapped,
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

        logger.info(f"Entered Stage 2 CTM distillation with lr={stage2_lr}")
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

        # Initialize accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.training.gradient_accumulate_every,
            kwargs_handlers=[ddp_kwargs],
        )
        cuda_count = accelerator.num_processes

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
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        total_epochs = self.stage1_epochs + self.stage2_epochs

        # Stage 1 LR scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * max(1, self.stage1_epochs)
            ) // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        # Stage 1 EMA (standard external EMA for validation)
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        # Prepare with accelerator
        self.model, self.optimizer, train_dataloader, val_dataloader, lr_scheduler = (
            accelerator.prepare(
                self.model, self.optimizer,
                train_dataloader, val_dataloader, lr_scheduler,
            )
        )

        device = accelerator.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        # TensorBoard (only on main process)
        writer = None
        if accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=os.path.join(self.output_dir, 'tb_logs'))

        if cfg.training.debug:
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            self.stage1_epochs = min(self.stage1_epochs, 1)
            self.stage2_epochs = min(self.stage2_epochs, 1)
            total_epochs = self.stage1_epochs + self.stage2_epochs

        val_sample_every = cfg.training.get("val_sample_every", 5)
        sample_every = cfg.training.get("sample_every", 5)
        train_sampling_batch = None

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        logger.info("Starting DP-CTM training...")

        stage2_started = False
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(total_epochs):
                # ---- Stage transition ----
                if local_epoch_idx < self.stage1_epochs:
                    self.current_stage = 1
                else:
                    if not stage2_started:
                        # Save Stage 1 checkpoint
                        if accelerator.is_main_process and self.stage1_epochs > 0:
                            stage1_path = os.path.join(
                                self.output_dir, "checkpoints", "stage1_final.ckpt"
                            )
                            model_ddp = self.model
                            self.model = accelerator.unwrap_model(self.model)
                            self.save_checkpoint(path=stage1_path)
                            self.model = model_ddp
                            logger.info(f"Saved Stage-1 checkpoint to {stage1_path}")

                        train_dataloader, val_dataloader, lr_scheduler = (
                            self._enter_stage2(
                                cfg, accelerator,
                                train_dataloader, val_dataloader, device
                            )
                        )
                        stage2_started = True
                    self.current_stage = 2

                unwrapped = accelerator.unwrap_model(self.model)
                unwrapped.set_training_stage(self.current_stage)

                self.model.train()

                # Freeze encoders
                if cfg.training.get("freeze_encoder", False) and self.current_stage == 1:
                    unwrapped.obs_encoder.eval()
                    unwrapped.obs_encoder.requires_grad_(False)

                if self.current_stage == 2:
                    unwrapped.obs_encoder.eval()
                    unwrapped.obs_encoder.requires_grad_(False)
                    if unwrapped.rnn_lowfreq is not None:
                        unwrapped.rnn_lowfreq.eval()
                        unwrapped.rnn_lowfreq.requires_grad_(False)
                    if unwrapped.rnn_highfreq is not None:
                        unwrapped.rnn_highfreq.eval()
                        unwrapped.rnn_highfreq.requires_grad_(False)

                # ---- Training loop ----
                step_log = {}
                train_losses = []
                ctm_losses = []
                dsm_losses = []
                preserve_losses = []
                edm_10hz_losses = []
                edm_50hz_losses = []

                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch} (stage {self.current_stage})",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                    disable=not accelerator.is_main_process,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        with accelerator.accumulate(self.model):
                            with autocast(enabled=use_amp, dtype=amp_dtype):
                                raw_loss, loss_dict = unwrapped.compute_loss(batch)

                            if not torch.isfinite(raw_loss):
                                logger.warning(
                                    f"Step {self.global_step}: loss={raw_loss.item()}, skipping"
                                )
                                self.optimizer.zero_grad(set_to_none=True)
                                self.global_step += 1
                                continue

                            accelerator.backward(raw_loss)

                            max_grad_norm = cfg.training.get("max_grad_norm", 1.0)
                            torch.nn.utils.clip_grad_norm_(
                                unwrapped.parameters(),
                                max_norm=max_grad_norm,
                            )

                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # EMA update
                        if self.current_stage == 2:
                            # Stage 2: internal CTM EMA (matches consistency-policy)
                            unwrapped.ema_update()
                        else:
                            # Stage 1: external standard EMA
                            if ema is not None:
                                ema.step(accelerator.unwrap_model(self.model))

                        # Collect per-step losses
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)

                        # Collect sub-losses for Stage 2
                        if "ctm" in loss_dict:
                            ctm_losses.append(loss_dict["ctm"])
                        if "dsm" in loss_dict:
                            dsm_losses.append(loss_dict["dsm"])
                        if "preserve_10hz" in loss_dict:
                            preserve_losses.append(loss_dict["preserve_10hz"])
                        # Collect sub-losses for Stage 1
                        if "edm_loss_10hz" in loss_dict:
                            edm_10hz_losses.append(loss_dict["edm_loss_10hz"])
                        if "edm_loss_50hz" in loss_dict:
                            edm_50hz_losses.append(loss_dict["edm_loss_50hz"])

                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        # Per-step TensorBoard logging
                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch and accelerator.is_main_process:
                            if writer is not None:
                                writer.add_scalar('train/loss', raw_loss_cpu, self.global_step)
                                writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (
                            cfg.training.max_train_steps is not None
                            and batch_idx >= cfg.training.max_train_steps - 1
                        ):
                            break

                # Epoch-level aggregation
                train_loss = np.mean(train_losses) if len(train_losses) > 0 else 0.0
                step_log["train_loss"] = train_loss
                if len(ctm_losses) > 0:
                    step_log["ctm"] = np.mean(ctm_losses)
                if len(dsm_losses) > 0:
                    step_log["dsm"] = np.mean(dsm_losses)
                if len(preserve_losses) > 0:
                    step_log["preserve_10hz"] = np.mean(preserve_losses)
                if len(edm_10hz_losses) > 0:
                    step_log["edm_loss_10hz"] = np.mean(edm_10hz_losses)
                if len(edm_50hz_losses) > 0:
                    step_log["edm_loss_50hz"] = np.mean(edm_50hz_losses)

                # ========= Validation =========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema and self.ema_model is not None:
                    policy = self.ema_model
                policy.eval()

                if (self.epoch % cfg.training.val_every) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        val_losses = []
                        val_mse_errors = []

                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Val epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                            disable=not accelerator.is_main_process,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                # Validation loss
                                raw_vloss, _ = policy.compute_loss(batch)
                                val_losses.append(raw_vloss.item())

                                # Validation MSE (predict_action vs gt)
                                if (self.epoch % val_sample_every) == 0:
                                    obs_dict = batch["obs"]
                                    gt_action = batch[self.action_train_key]
                                    result = policy.predict_action(
                                        obs_dict,
                                        extended_obs_dict=batch.get("extended_obs", None),
                                    )
                                    pred_action = result["action_pred"]
                                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                                    val_mse_errors.append(mse.item())

                                if (
                                    cfg.training.max_val_steps is not None
                                    and batch_idx >= cfg.training.max_val_steps - 1
                                ):
                                    break

                        if len(val_losses) > 0:
                            step_log["val_loss"] = np.mean(val_losses)
                        if len(val_mse_errors) > 0:
                            step_log["val_mse_error"] = np.mean(val_mse_errors)

                # Sample from training batch (predict_action MSE)
                if (self.epoch % sample_every) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        obs_dict = train_sampling_batch["obs"]
                        gt_action = train_sampling_batch[self.action_train_key]
                        result = policy.predict_action(
                            obs_dict,
                            extended_obs_dict=train_sampling_batch.get("extended_obs", None),
                        )
                        pred_action = result["action_pred"]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()

                # ========= Epoch-level TensorBoard & Checkpoint =========
                if accelerator.is_main_process:
                    if writer is not None:
                        # -- Always log --
                        writer.add_scalar('train/epoch_loss', step_log.get('train_loss', 0), self.epoch)
                        writer.add_scalar('train/training_stage', float(self.current_stage), self.epoch)

                        # -- Stage 2 sub-losses --
                        if 'ctm' in step_log:
                            writer.add_scalar('train/ctm_loss', step_log['ctm'], self.epoch)
                        if 'dsm' in step_log:
                            writer.add_scalar('train/dsm_loss', step_log['dsm'], self.epoch)
                        if 'preserve_10hz' in step_log:
                            writer.add_scalar('train/preserve_10hz_loss', step_log['preserve_10hz'], self.epoch)

                        # -- Stage 1 sub-losses --
                        if 'edm_loss_10hz' in step_log:
                            writer.add_scalar('train/edm_loss_10hz', step_log['edm_loss_10hz'], self.epoch)
                        if 'edm_loss_50hz' in step_log:
                            writer.add_scalar('train/edm_loss_50hz', step_log['edm_loss_50hz'], self.epoch)

                        # -- Validation --
                        if 'val_loss' in step_log:
                            writer.add_scalar('val/loss', step_log['val_loss'], self.epoch)
                        if 'val_mse_error' in step_log:
                            writer.add_scalar('val/mse_error', step_log['val_mse_error'], self.epoch)
                        if 'train_action_mse_error' in step_log:
                            writer.add_scalar('train/action_mse_error', step_log['train_action_mse_error'], self.epoch)

                    # Checkpoint
                    if (self.epoch % cfg.training.checkpoint_every) == 0:
                        model_ddp = self.model
                        self.model = accelerator.unwrap_model(self.model)
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        metric_dict = {
                            k.replace("/", "_"): v for k, v in step_log.items()
                        }
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)
                        self.model = model_ddp

                    json_logger.log(step_log)

                policy.train()
                self.global_step += 1
                self.epoch += 1

        # Save final checkpoint & close writer
        if accelerator.is_main_process:
            final_path = os.path.join(
                self.output_dir, "checkpoints", "dp_ctm_final.ckpt"
            )
            model_ddp = self.model
            self.model = accelerator.unwrap_model(self.model)
            self.save_checkpoint(path=final_path)
            self.model = model_ddp
            logger.info(f"Saved final DP-CTM checkpoint to {final_path}")

        if writer is not None:
            writer.close()

        logger.info("DP-CTM training completed")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainDPCTMMultiFreqWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
