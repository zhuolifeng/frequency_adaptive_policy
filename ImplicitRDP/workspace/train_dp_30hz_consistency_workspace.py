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
from ImplicitRDP.policy.dp_30hz_consistency_policy import DP30HzConsistencyPolicy
from ImplicitRDP.workspace.base_workspace import BaseWorkspace

from loguru import logger


OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDP30HzConsistencyWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch", "current_stage", "ema_optimization_step", "stage2_epochs_completed"]

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: DP30HzConsistencyPolicy = hydra.utils.instantiate(cfg.policy)
        self.teacher_model: DP30HzConsistencyPolicy = copy.deepcopy(self.model)
        self.ema_model: DP30HzConsistencyPolicy = None
        if not cfg.training.use_ema:
            raise ValueError("TrainDP30HzConsistencyWorkspace expects training.use_ema=True")
        self.ema_model = copy.deepcopy(self.model)
        self.teacher_model.set_training_stage(1)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)

        self.stage1_epochs = int(cfg.training.get("stage1_epochs", 600))
        self.stage2_epochs = int(cfg.training.get("stage2_epochs", 100))
        if cfg.training.debug:
            self.stage1_epochs = min(1, max(0, self.stage1_epochs))
            self.stage2_epochs = 1
        self.stage2_epochs_completed = 0
        self.current_stage = 1 if self.stage1_epochs > 0 else 2
        self.model.set_training_stage(self.current_stage)
        self.model.apply_stage_freeze()

        self.optimizer = self.model.get_optimizer(**self._optimizer_cfg_for_stage(cfg, self.current_stage))
        self.global_step = 0
        self.epoch = 0
        self.ema_optimization_step = 0
        self.scaler = GradScaler() if cfg.training.get("use_amp", False) else None
        # Optimizer/scaler restoration is not stage-safe because Stage 1 and
        # Stage 2 use different parameter groups. Rebuild them from current_stage
        # on resume, but always checkpoint the frozen teacher and EMA target.
        self.exclude_keys = ["optimizer", "scaler"]

    def _optimizer_cfg_for_stage(self, cfg, stage: int):
        opt = OmegaConf.to_container(cfg.optimizer, resolve=True)
        common = {
            "weight_decay": opt.get("weight_decay", 1.0e-6),
            "obs_encoder_weight_decay": opt.get("obs_encoder_weight_decay", 1.0e-6),
            "betas": opt.get("betas", (0.95, 0.999)),
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
                "lr": opt.get("stage2_lr", 1.0e-5),
                "obs_encoder_lr": opt.get("stage2_lr", 1.0e-5),
            }
        raise ValueError(f"Invalid stage={stage}")

    def _make_accelerator(self, cfg):
        backend = cfg.logging.get("backend", "tensorboard")
        if backend == "tensorboard":
            return Accelerator(log_with="tensorboard", project_dir=os.path.join(self.output_dir, "tb_logs"))
        if backend == "wandb":
            return Accelerator(log_with="wandb")
        if backend in ("none", "disabled", None):
            return Accelerator()
        raise ValueError(f"Unsupported logging.backend={backend}")

    def _init_trackers(self, accelerator, cfg):
        backend = cfg.logging.get("backend", "tensorboard")
        if backend == "wandb":
            wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            wandb_cfg.pop("project", None)
            wandb_cfg.pop("backend", None)
            accelerator.init_trackers(
                project_name=cfg.logging.project,
                config=OmegaConf.to_container(cfg, resolve=True),
                init_kwargs={"wandb": wandb_cfg},
            )
        elif backend == "tensorboard":
            accelerator.init_trackers(project_name=cfg.logging.project)

    def _save_named_checkpoint(self, accelerator, name: str):
        if not accelerator.is_main_process:
            return
        model_ddp = self.model
        self.model = accelerator.unwrap_model(self.model)
        path = os.path.join(self.output_dir, "checkpoints", name)
        self.save_checkpoint(path=path, use_thread=False)
        self.model = model_ddp
        logger.info(f"Saved checkpoint: {path}")

    def _start_distillation(self, accelerator, save_stage1=False):
        unwrapped_model = accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(self.ema_model.state_dict(), strict=True)
        if save_stage1:
            self._save_named_checkpoint(accelerator, "stage1_final.ckpt")
        self.teacher_model.load_state_dict(unwrapped_model.state_dict(), strict=True)
        self.current_stage = 2
        unwrapped_model.set_training_stage(2)
        self.ema_model.set_training_stage(2)
        self.ema_optimization_step = 0
        logger.info("Initialized Stage 2 student and frozen teacher from Stage 1 EMA")

    def _rebuild_for_stage(self, cfg, accelerator, train_dataloader, val_dataloader, stage: int, stage_epochs: int):
        unwrapped_model = accelerator.unwrap_model(self.model)
        unwrapped_model.set_training_stage(stage)
        unwrapped_model.apply_stage_freeze()
        self.current_stage = stage
        self.optimizer = unwrapped_model.get_optimizer(**self._optimizer_cfg_for_stage(cfg, stage))
        if self.scaler is not None:
            self.scaler = GradScaler()
        num_steps = (len(train_dataloader) * max(1, stage_epochs)) // cfg.training.gradient_accumulate_every
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=max(1, num_steps),
            last_epoch=-1,
        )
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader,
            val_dataloader,
            unwrapped_model,
            self.optimizer,
            lr_scheduler,
        )
        return train_dataloader, val_dataloader, lr_scheduler

    @staticmethod
    def _with_stage1_weights(payload):
        # Older Stage 1 checkpoints may have only one trained copy.
        states = payload["state_dicts"]
        teacher = states.get("ema_model", states.get("model"))
        if not teacher:
            raise ValueError("Stage-1 checkpoint needs trained model or EMA weights.")
        return {**payload, "state_dicts": {"model": teacher, "ema_model": teacher, **states}}

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        restore_stage = include_keys is None or "current_stage" in include_keys
        if restore_stage:
            if "current_stage" not in payload.get("pickles", {}):
                raise ValueError("Consistency checkpoint is missing current_stage metadata.")
            stored_stage = dill.loads(payload["pickles"]["current_stage"])
            states = payload["state_dicts"]
            if stored_stage == 1:
                payload = self._with_stage1_weights(payload)
            elif stored_stage == 2 and (not states.get("model") or not states.get("ema_model")):
                raise ValueError("Stage-2 checkpoint needs both model and EMA weights.")
        super().load_payload(payload, exclude_keys, include_keys, **kwargs)
        if restore_stage:
            # The stage is workspace metadata, not part of the policy state_dict.
            self.model.set_training_stage(self.current_stage)
            self.ema_model.set_training_stage(self.current_stage)
            self._preserve_checkpoint_time_condition_layout(payload)

    def _resume_training(self, cfg, accelerator, checkpoint_path):
        payload = self.load_checkpoint(
            path=checkpoint_path, exclude_keys=self.exclude_keys,
            include_keys=self.include_keys, map_location="cpu",
        )
        metadata = payload["pickles"]
        if "epoch" not in metadata:
            raise ValueError("Consistency training resume needs epoch metadata.")
        # Older periodic saves used the just-finished epoch index, while final
        # saves used an epoch count. Without a marker, a legacy resume can
        # repeat its last epoch. Keep the recorded value for those checkpoints.
        if self.current_stage == 1:
            if "ema_model" not in payload["state_dicts"]:
                self.ema_optimization_step = 0
            if self.epoch >= self.stage1_epochs and self.stage2_epochs > 0:
                self._start_distillation(accelerator)
        else:
            self._check_stage2_checkpoint(payload, checkpoint_path)
            if "stage2_epochs_completed" not in metadata:
                # Legacy checkpoints stored only the total across both stages.
                training = payload.get("cfg", {}).get("training", {})
                if "stage1_epochs" not in training:
                    raise ValueError("Legacy Stage-2 resume needs the checkpoint's training.stage1_epochs.")
                stage1_epochs = int(training["stage1_epochs"])
                if training.get("debug"):
                    stage1_epochs = min(1, max(0, stage1_epochs))
                self.stage2_epochs_completed = max(0, self.epoch - stage1_epochs)
        self.optimizer = self.model.get_optimizer(**self._optimizer_cfg_for_stage(cfg, self.current_stage))
        if self.scaler is not None:
            self.scaler = GradScaler()
        logger.info(f"Rebuilt optimizer for resumed Stage {self.current_stage}")

    def _remaining_stages(self):
        stage1 = max(0, self.stage1_epochs - self.epoch) if self.current_stage == 1 else 0
        stage2 = max(0, self.stage2_epochs - self.stage2_epochs_completed)
        return [1] * stage1 + [2] * stage2

    def _load_pretrained_if_needed(self, cfg):
        ckpt_path = cfg.training.get("pretrained_ckpt_path")
        if ckpt_path is None:
            if self.current_stage == 2:
                raise ValueError(
                    "Stage 2 needs a trained teacher. Set training.pretrained_ckpt_path "
                    "to a Stage-1 checkpoint, resume an existing run, or train Stage 1 first."
                )
            return
        payload = torch.load(ckpt_path, pickle_module=dill, map_location="cpu")
        if self.current_stage == 1:
            payload = self._with_stage1_weights(payload)
            self.load_payload(payload, exclude_keys=["optimizer", "scaler"], include_keys=[], strict=False)
        else:
            states = payload["state_dicts"]
            stored_stage = payload.get("pickles", {}).get("current_stage")
            if stored_stage is not None and dill.loads(stored_stage) == 2:
                self.load_payload(payload, exclude_keys=["optimizer", "scaler"], include_keys=[], strict=True)
                self._check_stage2_checkpoint(payload, ckpt_path)
                self._preserve_checkpoint_time_condition_layout(payload)
            else:
                # Match the normal Stage 1 -> 2 transition, including EMA selection.
                teacher_state = states.get("ema_model", states.get("model"))
                if not teacher_state:
                    raise ValueError(f"No trained policy weights in teacher checkpoint {ckpt_path}.")
                for policy in (self.model, self.ema_model, self.teacher_model):
                    policy.load_state_dict(teacher_state, strict=True)
        logger.info(f"Loaded pretrained checkpoint: {ckpt_path}")

    def _check_stage2_checkpoint(self, payload, checkpoint_path):
        states = payload["state_dicts"]
        if not states.get("model") or not states.get("ema_model"):
            raise ValueError(f"Stage-2 checkpoint {checkpoint_path} needs both model and EMA weights.")
        if not states.get("teacher_model"):
            self._restore_stage2_teacher_from_stage1_checkpoint(checkpoint_path)

    def _restore_stage2_teacher_from_stage1_checkpoint(self, stage2_checkpoint_path):
        stage1_path = pathlib.Path(stage2_checkpoint_path).parent.joinpath("stage1_final.ckpt")
        if not stage1_path.is_file():
            raise RuntimeError(
                "Stage-2 checkpoint does not contain teacher_model and no "
                f"Stage-1 teacher checkpoint was found at {stage1_path}. "
                "Refusing to replace the frozen teacher with the current student."
            )
        payload = torch.load(
            stage1_path.open("rb"),
            pickle_module=dill,
            map_location="cpu",
        )
        state_dicts = payload.get("state_dicts", {})
        teacher_state = state_dicts.get("ema_model", state_dicts.get("model"))
        if teacher_state is None:
            raise RuntimeError(
                f"No model or ema_model state was found in Stage-1 checkpoint {stage1_path}."
            )
        self.teacher_model.load_state_dict(teacher_state, strict=True)
        logger.info(f"Restored frozen Stage-1 teacher from {stage1_path}")

    def _preserve_checkpoint_time_condition_layout(self, payload):
        if self.current_stage != 2:
            return
        checkpoint_cfg = payload.get("cfg", None)
        checkpoint_position = "prepend"
        if checkpoint_cfg is not None:
            checkpoint_position = OmegaConf.select(
                checkpoint_cfg,
                "policy.model.time_cond_position",
                default="prepend",
            )
        current_position = self.model.model.time_cond_position
        if checkpoint_position == current_position:
            return
        for policy in (self.model, self.teacher_model, self.ema_model):
            policy.model.time_cond_position = checkpoint_position
        OmegaConf.update(self.cfg, "policy.model.time_cond_position", checkpoint_position, force_add=True)
        logger.warning(
            "Stage-2 checkpoint was trained with "
            f"time_cond_position={checkpoint_position}; overriding the current "
            f"setting ({current_position}) to preserve its token positions."
        )

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        use_amp = cfg.training.get("use_amp", False)
        amp_dtype = cfg.training.get("amp_dtype", "bf16")
        if use_amp:
            amp_dtype = torch.bfloat16 if amp_dtype == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
        else:
            amp_dtype = None

        accelerator = self._make_accelerator(cfg)
        self._init_trackers(accelerator, cfg)
        cuda_count = accelerator.num_processes

        resume_checkpoint_path = None
        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                logger.info(f"Resuming from checkpoint {latest_ckpt_path}")
                self._resume_training(cfg, accelerator, latest_ckpt_path)
                resume_checkpoint_path = latest_ckpt_path
        if resume_checkpoint_path is None:
            self._load_pretrained_if_needed(cfg)

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        val_dataset = dataset.get_validation_dataset()
        dataloader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
        val_dataloader_cfg = OmegaConf.to_container(cfg.val_dataloader, resolve=True)
        dataloader_cfg["batch_size"] = dataloader_cfg["batch_size"] // cuda_count
        val_dataloader_cfg["batch_size"] = val_dataloader_cfg["batch_size"] // cuda_count
        train_dataloader = DataLoader(dataset, **dataloader_cfg)
        val_dataloader = DataLoader(val_dataset, **val_dataloader_cfg)

        if self.current_stage == 1 or not self.model.normalizer.params_dict:
            normalizer = dataset.get_normalizer()
            self.model.set_normalizer(normalizer)
            self.teacher_model.set_normalizer(normalizer)
            self.ema_model.set_normalizer(normalizer)

        if cfg.training.debug:
            cfg.training.max_train_steps = 2
            cfg.training.max_val_steps = 2
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1

        stage_schedule = self._remaining_stages()
        total_epochs = len(stage_schedule)

        current_stage_epochs = stage_schedule.count(self.current_stage)
        num_steps = (
            len(train_dataloader) * max(1, current_stage_epochs)
        ) // cfg.training.gradient_accumulate_every
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=max(1, num_steps),
            last_epoch=-1 if resume_checkpoint_path is not None else self.global_step - 1,
        )
        ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
        ema.optimization_step = int(self.ema_optimization_step)
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader,
            val_dataloader,
            self.model,
            self.optimizer,
            lr_scheduler,
        )
        device = accelerator.device
        self.teacher_model.to(device)
        self.ema_model.to(device)

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx, stage in enumerate(stage_schedule):
                stage_name = "Stage 1 Base DP" if stage == 1 else "Stage 2 Consistency Distillation"
                unwrapped_model: DP30HzConsistencyPolicy = accelerator.unwrap_model(self.model)
                unwrapped_model.set_training_stage(stage)
                unwrapped_model.update_progress(self.epoch / max(1, self.stage1_epochs + self.stage2_epochs))
                self.model.train()
                unwrapped_model.apply_stage_freeze()
                self.teacher_model.eval()
                self.ema_model.set_training_stage(stage)
                self.ema_model.apply_stage_freeze()
                self.ema_model.eval()
                logger.info(f"Epoch {self.epoch}: {stage_name}")

                step_log = {}
                train_losses = []
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch} ({stage_name})", leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        with autocast(enabled=use_amp, dtype=amp_dtype):
                            if stage == 2:
                                raw_loss, loss_dict = unwrapped_model.compute_loss(
                                    batch,
                                    target_policy=self.ema_model,
                                    teacher_policy=self.teacher_model,
                                )
                            else:
                                raw_loss, loss_dict = unwrapped_model.compute_loss(batch)
                            loss = raw_loss / cfg.training.gradient_accumulate_every

                        if use_amp:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            if use_amp:
                                self.scaler.unscale_(self.optimizer)
                            max_grad_norm = cfg.training.get("max_grad_norm", 1.0)
                            torch.nn.utils.clip_grad_norm_(unwrapped_model.parameters(), max_norm=max_grad_norm, error_if_nonfinite=False)
                            if use_amp:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                            lr_scheduler.step()
                            ema.step(accelerator.unwrap_model(self.model))
                            self.ema_optimization_step = int(ema.optimization_step)

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
                        if cfg.training.max_train_steps is not None and batch_idx >= cfg.training.max_train_steps - 1:
                            break

                step_log["train_loss"] = float(np.nanmean(train_losses)) if len(train_losses) > 0 else 0.0

                if self.epoch % cfg.training.val_every == 0 and len(val_dataloader) > 0 and accelerator.is_main_process:
                    policy = self.ema_model
                    policy.set_training_stage(stage)
                    policy.eval()
                    with torch.no_grad():
                        val_losses = []
                        for batch_idx, batch in enumerate(val_dataloader):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            with autocast(enabled=use_amp, dtype=amp_dtype):
                                if stage == 2:
                                    val_loss, _ = policy.compute_loss(
                                        batch,
                                        target_policy=self.ema_model,
                                        teacher_policy=self.teacher_model,
                                    )
                                else:
                                    val_loss, _ = policy.compute_loss(batch)
                            val_losses.append(val_loss.item())
                            if cfg.training.max_val_steps is not None and batch_idx >= cfg.training.max_val_steps - 1:
                                break
                        if len(val_losses) > 0:
                            step_log["val_loss"] = float(np.nanmean(val_losses))

                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                checkpoint_due = self.epoch % cfg.training.checkpoint_every == 0
                self.epoch += 1
                if stage == 2:
                    self.stage2_epochs_completed += 1

                if accelerator.is_main_process and checkpoint_due:
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
                    topk_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_path is not None:
                        self.save_checkpoint(path=topk_path)
                    self.model = model_ddp

                next_stage = stage_schedule[local_epoch_idx + 1] if local_epoch_idx + 1 < total_epochs else None
                if stage == 1 and next_stage == 2:
                    self._start_distillation(accelerator, save_stage1=True)
                    ema.decay = 0.0
                    ema.optimization_step = self.ema_optimization_step
                    stage2_dataloader_cfg = dict(dataloader_cfg)
                    stage2_val_dataloader_cfg = dict(val_dataloader_cfg)
                    stage2_dataloader_cfg["pin_memory"] = False
                    stage2_val_dataloader_cfg["pin_memory"] = False
                    new_train_dataloader = DataLoader(dataset, **stage2_dataloader_cfg)
                    new_val_dataloader = DataLoader(val_dataset, **stage2_val_dataloader_cfg)
                    train_dataloader, val_dataloader, lr_scheduler = self._rebuild_for_stage(
                        cfg,
                        accelerator,
                        new_train_dataloader,
                        new_val_dataloader,
                        stage=2,
                        stage_epochs=self.stage2_epochs,
                    )

        if accelerator.is_main_process:
            model_ddp = self.model
            self.model = accelerator.unwrap_model(self.model)
            self.save_checkpoint(use_thread=False)
            self.model = model_ddp
        accelerator.end_training()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainDP30HzConsistencyWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
