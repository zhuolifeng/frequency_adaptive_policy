import copy
import os
import pickle
import random
from pathlib import Path
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import tqdm
import dill
from omegaconf import OmegaConf
import hydra
from accelerate import Accelerator

from ImplicitRDP.workspace.base_workspace import BaseWorkspace
from ImplicitRDP.dataset.base_dataset import BaseImageDataset
from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.common.checkpoint_util import TopKCheckpointManager
from ImplicitRDP.model.diffusion.ema_model import EMAModel
from ImplicitRDP.model.common.lr_scheduler import get_scheduler
from ImplicitRDP.common.json_logger import JsonLogger

from loguru import logger

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainACTConfidenceStagewiseWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        accelerator = Accelerator()
        cuda_count = accelerator.num_processes
        print("###########################################")
        print(f"Number of available CUDA devices: {cuda_count}.")
        print(f"Original learning rate: {optimizer_cfg['lr']}")
        print(f"Updated learning rate: {optimizer_cfg['lr']}")
        print("###########################################")
        self.optimizer = self.model.get_optimizer(**optimizer_cfg)

        self.global_step = 0
        self.epoch = 0
        
        # Initialize AMP scaler if AMP is enabled
        self.scaler = None
        if cfg.training.get('use_amp', False):
            self.scaler = GradScaler()
        
        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer', 'scaler']

        # Stagewise training config
        self.stage1_epochs = cfg.training.get('stage1_epochs', 600)
        self.stage2_epochs = cfg.training.get('stage2_epochs', 300)
        self.stage2_dataset_path = cfg.training.get('stage2_dataset_path', None)
        self.confidence_lr = cfg.training.get('confidence_lr', None)
        self.confidence_weight_decay = cfg.training.get('confidence_weight_decay', None)
        self.current_stage = 1

        # Prepare two dataset configs for stage 1 and stage 2
        self._dataset_stage1_cfg = cfg.task.dataset
        self._dataset_stage2_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task.dataset, resolve=True))
        stage2_dataset_path = self.stage2_dataset_path if self.stage2_dataset_path is not None else cfg.task.dataset_path
        self._dataset_stage2_cfg['dataset_path'] = stage2_dataset_path

        logger.info(f"Two-stage dataset configuration:")
        logger.info(f"  Stage 1 dataset: {cfg.task.dataset_path}")
        logger.info(f"  Stage 2 dataset: {stage2_dataset_path}")
        self._dataset_stage2 = None  # Kept for reference; actual dataset is built in run()

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        # Get AMP configuration
        use_amp = cfg.training.get('use_amp', False)
        amp_dtype = cfg.training.get('amp_dtype', 'bf16')
        
        # Determine autocast dtype
        if use_amp:
            if amp_dtype == 'bf16' and torch.cuda.is_bf16_supported():
                logger.info(f"Using bf16 for AMP")
                amp_dtype = torch.bfloat16
            elif amp_dtype == 'fp16':
                logger.info(f"Using fp16 for AMP")
                amp_dtype = torch.float16
            else:
                logger.warning(f"AMP dtype {amp_dtype} not supported or bf16 not available, falling back to fp16")
                amp_dtype = torch.float16
        else:
            amp_dtype = None
        
        log_backend = cfg.logging.get('backend', 'tensorboard')
        if log_backend == 'wandb':
            accelerator = Accelerator(log_with='wandb')
        else:
            accelerator = Accelerator(log_with='tensorboard', project_dir=os.path.join(self.output_dir, "tb_logs"))
        cuda_count = accelerator.num_processes

        if log_backend == 'wandb':
            wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            wandb_cfg.pop('project', None)
            wandb_cfg.pop('backend', None)
            accelerator.init_trackers(
                project_name=cfg.logging.project,
                config=OmegaConf.to_container(cfg, resolve=True),
                init_kwargs={"wandb": wandb_cfg}
            )
        else:
            accelerator.init_trackers(project_name=cfg.logging.project)

        # resume training
        if cfg.training.resume:
            assert cfg.training.pretrained_ckpt_path is None, "pretrained_ckpt_path and resume cannot be used together"
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # init model from pretrained checkpoint
        if cfg.training.pretrained_ckpt_path is not None:
            ckpt_path = cfg.training.pretrained_ckpt_path
            payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
            # handle normalizer
            for key in self.model.state_dict():
                if 'normalizer' in key:
                    payload['state_dicts']['model'][key] = self.model.state_dict()[key]
                    if self.ema_model is not None:
                        payload['state_dicts']['ema_model'][key] = self.model.state_dict()[key]
            self.load_payload(payload, exclude_keys=['optimizer', 'scaler'], include_keys=[], strict=False)
            logger.info(f"Loaded model from checkpoint: {ckpt_path}")

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        # Pre-build Stage 2 dataset
        dataset_stage2: BaseImageDataset = hydra.utils.instantiate(self._dataset_stage2_cfg)
        assert isinstance(dataset_stage2, BaseImageDataset)

        dataloader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
        val_dataloader_cfg = OmegaConf.to_container(cfg.val_dataloader, resolve=True)
        print("###########################################")
        print(f"Number of available CUDA devices: {cuda_count}.")
        print(f"Original training batch size: {dataloader_cfg['batch_size']}.")
        print(f"Original validation batch size: {val_dataloader_cfg['batch_size']}.")
        dataloader_cfg['batch_size'] = dataloader_cfg['batch_size'] // cuda_count
        val_dataloader_cfg['batch_size'] = val_dataloader_cfg['batch_size'] // cuda_count
        print(f"Updated training batch size: {dataloader_cfg['batch_size']}")
        print(f"Updated validation batch size: {val_dataloader_cfg['batch_size']}")
        print("###########################################")
        
        train_dataloader = DataLoader(dataset, **dataloader_cfg)
        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **val_dataloader_cfg)

        # Compute normalizer BEFORE accelerator.prepare() — no cross-process file I/O needed
        normalizer = dataset.get_normalizer()
        if cfg.training.pretrained_ckpt_path is not None:
            normalizer_state_dict = normalizer.state_dict()
            current_normalizer_state_dict = self.model.normalizer.state_dict()
            for key in current_normalizer_state_dict.keys():
                normalizer_state_dict[key] = current_normalizer_state_dict[key]
            normalizer.load_state_dict(normalizer_state_dict)
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # Save dataset resume params on main process
        dataset_resume_params_path = os.path.join(self.output_dir, 'dataset_resume_params.pkl')
        if accelerator.is_main_process:
            dataset_resume_params = dataset.get_resume_params()
            with open(dataset_resume_params_path, 'wb') as f:
                pickle.dump(dataset_resume_params, f)

        # Stagewise num_epochs override (resolve before lr_scheduler needs it)
        cfg.training.num_epochs = self.stage1_epochs + self.stage2_epochs

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )
        
        # accelerator - prepare for training
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
        )
        if accelerator.state.num_processes > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                accelerator.unwrap_model(self.model),
                device_ids=[self.model.device],
                find_unused_parameters=True
            )
        
        device = accelerator.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        train_sampling_batch = None

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        logger.info(f"Starting stagewise training (Single Encoder)... Stage 1: {self.stage1_epochs} epochs, Stage 2: {self.stage2_epochs} epochs")
        
        
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                # Determine current stage
                if local_epoch_idx  < self.stage1_epochs:
                    self.current_stage = 1
                    stage_name = "Stage 1 (ACT)"
                else:
                    self.current_stage = 2
                    stage_name = "Stage 2 (Confidence)"
                
                # Use unwrap_model to get original module since DDP wraps the model
                unwrapped_model = accelerator.unwrap_model(self.model)
                unwrapped_model.update_progress(self.epoch / cfg.training.num_epochs)
                
                # Update training stage in model
                if hasattr(unwrapped_model, 'set_training_stage'):
                    unwrapped_model.set_training_stage(self.current_stage)
                
                if local_epoch_idx % 10 == 0 or local_epoch_idx == self.stage1_epochs:
                    logger.info(f"Epoch {local_epoch_idx}: {stage_name}")
                
                self.model.train()
                
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                device = accelerator.device
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch} ({stage_name})", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # always use the latest batch
                        train_sampling_batch = batch

                        # compute loss
                        with autocast(enabled=use_amp, dtype=amp_dtype):
                            raw_loss, loss_dict = self.model(batch)
                            loss = raw_loss / cfg.training.gradient_accumulate_every
                            
                        # Scale loss and backward pass
                        if use_amp:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()
                            
                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            if use_amp:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema and accelerator.is_main_process:
                            ema.step(accelerator.unwrap_model(self.model))

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0],
                            'training_stage': float(self.current_stage)
                        }
                        # 添加各个损失项
                        for key, value in loss_dict.items():
                            step_log[f'train_{key}'] = value

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.nanmean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.update_progress(self.epoch / cfg.training.num_epochs)
                policy.eval()
                
                # run validation
                val_interval = cfg.training.get('val_interval', 1)
                if (self.epoch % val_interval == 0) and len(val_dataloader) > 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        val_losses = list()
                        device = accelerator.device
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch in tepoch:
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                
                                with autocast(enabled=use_amp, dtype=amp_dtype):
                                    raw_loss, loss_dict = policy.compute_loss(batch)
                                
                                val_losses.append(raw_loss.item())
                                tepoch.set_postfix(loss=raw_loss.item(), refresh=False)
                        
                        if len(val_losses) > 0:
                            val_loss = np.mean(val_losses)
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                            logger.info(f"Epoch {self.epoch}, val_loss: {val_loss:.6f}")

                # checkpoint
                checkpoint_interval = cfg.training.get('checkpoint_interval', 10)
                if (self.epoch % checkpoint_interval == 0) and accelerator.is_main_process:
                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.get('save_last_snapshot', False):
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    # recover the DDP model
                    self.model = model_ddp
                
                # ========= Stage transition handling ==========
                # When transitioning from stage 1 to stage 2
                if self.stage2_epochs > 0 and local_epoch_idx == self.stage1_epochs - 1:
                    logger.info(f"Stage 1 completed! Now transitioning to Stage 2...")
                    
                    # Save stage 1 checkpoint
                    if accelerator.is_main_process:
                        stage1_ckpt_path = os.path.join(self.output_dir, 'checkpoints', 'stage1_final.ckpt')
                        model_to_save = accelerator.unwrap_model(self.model)
                        self.save_checkpoint(path=stage1_ckpt_path)
                        logger.info(f"Saved Stage 1 checkpoint to {stage1_ckpt_path}")
                    
                    # Switch to Stage 2 dataset if different from Stage 1
                    if self._dataset_stage2_cfg['dataset_path'] != self._dataset_stage1_cfg['dataset_path']:
                        logger.info(f"Switching to Stage 2 dataset: {self._dataset_stage2_cfg['dataset_path']}")
                        
                        # dataset_stage2 was pre-built at the start of run() — reuse it
                        new_train_dataset = dataset_stage2
                        new_val_dataset = dataset_stage2.get_validation_dataset()
                        new_train_dataloader = DataLoader(new_train_dataset, **dataloader_cfg)
                        new_val_dataloader = DataLoader(new_val_dataset, **val_dataloader_cfg)
                        
                        # Compute and switch normalizer for stage 2 dataset (no cross-process file I/O)
                        dataset_stage2_normalizer = dataset_stage2.get_normalizer()
                        unwrapped_model = accelerator.unwrap_model(self.model)
                        unwrapped_model.set_normalizer(dataset_stage2_normalizer)
                        if cfg.training.use_ema:
                            self.ema_model.set_normalizer(dataset_stage2_normalizer)
                            self.ema_model.to(accelerator.device)
                        logger.info("Normalizer switched to Stage 2 dataset normalizer")
                    else:
                        # Same dataset, just rebuild dataloader to reset epoch
                        logger.info("Stage 2 uses same dataset as Stage 1, rebuilding DataLoader")
                        new_train_dataloader = DataLoader(dataset, **dataloader_cfg)
                        new_val_dataloader = DataLoader(dataset.get_validation_dataset(), **val_dataloader_cfg)
                    
                    # Rebuild optimizer for stage 2 (different parameters to optimize)
                    logger.info("Rebuilding optimizer for Stage 2...")
                    optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
                    unwrapped_model = accelerator.unwrap_model(self.model)
                    if hasattr(unwrapped_model, 'set_training_stage'):
                        unwrapped_model.set_training_stage(2)
                    
                    # Add confidence_lr and confidence_weight_decay to optimizer config if specified
                    if self.confidence_lr is not None:
                        optimizer_cfg['confidence_lr'] = self.confidence_lr
                    if self.confidence_weight_decay is not None:
                        optimizer_cfg['confidence_weight_decay'] = self.confidence_weight_decay
                    
                    self.optimizer = unwrapped_model.get_optimizer(**optimizer_cfg)
                    
                    # Reset GradScaler for Stage 2 to avoid stale state
                    if self.scaler is not None:
                        self.scaler = GradScaler()
                    
                    # Rebuild lr_scheduler for Stage 2 with correct total steps
                    num_steps_stage2 = (len(new_train_dataloader) * self.stage2_epochs) \
                        // cfg.training.gradient_accumulate_every
                    lr_scheduler_stage2 = get_scheduler(
                        cfg.training.lr_scheduler,
                        optimizer=self.optimizer,
                        num_warmup_steps=cfg.training.lr_warmup_steps,
                        num_training_steps=num_steps_stage2,
                        last_epoch=-1
                    )
                    
                    # Re-prepare with new DataLoader, optimizer, and lr_scheduler
                    train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
                        new_train_dataloader, new_val_dataloader, unwrapped_model, self.optimizer, lr_scheduler_stage2
                    )
                    
                    if accelerator.state.num_processes > 1:
                        self.model = torch.nn.parallel.DistributedDataParallel(
                            accelerator.unwrap_model(self.model),
                            device_ids=[self.model.device],
                            find_unused_parameters=True
                        )
                    
                    logger.info("Dataset, normalizer, optimizer rebuilt for Stage 2!")
                
                # ========= eval end for this epoch ==========
                # end of epoch
                # log of last step is combined with validation
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()
        logger.info("Training completed!")
