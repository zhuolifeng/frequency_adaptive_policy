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


class TrainHybridDiffusionACTConfidenceWorkspace(BaseWorkspace):
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
        
        accelerator = Accelerator(log_with='wandb')
        cuda_count = accelerator.num_processes

        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

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

        dataset_resume_params_path = os.path.join(self.output_dir, 'dataset_resume_params.pkl')
        if accelerator.is_main_process:
            dataset_resume_params = dataset.get_resume_params()
            with open(dataset_resume_params_path, 'wb') as f:
                pickle.dump(dataset_resume_params, f)

        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            if cfg.training.pretrained_ckpt_path is not None:
                normalizer_state_dict = normalizer.state_dict()
                current_normalizer_state_dict = self.model.normalizer.state_dict()
                for key in current_normalizer_state_dict.keys():
                    normalizer_state_dict[key] = current_normalizer_state_dict[key]
                normalizer.load_state_dict(normalizer_state_dict)
            pickle.dump(normalizer, open(normalizer_path, 'wb'))

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

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
        
        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        train_sampling_batch = None

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        logger.info("Starting training...")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                self.model.update_progress(local_epoch_idx / cfg.training.num_epochs)
                print(f"local_epoch_idx: {local_epoch_idx}, cfg.training.num_epochs: {cfg.training.num_epochs}")
                # Update epoch for stage-based training
                if hasattr(self.model, 'set_epoch'):
                    self.model.set_epoch(self.epoch)
                
                # 打印当前训练阶段信息
                if hasattr(self.model, 'stage1_epochs'):
                    if self.epoch < self.model.stage1_epochs:
                        stage = 'Stage 1 (ACT + Confidence only)'
                        dp_weight = 0.0
                    else:
                        stage = 'Stage 2 (All components)'
                        dp_weight = 1.0
                    if self.epoch % 10 == 0 or self.epoch == self.model.stage1_epochs - 1:
                        logger.info(f"Epoch {self.epoch}: {stage}, DP loss weight = {dp_weight}")
                
                self.model.train()
                
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
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
                            raw_loss, loss_dict = self.model.compute_loss(batch)
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
                            'lr': lr_scheduler.get_last_lr()[0]
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
                
                # ========= eval end for this epoch ==========
                # end of epoch
                # log of last step is combined with validation
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()
        logger.info("Training completed!")

