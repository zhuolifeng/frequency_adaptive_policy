# %%
import pathlib
import torch
import dill
import hydra
from omegaconf import OmegaConf
from ImplicitRDP.workspace.base_workspace import BaseWorkspace
from ImplicitRDP.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from ImplicitRDP.policy.diffusion_transformer_image_policy import DiffusionTransformerImagePolicy
from ImplicitRDP.policy.fm_policy import FlowMatchingTransformerImagePolicy
from ImplicitRDP.policy.fm_multifreq_policy import FMMultiFreqPolicy
from ImplicitRDP.policy.fm_multifreq_confidence_policy import FMMultiFreqConfidencePolicy
from ImplicitRDP.policy.fm_multifreq_rdt_action_chunk_policy import FMMultiFreqRDTActionChunkPolicy
from ImplicitRDP.policy.dp_ctm_multifreq_policy import DPCTMMultiFreqPolicy
from ImplicitRDP.policy.act_50hz_policy import ACT50HzPolicy
from ImplicitRDP.policy.hybrid_diffusion_act_confidence_policy import HybridDiffusionACTConfidencePolicy
from ImplicitRDP.policy.hybrid_diffusion_act_confidence_stagewise_policy import HybridDiffusionACTConfidenceStagewisePolicy

import os
from loguru import logger
import psutil

# add this to prevent assigning too may threads when using numpy
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"

import cv2
# add this to prevent assigning too may threads when using open-cv
cv2.setNumThreads(12)

# Get the total number of CPU cores
total_cores = psutil.cpu_count()
# Define the number of cores you want to bind to
num_cores_to_bind = 24
# Calculate the indices of the first ten cores
# Ensure the number of cores to bind does not exceed the total number of cores
cores_to_bind = set(range(min(num_cores_to_bind, total_cores)))
# Set CPU affinity for the current process to the first ten cores
os.sched_setaffinity(0, cores_to_bind)

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'ImplicitRDP', 'config')),
    config_name="train_diffusion_unet_real_image_workspace"
)
def main(cfg):
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    # load checkpoint
    ckpt_path = cfg.ckpt_path
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    workspace.load_payload(payload, exclude_keys=["optimizer", "scaler"], include_keys=None, strict=False)

    # hacks for method-specific setup.
    if 'unet' in cfg.name:
        # diffusion unet model
        policy: DiffusionUnetImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        if 'latent' in cfg.name:
            policy.at.set_normalizer(policy.normalizer)

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 8  # DDIM inference iterations
    elif isinstance(policy, FMMultiFreqConfidencePolicy):
        # FM Multi-Freq + Confidence policy (must check before FMMultiFreqPolicy since it's a subclass)
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        device = torch.device('cuda')
        policy.eval().to(device)
        policy.num_inference_steps = cfg.policy.get('num_inference_steps', 5)
        # inference head selection
        policy.inference_head = cfg.policy.get('inference_head', '10hz')
        policy.use_confidence_based_head_selection = cfg.policy.get('use_confidence_based_head_selection', True)
        policy.confidence_threshold_for_head_selection = cfg.policy.get('confidence_threshold_for_head_selection', 0.0)
        logger.info(f"FMMultiFreqConfidencePolicy: inference_head={policy.inference_head}, "
                    f"use_confidence={policy.use_confidence_based_head_selection}, "
                    f"threshold={policy.confidence_threshold_for_head_selection}")
    elif isinstance(policy, DPCTMMultiFreqPolicy):
        # DP-CTM Multi-Frequency policy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        device = torch.device('cuda')
        policy.eval().to(device)
        # Drop frozen teacher to save GPU memory
        if hasattr(policy, 'drop_teacher'):
            policy.drop_teacher()
        # inference_head: "10hz" (multi-step Karras ODE) or "50hz" (CTM single/multi-step)
        inference_head = cfg.policy.get('inference_head', '50hz')
        policy.inference_head = inference_head
        # training_stage: 1=EDM, 2=CTM
        training_stage = cfg.policy.get('training_stage', 2)
        policy.set_training_stage(int(training_stage))
        # 10hz inference steps (Karras ODE bins)
        num_inference_steps = cfg.policy.get('num_inference_steps', None)
        if num_inference_steps is not None:
            policy.karras_scheduler.bins = int(num_inference_steps)
            logger.info(f"DPCTMMultiFreqPolicy: karras_scheduler.bins -> {int(num_inference_steps)}")
        # 50hz inference steps: 1=single-step CTM, >1=native CTM multi-step
        num_inference_steps_50hz = cfg.policy.get('num_inference_steps_50hz', 1)
        policy.num_inference_steps_50hz = int(num_inference_steps_50hz)
        logger.info(f"DPCTMMultiFreqPolicy: inference_head={inference_head}, "
                    f"training_stage={training_stage}, "
                    f"num_inference_steps_50hz={num_inference_steps_50hz}")
    elif isinstance(policy, FMMultiFreqRDTActionChunkPolicy):
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        device = torch.device('cuda')
        policy.eval().to(device)
        policy.inference_head = cfg.policy.get('inference_head', '50hz')
        policy.num_inference_steps = cfg.policy.get('num_inference_steps', 10)
        policy.set_training_stage(int(cfg.policy.get('training_stage', 2)))
        logger.info(
            f"FMMultiFreqRDTActionChunkPolicy: inference_head={policy.inference_head}, "
            f"num_inference_steps={policy.num_inference_steps}"
        )
    elif isinstance(policy, FMMultiFreqPolicy):
        # FM Multi-Freq policy (no confidence head)
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        device = torch.device('cuda')
        policy.eval().to(device)
    elif isinstance(policy, ACT50HzPolicy):
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        device = torch.device('cuda')
        policy.eval().to(device)
        policy.action_start_50hz = int(cfg.policy.get('action_start_50hz', policy.action_start_50hz))
        logger.info(
            f"ACT50HzPolicy: horizon={policy.action_horizon}, "
            f"action_start_50hz={policy.action_start_50hz}"
        )
    elif isinstance(policy, FlowMatchingTransformerImagePolicy) or 'fm' in cfg.name:
        # flow matching transformer model
        policy: FlowMatchingTransformerImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)
        #logger.info(f"FlowMatchingTransformerImagePolicy: num_inference_steps={policy.num_inference_steps}")
    elif 'transformer' in cfg.name:
        # diffusion transformer model
        policy: DiffusionTransformerImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 5  # DDIM inference iterations
    elif isinstance(policy, HybridDiffusionACTConfidencePolicy):
            # Hybrid Diffusion + ACT model
            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            device = torch.device('cuda')
            policy.eval().to(device)

            # set inference params
            policy.num_inference_steps = 5  # DDIM inference iterations (same as transformer)
            
            # set inference head (default to 'diffusion' if not specified)
            inference_head = cfg.policy.get('inference_head', 'diffusion')
            logger.info("inference head:", inference_head)
            if inference_head not in ['diffusion', 'act']:
                raise ValueError(f"inference_head must be 'diffusion' or 'act', got '{inference_head}'")
            policy.inference_head = inference_head
    elif isinstance(policy, HybridDiffusionACTConfidenceStagewisePolicy):
            # Hybrid Diffusion + ACT model
            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model            
            device = torch.device('cuda')
            policy.eval().to(device)

            # set inference params
            policy.num_inference_steps = 5  # DDIM inference iterations (same as transformer)
            
    else:
        raise NotImplementedError

    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner)
    env_runner.run(policy)


# %%
if __name__ == '__main__':
    main()
