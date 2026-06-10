# %%
import pathlib
import torch
import dill
import hydra
from omegaconf import OmegaConf
from ImplicitRDP.workspace.base_workspace import BaseWorkspace
from ImplicitRDP.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from ImplicitRDP.policy.diffusion_transformer_image_policy import DiffusionTransformerImagePolicy
from ImplicitRDP.policy.dp_multifreq_cd_30hz_confidence_policy import DPMultiFreqCD30HzConfidencePolicy
from ImplicitRDP.policy.act_30hz_policy import ACT30HzPolicy
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
# Bind the eval entry process to logical CPU ids 0-17.
eval_start_core_id = 0
eval_num_cores = 18
cores_to_bind = set(range(eval_start_core_id, min(eval_start_core_id + eval_num_cores, total_cores)))
if not cores_to_bind:
    cores_to_bind = {max(0, total_cores - 1)}
# Set CPU affinity for the current process
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
    elif isinstance(policy, DPMultiFreqCD30HzConfidencePolicy):
        # DP Multi-Frequency CD 30Hz + Confidence policy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        device = torch.device('cuda')
        policy.eval().to(device)
        # training stage: 1=DP base, 2=confidence, 3=CD
        training_stage = int(cfg.policy.get('training_stage', 3))
        policy.set_training_stage(training_stage)
        # inference head: "10hz" or "53hz"
        policy.inference_head = cfg.policy.get('inference_head', '30hz')
        # per-head inference steps (stage 3 + 30hz + steps=1 triggers one-step jump)
        policy.num_inference_steps = int(cfg.policy.get('num_inference_steps', 5))
        policy.num_inference_steps_10hz = int(
            cfg.policy.get('num_inference_steps_10hz', policy.num_inference_steps)
        )
        policy.num_inference_steps_30hz = int(
            cfg.policy.get('num_inference_steps_30hz', policy.num_inference_steps)
        )
        # confidence-based head selection
        policy.use_confidence_based_head_selection = bool(
            cfg.policy.get('use_confidence_based_head_selection', False)
        )
        policy.confidence_threshold_for_head_selection = float(
            cfg.policy.get('confidence_threshold_for_head_selection', 0.0)
        )
        logger.info(
            f"DPMultiFreqCD30HzConfidencePolicy: training_stage={training_stage}, "
            f"inference_head={policy.inference_head}, "
            f"steps_10hz={policy.num_inference_steps_10hz}, "
            f"steps_30hz={policy.num_inference_steps_30hz}, "
            f"use_confidence={policy.use_confidence_based_head_selection}, "
            f"threshold={policy.confidence_threshold_for_head_selection}"
        )
    elif isinstance(policy, ACT30HzPolicy):
        # ACT 30Hz policy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        device = torch.device('cuda')
        policy.eval().to(device)
        logger.info(
            f"ACT30HzPolicy: horizon={getattr(policy, 'horizon', None)}, "
            f"action_start_30hz={getattr(policy, 'action_start_30hz', None)}, "
            f"n_action_steps={getattr(policy, 'n_action_steps', None)}"
        )

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
