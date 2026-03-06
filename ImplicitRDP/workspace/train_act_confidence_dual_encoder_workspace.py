from ImplicitRDP.workspace.train_hybrid_diffusion_act_workspace import (
    TrainHybridDiffusionACTWorkspace,
)
from loguru import logger

class TrainACTConfidenceDualEncoderWorkspace(TrainHybridDiffusionACTWorkspace):
    """
    Training workspace for ACT with dual encoders:
    - ACT encoder: encodes observations for action prediction
    - Confidence encoder: encodes observations for confidence estimation
    
    This workspace inherits from TrainHybridDiffusionACTWorkspace and uses
    the ACTConfidenceDualEncoderPolicy which has two separate encoders.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        logger.info("TrainACTConfidenceDualEncoderWorkspace initialized (dual encoder mode)")
