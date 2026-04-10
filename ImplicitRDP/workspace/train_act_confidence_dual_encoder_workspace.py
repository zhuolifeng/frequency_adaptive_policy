from ImplicitRDP.workspace.train_hybrid_diffusion_act_confidence_workspace import (
    TrainHybridDiffusionACTConfidenceWorkspace,
)
from loguru import logger

class TrainACTConfidenceDualEncoderWorkspace(TrainHybridDiffusionACTConfidenceWorkspace):
    def __init__(self, cfg):
        super().__init__(cfg)
        logger.info("TrainACTConfidenceDualEncoderWorkspace initialized (dual encoder mode)")
