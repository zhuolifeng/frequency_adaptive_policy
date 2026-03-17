from ImplicitRDP.workspace.train_hybrid_diffusion_act_workspace import (
    TrainHybridDiffusionACTWorkspace,
)
from loguru import logger

class TrainACTConfidenceDualEncoderWorkspace(TrainHybridDiffusionACTWorkspace):
    def __init__(self, cfg):
        super().__init__(cfg)
        logger.info("TrainACTConfidenceDualEncoderWorkspace initialized (dual encoder mode)")
