from ImplicitRDP.workspace.train_hybrid_diffusion_act_confidence_workspace import (
    TrainHybridDiffusionACTConfidenceWorkspace,
)

class TrainACTConfidenceWorkspace(TrainHybridDiffusionACTConfidenceWorkspace):
    def __init__(self, cfg):
        super().__init__(cfg)
