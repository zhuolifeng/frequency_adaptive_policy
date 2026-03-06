from ImplicitRDP.workspace.train_hybrid_diffusion_act_workspace import (
    TrainHybridDiffusionACTWorkspace,
)

class TrainACTConfidenceWorkspace(TrainHybridDiffusionACTWorkspace):
    def __init__(self, cfg):
        super().__init__(cfg)
