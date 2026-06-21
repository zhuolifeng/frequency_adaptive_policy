import pathlib

import hydra
from omegaconf import OmegaConf

from ImplicitRDP.workspace.train_hybrid_diffusion_act_confidence_workspace import (
    TrainHybridDiffusionACTConfidenceWorkspace,
)


OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDP30HziMeanFlowWorkspace(TrainHybridDiffusionACTConfidenceWorkspace):
    pass


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainDP30HziMeanFlowWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
