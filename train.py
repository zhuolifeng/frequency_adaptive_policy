"""Train a policy using one of the workspace configs in ImplicitRDP/config."""

import pathlib
import sys

import hydra
from omegaconf import DictConfig, OmegaConf


# Workspace configs use this resolver for arithmetic on horizons and step counts.
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent / "ImplicitRDP" / "config"),
)
def main(cfg: DictConfig):
    # Resolve timestamps once so all output paths agree.
    OmegaConf.resolve(cfg)
    workspace_class = hydra.utils.get_class(cfg._target_)
    workspace_class(cfg).run()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    main()
