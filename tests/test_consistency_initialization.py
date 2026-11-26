import copy
from pathlib import Path

import dill
import pytest
import torch
from hydra import compose, initialize_config_dir

from ImplicitRDP.workspace.train_dp_30hz_consistency_workspace import TrainDP30HzConsistencyWorkspace


@pytest.fixture
def workspace(tmp_path):
    config_dir = Path(__file__).resolve().parents[1] / "ImplicitRDP" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train_dp_30hz_consistency_workspace", overrides=[
            "training.stage1_epochs=0", "logging.backend=none",
            "n_emb=8", "policy.model.n_layer=1", "policy.rnn_obs_encoder_hidden_dim=4",
        ])
    cfg.policy.obs_encoder = {"_target_": "torch.nn.Identity"}
    workspace = TrainDP30HzConsistencyWorkspace(cfg)
    workspace._output_dir = str(tmp_path)
    return workspace


@pytest.mark.parametrize("resume", [False, True])
def test_distillation_requires_teacher_before_loading_data(workspace, resume):
    workspace.cfg.training.resume = resume
    with pytest.raises(ValueError, match="Stage 2 needs a trained teacher"):
        workspace.run()


def stage1_checkpoint(workspace, with_ema):
    with torch.no_grad():
        workspace.model.model.head.bias.fill_(1)
        workspace.ema_model.model.head.bias.fill_(2)
    states = {"model": workspace.model.state_dict()}
    if with_ema:
        states["ema_model"] = workspace.ema_model.state_dict()
    return {
        "state_dicts": states,
        "pickles": {"current_stage": dill.dumps(1), "epoch": dill.dumps(4), "ema_optimization_step": dill.dumps(10)},
        "cfg": workspace.cfg,
    }


def assert_distillation_weights(workspace, expected):
    for policy in (workspace.teacher_model, workspace.model, workspace.ema_model):
        bias = policy.model.head.bias
        torch.testing.assert_close(bias, torch.full_like(bias, expected))
    assert not workspace.teacher_model.training
    assert not any(parameter.requires_grad for parameter in workspace.teacher_model.parameters())


@pytest.mark.parametrize("with_ema", [False, True])
def test_distillation_initializes_all_policies_from_teacher(workspace, tmp_path, with_ema):
    path = tmp_path / "stage1.ckpt"
    torch.save(stage1_checkpoint(workspace, with_ema), path, pickle_module=dill)
    workspace.cfg.training.pretrained_ckpt_path = str(path)
    workspace._load_pretrained_if_needed(workspace.cfg)
    assert_distillation_weights(workspace, 2 if with_ema else 1)


def test_partial_teacher_checkpoint_fails(workspace, tmp_path):
    path = tmp_path / "incomplete.ckpt"
    torch.save({"state_dicts": {"model": {"model.head.bias": workspace.model.model.head.bias}}}, path)
    workspace.cfg.training.pretrained_ckpt_path = str(path)
    with pytest.raises(RuntimeError, match="Missing key"):
        workspace._load_pretrained_if_needed(workspace.cfg)


def test_stage1_warm_start_copies_model_to_missing_ema(workspace, tmp_path):
    path = tmp_path / "model_only.ckpt"
    torch.save(stage1_checkpoint(workspace, with_ema=False), path, pickle_module=dill)
    workspace.cfg.training.stage1_epochs = 2
    workspace.cfg.training.pretrained_ckpt_path = str(path)
    warmed = TrainDP30HzConsistencyWorkspace(workspace.cfg)
    warmed._load_pretrained_if_needed(workspace.cfg)
    for policy in (warmed.model, warmed.ema_model):
        bias = policy.model.head.bias
        torch.testing.assert_close(bias, torch.ones_like(bias))


@pytest.mark.parametrize("with_ema,debug", [(False, False), (True, False), (True, True)])
def test_resume_at_stage_boundary_initializes_distillation(workspace, tmp_path, with_ema, debug):
    payload = stage1_checkpoint(workspace, with_ema)
    path = tmp_path / "stage1.ckpt"
    torch.save(payload, path, pickle_module=dill)
    cfg = copy.deepcopy(workspace.cfg)
    cfg.training.stage1_epochs = 8 if debug else 1
    cfg.training.stage2_epochs = 3
    cfg.training.debug = debug
    resumed = TrainDP30HzConsistencyWorkspace(cfg)
    resumed._resume_training(cfg, resumed._make_accelerator(cfg), path)

    assert resumed.current_stage == resumed.model.training_stage == resumed.ema_model.training_stage == 2
    assert resumed.teacher_model.training_stage == 1
    assert resumed.ema_optimization_step == 0
    assert resumed._remaining_stages() == [2] * (1 if debug else 3)
    assert_distillation_weights(resumed, 2 if with_ema else 1)
    optimizer_params = {id(p) for group in resumed.optimizer.param_groups for p in group["params"]}
    assert optimizer_params == {id(p) for p in resumed.model.model.parameters() if p.requires_grad}
    assert {group["lr"] for group in resumed.optimizer.param_groups} == {cfg.optimizer.stage2_lr}

    # After one student epoch, changing the teacher epoch budget again must not
    # change the number of completed student epochs in the next resume.
    resumed.epoch += 1
    resumed.stage2_epochs_completed += 1
    next_path = tmp_path / "stage2.ckpt"
    resumed.save_checkpoint(path=next_path, use_thread=False)
    cfg.training.stage1_epochs = 20
    cfg.training.debug = False
    restarted = TrainDP30HzConsistencyWorkspace(cfg)
    restarted._resume_training(cfg, restarted._make_accelerator(cfg), next_path)
    assert restarted._remaining_stages() == [2, 2]
    assert_distillation_weights(restarted, 2 if with_ema else 1)


@pytest.mark.parametrize("missing,match", [
    ("current_stage", "current_stage metadata"), ("epoch", "epoch metadata"), ("model", "trained model or EMA"),
])
def test_incomplete_stage1_resume_fails(workspace, tmp_path, missing, match):
    payload = stage1_checkpoint(workspace, with_ema=False)
    if missing == "model":
        del payload["state_dicts"][missing]
    else:
        del payload["pickles"][missing]
    path = tmp_path / "incomplete.ckpt"
    torch.save(payload, path, pickle_module=dill)
    with pytest.raises(ValueError, match=match):
        workspace._resume_training(workspace.cfg, workspace._make_accelerator(workspace.cfg), path)


def test_legacy_stage2_resume_uses_original_stage1_epoch_budget(workspace, tmp_path):
    payload = stage1_checkpoint(workspace, with_ema=True)
    payload["pickles"]["current_stage"] = dill.dumps(2)
    payload["state_dicts"]["teacher_model"] = workspace.teacher_model.state_dict()
    payload["cfg"].training.stage1_epochs = 3
    path = tmp_path / "legacy.ckpt"
    torch.save(payload, path, pickle_module=dill)
    workspace.cfg.training.stage1_epochs = 0
    workspace.cfg.training.stage2_epochs = 3
    resumed = TrainDP30HzConsistencyWorkspace(workspace.cfg)
    resumed._resume_training(workspace.cfg, resumed._make_accelerator(workspace.cfg), path)
    assert resumed.stage2_epochs_completed == 1
    assert resumed._remaining_stages() == [2, 2]
