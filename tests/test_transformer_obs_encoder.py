import pytest
import timm
import torch
from omegaconf import OmegaConf
from torch import nn
from torchvision import transforms as T

from ImplicitRDP.model.vision.transformer_obs_encoder import TransformerObsEncoder


@pytest.mark.parametrize(("crop_type", "ratio"), [("CenterCrop", 1.0), ("RandomCrop", 0.75)])
def test_encoder_augmentation_is_training_only(monkeypatch, crop_type, ratio):
    # An identity backbone exposes the exact RGB values supplied to it.
    monkeypatch.setattr(timm, "create_model", lambda **kwargs: nn.Identity())
    image = torch.linspace(0, 1, 3 * 24 * 40).reshape(1, 3, 24, 40)
    normalize = T.Normalize([0.1, 0.2, 0.3], [0.5, 0.5, 0.5])
    encoder = TransformerObsEncoder(
        shape_meta={"obs": {"camera": {"shape": image.shape[1:], "type": "rgb"}}},
        obs_horizon=1, model_name="identity", n_emb=3,
        transforms=[
            OmegaConf.create({"type": crop_type, "ratio": ratio}),
            T.RandomRotation(5),
            T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            normalize,
        ],
    )
    expected = nn.Sequential(
        T.Resize(224, antialias=True), T.CenterCrop(int(224 * ratio)),
        T.Resize(224, antialias=True), normalize,
    )(image).flatten(2).transpose(1, 2)
    obs = {"camera": image}
    encoder.eval()
    torch.testing.assert_close(encoder(obs), expected, rtol=0, atol=0)
    torch.testing.assert_close(encoder(obs), expected, rtol=0, atol=0)

    encoder.train()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        first = encoder(obs)
        torch.manual_seed(2)
        second = encoder(obs)
    assert not torch.equal(first, second)
