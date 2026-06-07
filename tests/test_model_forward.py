import torch

from src.model import BaselineCNN, GeneralEAN, build_model


def test_general_ean_forward_shapes():
    """General EAN should output logits, latent, and a next-latent prediction."""
    num_classes = 5
    model = GeneralEAN(
        num_classes=num_classes,
        latent_dim=64,
        num_concepts=4,
        top_k=2,
        concept_dim=32,
        pretrained=False,
    )
    model.eval()
    x = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out["logits"].shape == (3, num_classes)
    assert out["latent"].shape == (3, 64)
    assert out["predicted_next"].shape == (3, 64)
    assert out["route_weights"].shape == (3, 4)
    # top_k routing means at most 2 non-zero concepts per sample
    assert (out["route_weights"] > 0).sum(dim=1).max().item() <= 2


def test_baseline_cnn_forward_shapes():
    """The baseline CNN should produce one logit vector per input image."""
    model = BaselineCNN(num_classes=3)
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out["logits"].shape == (2, 3)


def test_build_model_selects_baseline():
    """build_model should return the baseline when configured to do so."""
    config = {
        "model": {
            "name": "baseline_cnn",
            "latent_dim": 64,
            "num_concepts": 4,
            "top_k": 2,
            "concept_dim": 32,
            "pretrained": False,
        }
    }
    model = build_model(config, num_classes=4)
    assert isinstance(model, BaselineCNN)
