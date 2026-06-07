import argparse

import torch

from .dataset import build_dataloaders
from .model import build_model
from .train import evaluate_loader
from .utils import apply_overrides, load_config, resolve_device, set_seed


def evaluate(config, checkpoint_path):
    """Load a checkpoint and print accuracy and macro F1 on the validation set."""
    set_seed(config.get("seed", 42))
    device = resolve_device(config.get("device", "auto"))

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" in ckpt:
        config = ckpt["config"]
    num_classes = ckpt.get("num_classes")

    _, val_loader, inferred_classes, _ = build_dataloaders(config)
    if num_classes is None:
        num_classes = inferred_classes
    if config["model"].get("num_classes") is None:
        config["model"]["num_classes"] = num_classes

    model = build_model(config, num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])

    metrics = evaluate_loader(model, val_loader, device)
    print(f"accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="runs/best.pt")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.override)
    evaluate(config, args.checkpoint)


if __name__ == "__main__":
    main()
