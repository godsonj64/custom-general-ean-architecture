import argparse
import os

import torch
import torch.nn as nn

from .model import build_model
from .utils import apply_overrides, ensure_dir, load_config, resolve_device


class _LogitsWrapper(nn.Module):
    """Wraps a model so its forward returns only the logits tensor (for clean ONNX export)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["logits"]


def export(config, checkpoint_path):
    """Export the trained model to PyTorch (.pt) and ONNX (.onnx) formats."""
    device = resolve_device(config.get("device", "auto"))
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" in ckpt:
        config = ckpt["config"]
    num_classes = ckpt.get("num_classes") or config["model"].get("num_classes")
    if config["model"].get("num_classes") is None:
        config["model"]["num_classes"] = num_classes

    model = build_model(config, num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ecfg = config["export"]
    out_dir = ensure_dir(ecfg["output_dir"])
    formats = ecfg.get("formats", ["pytorch", "onnx"])
    image_size = config["data"]["image_size"]
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    if "pytorch" in formats:
        pt_path = os.path.join(out_dir, "model.pt")
        torch.save({"model_state": model.state_dict(), "config": config,
                    "num_classes": num_classes}, pt_path)
        print(f"[export] Saved PyTorch checkpoint to {pt_path}")

    if "onnx" in formats:
        onnx_path = os.path.join(out_dir, "model.onnx")
        wrapped = _LogitsWrapper(model)
        torch.onnx.export(
            wrapped,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["logits"],
            opset_version=ecfg.get("opset", 17),
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        )
        print(f"[export] Saved ONNX model to {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="Export a trained model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="runs/best.pt")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.override)
    export(config, args.checkpoint)


if __name__ == "__main__":
    main()
