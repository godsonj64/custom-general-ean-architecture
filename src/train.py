import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import build_dataloaders
from .model import EvolutionController, GeneralEAN, build_model
from .utils import (
    apply_overrides,
    compute_metrics,
    ensure_dir,
    load_config,
    resolve_device,
    set_seed,
)


def _world_model_loss(outputs) -> torch.Tensor:
    """Self-supervised loss: predict the next sample's latent within the batch."""
    latent = outputs["latent"]
    predicted = outputs["predicted_next"]
    if latent.shape[0] < 2:
        return latent.new_zeros(())
    target = latent.roll(-1, dims=0).detach()
    return F.mse_loss(predicted[:-1], target[:-1])


@torch.no_grad()
def evaluate_loader(model, loader, device):
    """Run the model over a loader and return accuracy and macro F1."""
    model.eval()
    all_true, all_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs["logits"].argmax(dim=1).cpu()
        all_true.extend(labels.tolist())
        all_pred.extend(preds.tolist())
    return compute_metrics(all_true, all_pred)


def train(config):
    """Full training routine: build data + model, optimize, and save best checkpoint."""
    set_seed(config.get("seed", 42))
    device = resolve_device(config.get("device", "auto"))

    train_loader, val_loader, num_classes, classes = build_dataloaders(config)
    if config["model"].get("num_classes") is None:
        config["model"]["num_classes"] = num_classes

    model = build_model(config, num_classes).to(device)

    tcfg = config["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg.get("label_smoothing", 0.0))

    is_ean = isinstance(model, GeneralEAN)
    wm_weight = config["model"].get("world_model_weight", 0.0)
    evo = None
    evo_cfg = config["model"].get("evolution", {})
    if is_ean and evo_cfg.get("enabled", False):
        evo = EvolutionController(
            num_concepts=config["model"]["num_concepts"],
            prune_threshold=evo_cfg.get("prune_threshold", 0.02),
            warmup_epochs=evo_cfg.get("warmup_epochs", 0),
        )

    output_dir = ensure_dir(tcfg["output_dir"])
    best_acc = -1.0
    total_epochs = tcfg["epochs"]

    for epoch in range(1, total_epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs["logits"], labels)
            if is_ean and wm_weight > 0:
                loss = loss + wm_weight * _world_model_loss(outputs)
            loss.backward()
            optimizer.step()

            if evo is not None:
                evo.update(outputs["route_weights"])

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        metrics = evaluate_loader(model, val_loader, device)
        acc = metrics["accuracy"]

        if evo is not None:
            evo.maybe_evolve(model, epoch)

        print(f"epoch {epoch}/{total_epochs} loss={avg_loss:.4f} val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "num_classes": num_classes,
                    "classes": classes,
                    "metrics": metrics,
                },
                os.path.join(output_dir, "best.pt"),
            )

    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Train the General EAN model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.override)
    train(config)


if __name__ == "__main__":
    main()
