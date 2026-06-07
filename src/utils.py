import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a nested dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(config: Dict[str, Any], overrides):
    """Apply dotted-key overrides like 'model.name=baseline_cnn' to the config."""
    if not overrides:
        return config
    for item in overrides:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parts = key.split(".")
        node = config
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(value)
    return config


def _coerce(value: str):
    """Convert a string override value into bool/int/float/None when possible."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def set_seed(seed: int) -> None:
    """Make runs reproducible by fixing all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    """Choose CPU or GPU based on config and availability."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute accuracy and macro F1 from true and predicted labels."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(acc), "f1": float(f1)}


def ensure_dir(path: str) -> str:
    """Create a directory if it does not exist and return its path."""
    os.makedirs(path, exist_ok=True)
    return path
