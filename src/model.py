from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConceptModule(nn.Module):
    """A small specialized expert that transforms the latent into a concept output."""

    def __init__(self, latent_dim: int, concept_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, concept_dim),
            nn.GELU(),
            nn.Linear(concept_dim, concept_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AbstractionField(nn.Module):
    """Projects the encoder latent into a refined abstract representation."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.proj(x))


class ConceptRouter(nn.Module):
    """Scores concepts and selects the top-k most relevant ones per sample."""

    def __init__(self, latent_dim: int, num_concepts: int, top_k: int):
        super().__init__()
        self.num_concepts = num_concepts
        self.top_k = min(top_k, num_concepts)
        self.gate = nn.Linear(latent_dim, num_concepts)

    def forward(self, x: torch.Tensor):
        """Return sparse routing weights (B, num_concepts) and raw logits."""
        logits = self.gate(x)
        topv, topi = torch.topk(logits, self.top_k, dim=-1)
        topw = F.softmax(topv, dim=-1)
        weights = torch.zeros_like(logits)
        weights.scatter_(-1, topi, topw)
        return weights, logits


class WorldModel(nn.Module):
    """Tiny predictor that anticipates the next-step latent (self-supervised signal)."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EvolutionController:
    """Tracks concept usage and resets under-used concepts to keep the population useful."""

    def __init__(self, num_concepts: int, prune_threshold: float, warmup_epochs: int):
        self.num_concepts = num_concepts
        self.prune_threshold = prune_threshold
        self.warmup_epochs = warmup_epochs
        self.usage = torch.zeros(num_concepts)
        self.total = 0.0

    def update(self, weights: torch.Tensor) -> None:
        """Accumulate how often each concept was activated."""
        active = (weights > 0).float().sum(dim=0).detach().cpu()
        self.usage += active
        self.total += weights.shape[0]

    def maybe_evolve(self, model: "GeneralEAN", epoch: int) -> int:
        """Reset weights of rarely-used concepts after warmup; returns count reset."""
        if epoch < self.warmup_epochs or self.total == 0:
            self._reset_stats()
            return 0
        fractions = self.usage / max(self.total, 1.0)
        reset = 0
        for idx, frac in enumerate(fractions.tolist()):
            if frac < self.prune_threshold:
                self._reinit_concept(model.concepts[idx])
                reset += 1
        self._reset_stats()
        return reset

    def _reset_stats(self) -> None:
        self.usage = torch.zeros(self.num_concepts)
        self.total = 0.0

    @staticmethod
    def _reinit_concept(module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class GeneralEAN(nn.Module):
    """Encoder + Abstraction Field + Concept Router + concept modules + world model."""

    def __init__(
        self,
        num_classes: int,
        latent_dim: int = 256,
        num_concepts: int = 8,
        top_k: int = 2,
        concept_dim: int = 128,
        pretrained: bool = True,
    ):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        feat_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.encoder = backbone
        self.to_latent = nn.Linear(feat_dim, latent_dim)

        self.abstraction = AbstractionField(latent_dim)
        self.router = ConceptRouter(latent_dim, num_concepts, top_k)
        self.concepts = nn.ModuleList(
            [ConceptModule(latent_dim, concept_dim) for _ in range(num_concepts)]
        )
        self.world_model = WorldModel(latent_dim)
        self.classifier = nn.Linear(concept_dim, num_classes)

        self.num_concepts = num_concepts
        self.latent_dim = latent_dim
        self.concept_dim = concept_dim

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.encoder(x)
        latent = self.to_latent(feats)
        abstract = self.abstraction(latent)

        weights, logits = self.router(abstract)  # (B, C)
        concept_outs = torch.stack(
            [c(abstract) for c in self.concepts], dim=1
        )  # (B, C, concept_dim)
        combined = (weights.unsqueeze(-1) * concept_outs).sum(dim=1)  # (B, concept_dim)

        predicted_next = self.world_model(abstract)
        logits_out = self.classifier(combined)

        return {
            "logits": logits_out,
            "latent": abstract,
            "predicted_next": predicted_next,
            "route_weights": weights,
            "route_logits": logits,
        }


class BaselineCNN(nn.Module):
    """A small convolutional network trained from scratch, used as a comparison baseline."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return {"logits": self.classifier(h)}


def build_model(config, num_classes: int):
    """Construct either the General EAN or the baseline CNN from config."""
    mcfg = config["model"]
    name = mcfg.get("name", "general_ean")
    if name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)
    return GeneralEAN(
        num_classes=num_classes,
        latent_dim=mcfg["latent_dim"],
        num_concepts=mcfg["num_concepts"],
        top_k=mcfg["top_k"],
        concept_dim=mcfg["concept_dim"],
        pretrained=mcfg.get("pretrained", True),
    )
