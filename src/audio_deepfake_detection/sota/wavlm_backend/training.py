from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from audio_deepfake_detection.metrics.eer import compute_eer

from .model import WavLMWA


def create_weighted_cross_entropy(
    class_weights: tuple[float, float] = (9.0, 1.0),
    *,
    device: torch.device | str | None = None,
) -> nn.CrossEntropyLoss:
    """Create paper-specified CE: bona fide=9, spoof=1."""
    if len(class_weights) != 2:
        raise ValueError("Exactly two class weights are required")

    weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    return nn.CrossEntropyLoss(weight=weights)


def create_optimizer_and_scheduler(
    model: WavLMWA,
    *,
    encoder_lr: float = 2.0e-5,
    backend_lr: float = 5.0e-3,
    lr_gamma: float = 0.95,
    layerwise_lr_decay: float | None = None,
) -> tuple[Optimizer, ExponentialLR]:
    """Create Adam groups for WavLM and the WA back-end.

    If layer-wise decay is enabled, Transformer layer 12 receives
    ``encoder_lr`` and each lower layer receives one additional decay
    factor. CNN/projection/encoder-input parameters receive the lowest
    rate. The paper does not report its decay factor, so the default is
    deliberately disabled.
    """
    if encoder_lr <= 0 or backend_lr <= 0:
        raise ValueError("Learning rates must be positive")

    if not 0 < lr_gamma <= 1:
        raise ValueError("lr_gamma must satisfy 0 < gamma <= 1")

    parameter_groups: list[dict] = []

    if layerwise_lr_decay is None:
        parameter_groups.append(
            {
                "params": list(model.wavlm.parameters()),
                "lr": encoder_lr,
                "name": "wavlm",
            }
        )
    else:
        if not 0 < layerwise_lr_decay <= 1:
            raise ValueError(
                "layerwise_lr_decay must satisfy 0 < decay <= 1"
            )

        layers = model.wavlm.encoder.layers
        layer_parameter_ids = {
            id(parameter)
            for layer in layers
            for parameter in layer.parameters()
        }
        base_parameters = [
            parameter
            for parameter in model.wavlm.parameters()
            if id(parameter) not in layer_parameter_ids
        ]

        parameter_groups.append(
            {
                "params": base_parameters,
                "lr": encoder_lr
                * layerwise_lr_decay ** len(layers),
                "name": "wavlm_base",
            }
        )

        for layer_index, layer in enumerate(layers):
            depth_from_top = len(layers) - 1 - layer_index
            parameter_groups.append(
                {
                    "params": list(layer.parameters()),
                    "lr": encoder_lr
                    * layerwise_lr_decay ** depth_from_top,
                    "name": f"wavlm_layer_{layer_index}",
                }
            )

    parameter_groups.append(
        {
            "params": list(model.backend.parameters()),
            "lr": backend_lr,
            "name": "wa_backend",
        }
    )

    optimizer = Adam(parameter_groups)
    scheduler = ExponentialLR(optimizer, gamma=lr_gamma)

    return optimizer, scheduler


def train_one_epoch(
    model: WavLMWA,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    *,
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    bf16: bool = False,
) -> float:
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be positive"
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_examples = 0
    num_batches = len(loader)
    autocast_enabled = bf16 and device.type == "cuda"

    for batch_index, batch in enumerate(loader):
        waveform = batch["waveform"].to(
            device,
            non_blocking=True,
        )
        attention_mask = batch["attention_mask"].to(
            device,
            non_blocking=True,
        )
        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            logits = model(waveform, attention_mask)
            loss = loss_function(logits, labels)

        accumulation_group_start = (
            batch_index // gradient_accumulation_steps
        ) * gradient_accumulation_steps
        accumulation_group_size = min(
            gradient_accumulation_steps,
            num_batches - accumulation_group_start,
        )
        (loss / accumulation_group_size).backward()

        should_step = (
            (batch_index + 1) % gradient_accumulation_steps == 0
            or batch_index + 1 == num_batches
        )

        if should_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach()) * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Training loader produced no examples")

    return total_loss / total_examples


@torch.no_grad()
def score_loader(
    model: WavLMWA,
    loader: DataLoader,
    *,
    device: torch.device,
    bf16: bool = False,
) -> tuple[list[int], list[float]]:
    model.eval()

    labels: list[int] = []
    scores: list[float] = []
    autocast_enabled = bf16 and device.type == "cuda"

    for batch in loader:
        waveform = batch["waveform"].to(
            device,
            non_blocking=True,
        )
        attention_mask = batch["attention_mask"].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            batch_scores = model.score(
                waveform,
                attention_mask,
            )

        labels.extend(int(value) for value in batch["label"].tolist())
        scores.extend(
            float(value)
            for value in batch_scores.float().cpu().tolist()
        )

    if not labels:
        raise ValueError("Evaluation loader produced no examples")

    return labels, scores


def score_source_dev_domains(
    model: WavLMWA,
    loaders: Mapping[str, DataLoader],
    *,
    device: torch.device,
    bf16: bool = False,
) -> dict[str, dict]:
    results: dict[str, dict] = {}

    for dataset_name, loader in loaders.items():
        labels, scores = score_loader(
            model,
            loader,
            device=device,
            bf16=bf16,
        )
        eer, threshold = compute_eer(labels, scores)
        results[dataset_name] = {
            "labels": labels,
            "scores": scores,
            "eer": eer,
            "eer_threshold_diagnostic": threshold,
        }

    return results


def macro_source_dev_eer(
    domain_results: Mapping[str, Mapping],
) -> float:
    if not domain_results:
        raise ValueError("At least one source-dev domain is required")

    return sum(
        float(result["eer"])
        for result in domain_results.values()
    ) / len(domain_results)


@dataclass
class EarlyStoppingState:
    patience: int = 5
    best_metric: float = float("inf")
    epochs_without_improvement: int = 0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be positive")

    def update(self, metric: float) -> bool:
        improved = metric < self.best_metric

        if improved:
            self.best_metric = float(metric)
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        return improved

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience


def save_checkpoint(
    path: str | Path,
    *,
    model: WavLMWA,
    optimizer: Optimizer,
    scheduler: ExponentialLR,
    epoch: int,
    early_stopping: EarlyStoppingState,
    config: Mapping,
    checkpoint_role: str,
) -> None:
    if checkpoint_role not in {"best", "last"}:
        raise ValueError("checkpoint_role must be 'best' or 'last'")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "checkpoint_role": checkpoint_role,
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "early_stopping": asdict(early_stopping),
            "config": dict(config),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    *,
    model: WavLMWA,
    optimizer: Optimizer | None = None,
    scheduler: ExponentialLR | None = None,
    map_location: torch.device | str = "cpu",
    expected_config: Mapping | None = None,
    expected_checkpoint_role: str | None = None,
) -> dict:
    checkpoint = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=False,
    )

    if (
        expected_checkpoint_role is not None
        and checkpoint.get("checkpoint_role")
        != expected_checkpoint_role
    ):
        raise ValueError(
            "Expected a "
            f"{expected_checkpoint_role!r} checkpoint, got "
            f"{checkpoint.get('checkpoint_role')!r}"
        )

    if expected_config is not None:
        saved_config = checkpoint.get("config", {})
        semantic_keys = (
            "model_name",
            "revision",
            "layer_weight_normalization",
            "input_normalization",
            "apply_spec_augment",
            "sample_rate",
            "lodo_fold",
            "held_out_dataset",
        )
        mismatches = {
            key: (saved_config.get(key), expected_config.get(key))
            for key in semantic_keys
            if saved_config.get(key) != expected_config.get(key)
        }

        if mismatches:
            raise ValueError(
                "Checkpoint/config semantic mismatch: "
                f"{mismatches}"
            )

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint
