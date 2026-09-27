from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

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
    epoch: int,
    gradient_accumulation_steps: int = 1,
    bf16: bool = False,
    progress_interval: int = 100,
) -> float:
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be positive"
        )

    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_examples = 0
    batches_processed = 0
    num_batches = len(loader)
    autocast_enabled = bf16 and device.type == "cuda"
    training_start = time.perf_counter()

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
        batches_processed = batch_index + 1

        if batches_processed % progress_interval == 0:
            elapsed = time.perf_counter() - training_start
            batches_per_second = batches_processed / elapsed
            examples_per_second = total_examples / elapsed
            remaining_batches = num_batches - batches_processed
            eta_seconds = remaining_batches / batches_per_second
            wavlm_lrs = [
                float(group["lr"])
                for group in optimizer.param_groups
                if str(group.get("name", "")).startswith("wavlm")
            ]
            backend_lrs = [
                float(group["lr"])
                for group in optimizer.param_groups
                if group.get("name") == "wa_backend"
            ]

            if not wavlm_lrs or len(backend_lrs) != 1:
                raise RuntimeError(
                    "Expected named WavLM and WA backend optimizer groups"
                )

            print(
                json.dumps(
                    {
                        "event": "train_progress",
                        "epoch": int(epoch),
                        "batch": batches_processed,
                        "total_batches": num_batches,
                        "percent_complete": (
                            100.0 * batches_processed / num_batches
                        ),
                        "examples": total_examples,
                        "mean_loss": total_loss / total_examples,
                        "elapsed_seconds": elapsed,
                        "batches_per_second": batches_per_second,
                        "examples_per_second": examples_per_second,
                        "eta_seconds": eta_seconds,
                        "wavlm_lr": max(wavlm_lrs),
                        "backend_lr": backend_lrs[0],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if total_examples == 0:
        raise ValueError("Training loader produced no examples")

    training_elapsed = time.perf_counter() - training_start
    mean_loss = total_loss / total_examples
    print(
        json.dumps(
            {
                "event": "train_epoch_summary",
                "epoch": int(epoch),
                "batches": batches_processed,
                "examples": total_examples,
                "mean_loss": mean_loss,
                "elapsed_seconds": training_elapsed,
                "batches_per_second": (
                    batches_processed / training_elapsed
                ),
                "examples_per_second": (
                    total_examples / training_elapsed
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    return mean_loss


@torch.inference_mode()
def _score_loader(
    model: WavLMWA,
    loader: DataLoader,
    *,
    device: torch.device,
    bf16: bool,
    include_metadata: bool,
    partition: str | None,
    progress_interval: int,
) -> tuple[
    list[int],
    list[float],
    list[dict[str, str | int | float]],
]:
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")

    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    records: list[dict[str, str | int | float]] = []
    autocast_enabled = bf16 and device.type == "cuda"
    total_batches = len(loader)
    total_examples = len(loader.dataset)
    scoring_start = time.perf_counter()

    for batch_index, batch in enumerate(loader):
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

        batch_labels = [
            int(value) for value in batch["label"].tolist()
        ]
        batch_score_values = [
            float(value)
            for value in batch_scores.float().cpu().tolist()
        ]

        if len(batch_labels) != len(batch_score_values):
            raise RuntimeError("Evaluation labels and scores are misaligned")

        labels.extend(batch_labels)
        scores.extend(batch_score_values)

        if include_metadata:
            metadata_fields = ("utterance_id", "dataset", "split")

            for field in metadata_fields:
                if (
                    field not in batch
                    or len(batch[field]) != len(batch_labels)
                ):
                    raise ValueError(
                        f"Evaluation batch has invalid {field!r} metadata"
                    )

            records.extend(
                {
                    "utterance_id": str(batch["utterance_id"][index]),
                    "dataset": str(batch["dataset"][index]),
                    "split": str(batch["split"][index]),
                    "label": label,
                    "score": score,
                }
                for index, (label, score) in enumerate(
                    zip(batch_labels, batch_score_values, strict=True)
                )
            )

        batches_completed = batch_index + 1
        examples_completed = len(labels)

        if partition is not None and (
            batches_completed % progress_interval == 0
        ):
            elapsed = time.perf_counter() - scoring_start
            examples_per_second = examples_completed / elapsed
            remaining_examples = total_examples - examples_completed
            print(
                json.dumps(
                    {
                        "event": "eval_progress",
                        "partition": partition,
                        "batch": batches_completed,
                        "total_batches": total_batches,
                        "examples_completed": examples_completed,
                        "total_examples": total_examples,
                        "percent_complete": (
                            100.0 * examples_completed / total_examples
                        ),
                        "elapsed_seconds": elapsed,
                        "examples_per_second": examples_per_second,
                        "eta_seconds": (
                            remaining_examples / examples_per_second
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if not labels:
        raise ValueError("Evaluation loader produced no examples")

    if include_metadata and len(records) != len(labels):
        raise RuntimeError("Evaluation metadata and scores are misaligned")

    if partition is not None:
        elapsed = time.perf_counter() - scoring_start
        print(
            json.dumps(
                {
                    "event": "eval_partition_summary",
                    "partition": partition,
                    "batches": total_batches,
                    "examples": len(labels),
                    "elapsed_seconds": elapsed,
                    "examples_per_second": len(labels) / elapsed,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    return labels, scores, records


@torch.inference_mode()
def score_loader(
    model: WavLMWA,
    loader: DataLoader,
    *,
    device: torch.device,
    bf16: bool = False,
) -> tuple[list[int], list[float]]:
    labels, scores, _records = _score_loader(
        model,
        loader,
        device=device,
        bf16=bf16,
        include_metadata=False,
        partition=None,
        progress_interval=1000,
    )

    return labels, scores


@torch.inference_mode()
def score_loader_with_metadata(
    model: WavLMWA,
    loader: DataLoader,
    *,
    device: torch.device,
    bf16: bool = False,
    partition: str,
    progress_interval: int = 1000,
) -> tuple[
    list[int],
    list[float],
    list[dict[str, str | int | float]],
]:
    """Score one ordered partition and retain aligned manifest metadata."""
    return _score_loader(
        model,
        loader,
        device=device,
        bf16=bf16,
        include_metadata=True,
        partition=partition,
        progress_interval=progress_interval,
    )


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
