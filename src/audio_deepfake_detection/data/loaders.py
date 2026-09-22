from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from torch.utils.data import DataLoader

from .collate import collate_audio_batch
from .datasets import ManifestDataset
from .samplers import DatasetClassBalancedSampler
from audio_deepfake_detection.training.reproducibility import (
    make_generator,
    seed_worker,
)


def create_train_loader(
    manifest_paths: Sequence[str | Path],
    *,
    batch_size: int,
    seed: int,
    max_seconds: float = 10.0,
    num_workers: int = 4,
    num_samples: int | None = None,
    pin_memory: bool = True,
) -> tuple[
    DataLoader,
    ManifestDataset,
    DatasetClassBalancedSampler,
]:
    """
    Create the common source-training DataLoader.

    Sampling policy:
        dataset uniformly
        -> class uniformly
        -> utterance uniformly

    Audio is cropped at its native sample rate. Model-specific
    resampling is intentionally performed downstream.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    dataset = ManifestDataset(
        manifest_paths,
        load_audio=True,
        max_seconds=max_seconds,
        random_crop=True,
        target_sample_rate=None,
    )

    sampler = DatasetClassBalancedSampler(
        dataset,
        num_samples=num_samples,
        seed=seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_audio_batch,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
        persistent_workers=(num_workers > 0),
    )

    return loader, dataset, sampler


def create_eval_loader(
    manifest_paths: str | Path | Sequence[str | Path],
    *,
    batch_size: int,
    max_seconds: float = 10.0,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, ManifestDataset]:
    """
    Create deterministic source-dev or target evaluation loader.

    Evaluation uses centre cropping and sequential ordering.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    dataset = ManifestDataset(
        manifest_paths,
        load_audio=True,
        max_seconds=max_seconds,
        random_crop=False,
        target_sample_rate=None,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_audio_batch,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return loader, dataset
