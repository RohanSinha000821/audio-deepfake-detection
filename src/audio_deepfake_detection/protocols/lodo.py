from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ManifestRef:
    dataset: str
    split: str
    path: str


@dataclass(frozen=True)
class LODOFold:
    fold: str
    held_out_dataset: str
    source_train: tuple[ManifestRef, ...]
    source_dev: tuple[ManifestRef, ...]
    target_primary: ManifestRef
    target_secondary: ManifestRef | None = None


def _manifest_ref(data: dict) -> ManifestRef:
    return ManifestRef(
        dataset=str(data["dataset"]),
        split=str(data["split"]),
        path=str(data["path"]),
    )


def load_lodo_config(
    path: str | Path,
    *,
    check_files: bool = False,
) -> LODOFold:
    path = Path(path)

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    secondary = raw.get("target_secondary")

    fold = LODOFold(
        fold=str(raw["fold"]),
        held_out_dataset=str(raw["held_out_dataset"]),
        source_train=tuple(
            _manifest_ref(item)
            for item in raw["source_train"]
        ),
        source_dev=tuple(
            _manifest_ref(item)
            for item in raw["source_dev"]
        ),
        target_primary=_manifest_ref(
            raw["target_primary"]
        ),
        target_secondary=(
            _manifest_ref(secondary)
            if secondary is not None
            else None
        ),
    )

    validate_lodo_fold(
        fold,
        check_files=check_files,
    )

    return fold


def validate_lodo_fold(
    fold: LODOFold,
    *,
    check_files: bool = False,
) -> None:
    train_datasets = [
        ref.dataset
        for ref in fold.source_train
    ]

    dev_datasets = [
        ref.dataset
        for ref in fold.source_dev
    ]

    if len(train_datasets) != len(set(train_datasets)):
        raise ValueError(
            f"{fold.fold}: duplicate source_train dataset"
        )

    if len(dev_datasets) != len(set(dev_datasets)):
        raise ValueError(
            f"{fold.fold}: duplicate source_dev dataset"
        )

    if set(train_datasets) != set(dev_datasets):
        raise ValueError(
            f"{fold.fold}: source train/dev datasets differ"
        )

    if fold.held_out_dataset in train_datasets:
        raise ValueError(
            f"{fold.fold}: held-out dataset appears "
            "in source_train"
        )

    if fold.held_out_dataset in dev_datasets:
        raise ValueError(
            f"{fold.fold}: held-out dataset appears "
            "in source_dev"
        )

    if fold.target_primary.dataset != fold.held_out_dataset:
        raise ValueError(
            f"{fold.fold}: primary target is not the "
            "held-out dataset"
        )

    if (
        fold.target_secondary is not None
        and fold.target_secondary.dataset
        != fold.held_out_dataset
    ):
        raise ValueError(
            f"{fold.fold}: secondary target is not the "
            "held-out dataset"
        )

    for ref in fold.source_train:
        if ref.split != "train":
            raise ValueError(
                f"{fold.fold}: source_train entry "
                f"{ref.dataset!r} uses split {ref.split!r}"
            )

    for ref in fold.source_dev:
        if ref.split != "dev":
            raise ValueError(
                f"{fold.fold}: source_dev entry "
                f"{ref.dataset!r} uses split {ref.split!r}"
            )

    if check_files:
        refs = (
            list(fold.source_train)
            + list(fold.source_dev)
            + [fold.target_primary]
        )

        if fold.target_secondary is not None:
            refs.append(fold.target_secondary)

        for ref in refs:
            if not Path(ref.path).is_file():
                raise FileNotFoundError(
                    f"{fold.fold}: manifest not found: "
                    f"{ref.path}"
                )
