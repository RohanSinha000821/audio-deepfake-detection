from __future__ import annotations

import csv
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
    check_manifest_contents: bool = False,
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
        check_manifest_contents=check_manifest_contents,
    )

    return fold


def validate_lodo_fold(
    fold: LODOFold,
    *,
    check_files: bool = False,
    check_manifest_contents: bool = False,
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

    train_paths = [
        Path(ref.path).resolve()
        for ref in fold.source_train
    ]
    dev_paths = [
        Path(ref.path).resolve()
        for ref in fold.source_dev
    ]

    if len(train_paths) != len(set(train_paths)):
        raise ValueError(
            f"{fold.fold}: duplicate source_train manifest path"
        )

    if len(dev_paths) != len(set(dev_paths)):
        raise ValueError(
            f"{fold.fold}: duplicate source_dev manifest path"
        )

    if set(train_datasets) != set(dev_datasets):
        raise ValueError(
            f"{fold.fold}: source train/dev datasets differ"
        )

    train_by_dataset = {
        ref.dataset: Path(ref.path).resolve()
        for ref in fold.source_train
    }
    dev_by_dataset = {
        ref.dataset: Path(ref.path).resolve()
        for ref in fold.source_dev
    }

    for dataset_name in train_by_dataset:
        if (
            train_by_dataset[dataset_name]
            == dev_by_dataset[dataset_name]
        ):
            raise ValueError(
                f"{fold.fold}: source train/dev for "
                f"dataset {dataset_name!r} use the same "
                "manifest file"
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

    source_paths = set(train_paths + dev_paths)
    target_refs = [fold.target_primary]

    if fold.target_secondary is not None:
        target_refs.append(fold.target_secondary)

    for ref in target_refs:
        if Path(ref.path).resolve() in source_paths:
            raise ValueError(
                f"{fold.fold}: target manifest {ref.path!r} "
                "is also used as a source manifest"
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

    refs = (
        list(fold.source_train)
        + list(fold.source_dev)
        + target_refs
    )

    if check_files:
        for ref in refs:
            if not Path(ref.path).is_file():
                raise FileNotFoundError(
                    f"{fold.fold}: manifest not found: "
                    f"{ref.path}"
                )

    if check_manifest_contents:
        for ref in refs:
            _validate_manifest_contents(ref)


def _validate_manifest_contents(
    ref: ManifestRef,
) -> None:
    path = Path(ref.path)

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")

        missing = {"dataset", "split"} - set(
            reader.fieldnames
        )

        if missing:
            raise ValueError(
                f"{path}: missing required columns: "
                f"{sorted(missing)}"
            )

        for line_number, row in enumerate(reader, start=2):
            actual_dataset = row["dataset"]
            actual_split = row["split"]

            if (
                actual_dataset != ref.dataset
                or actual_split != ref.split
            ):
                raise ValueError(
                    f"{path}:{line_number}: expected "
                    f"dataset={ref.dataset!r}, "
                    f"split={ref.split!r}; got "
                    f"dataset={actual_dataset!r}, "
                    f"split={actual_split!r}"
                )
