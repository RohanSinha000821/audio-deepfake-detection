from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from .schema import ManifestRow, write_manifest


DATASET_NAME = "speechfake"

EXPECTED_COLUMNS = {
    "file",
    "label",
    "generator",
    "model",
    "speaker",
    "language",
}

SPLIT_FILES = {
    "train": "train_all.csv",
    "dev": "dev_all.csv",
    "test": "test_all.csv",
}


def _clean(value: str) -> str:
    value = value.strip()
    return "" if value == "-" else value


def _metadata_path(
    extracted_root: Path,
    split: str,
) -> Path:
    if split not in SPLIT_FILES:
        raise ValueError(
            f"Unknown split {split!r}. "
            f"Expected one of {sorted(SPLIT_FILES)}"
        )

    return (
        extracted_root
        / "metadata"
        / "experiments"
        / "baseline"
        / SPLIT_FILES[split]
    )


def _iter_metadata(
    extracted_root: Path,
    split: str,
) -> Iterator[dict[str, str]]:
    path = _metadata_path(extracted_root, split)

    if not path.is_file():
        raise FileNotFoundError(
            f"SpeechFake metadata not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")

        if set(reader.fieldnames) != EXPECTED_COLUMNS:
            raise ValueError(
                f"{path}: unexpected columns. "
                f"Expected {sorted(EXPECTED_COLUMNS)}, "
                f"got {reader.fieldnames}"
            )

        for line_number, row in enumerate(reader, start=2):
            for column in EXPECTED_COLUMNS:
                if row[column] is None:
                    raise ValueError(
                        f"{path}:{line_number}: "
                        f"missing value for {column!r}"
                    )

            yield row


def _signature(
    row: dict[str, str],
) -> tuple[str, str, str, str, str]:
    return (
        row["label"].strip(),
        row["generator"].strip(),
        row["model"].strip(),
        row["speaker"].strip(),
        row["language"].strip(),
    )


def _to_manifest_row(
    extracted_root: Path,
    split: str,
    row: dict[str, str],
) -> ManifestRow:
    relative_file = Path(row["file"])

    if relative_file.is_absolute():
        raise ValueError(
            f"SpeechFake file path must be relative: "
            f"{relative_file}"
        )

    label_text = row["label"].strip().lower()

    if label_text == "bonafide":
        label = 0
    elif label_text == "spoof":
        label = 1
    else:
        raise ValueError(
            f"Unexpected SpeechFake label: {row['label']!r}"
        )

    generator = _clean(row["generator"])
    model = _clean(row["model"])
    speaker = _clean(row["speaker"])
    language = _clean(row["language"])

    parts = relative_file.parts

    if not parts:
        raise ValueError(
            f"Invalid SpeechFake file path: {row['file']!r}"
        )

    # Top-level SpeechFake domain:
    # Real / BD / MD
    condition = parts[0]

    if label == 0:
        attack_id = ""
        attack_family = ""

        if len(parts) >= 2 and condition == "Real":
            source_id = parts[1]
        else:
            source_id = ""
    else:
        attack_id = model
        attack_family = generator.lower()
        source_id = ""

    return ManifestRow(
        path=str(extracted_root / relative_file),
        label=label,
        dataset=DATASET_NAME,
        split=split,
        utterance_id=relative_file.with_suffix("").as_posix(),
        speaker_id=speaker,
        attack_id=attack_id,
        attack_family=attack_family,
        source_id=source_id,
        language=language,
        condition=condition,
    )


def collect_unique_paths(
    extracted_root: str | Path,
    split: str,
) -> set[str]:
    root = Path(extracted_root)

    return {
        row["file"]
        for row in _iter_metadata(root, split)
    }


def iter_unique_split(
    extracted_root: str | Path,
    split: str,
    *,
    exclude_paths: set[str] | None = None,
) -> Iterator[ManifestRow]:
    """
    Emit at most one row for each SpeechFake audio path.

    Repeated paths must have identical metadata. A repeated path
    with conflicting metadata is treated as a dataset error.
    """

    root = Path(extracted_root)
    excluded = exclude_paths or set()

    seen: dict[
        str,
        tuple[str, str, str, str, str],
    ] = {}

    for row in _iter_metadata(root, split):
        relative_path = row["file"]
        signature = _signature(row)

        if relative_path in seen:
            if seen[relative_path] != signature:
                raise ValueError(
                    "Conflicting metadata for repeated "
                    f"SpeechFake path {relative_path!r}"
                )

            continue

        seen[relative_path] = signature

        if relative_path in excluded:
            continue

        yield _to_manifest_row(
            root,
            split,
            row,
        )


def build_split_manifest(
    extracted_root: str | Path,
    split: str,
    output_path: str | Path,
) -> int:
    """
    Build the project's canonical SpeechFake manifest.

    Source training additionally excludes any file occurring in
    official dev, preventing source train/dev leakage.
    """

    if split == "train":
        dev_paths = collect_unique_paths(
            extracted_root,
            "dev",
        )

        rows = iter_unique_split(
            extracted_root,
            "train",
            exclude_paths=dev_paths,
        )

    elif split in {"dev", "test"}:
        rows = iter_unique_split(
            extracted_root,
            split,
        )

    else:
        raise ValueError(
            f"Unknown split {split!r}. "
            "Expected train, dev, or test."
        )

    return write_manifest(
        rows,
        output_path,
    )
