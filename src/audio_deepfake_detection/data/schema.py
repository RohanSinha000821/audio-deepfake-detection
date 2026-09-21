from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ManifestRow:
    """
    Canonical representation of one audio example.

    Global label convention:
        0 = bonafide
        1 = spoof
    """

    path: str
    label: int
    dataset: str
    split: str
    utterance_id: str

    speaker_id: str = ""
    gender: str = ""
    attack_id: str = ""
    attack_family: str = ""
    attack_config: str = ""
    source_id: str = ""
    language: str = ""
    condition: str = ""
    codec: str = ""
    codec_quality: str = ""
    codec_seed: str = ""

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError(
                f"label must be 0 (bonafide) or 1 (spoof), got {self.label}"
            )

        required = {
            "path": self.path,
            "dataset": self.dataset,
            "split": self.split,
            "utterance_id": self.utterance_id,
        }

        for name, value in required.items():
            if not str(value).strip():
                raise ValueError(f"{name} must not be empty")


MANIFEST_COLUMNS = [field.name for field in fields(ManifestRow)]


def write_manifest(
    rows: Iterable[ManifestRow],
    output_path: str | Path,
) -> int:
    """
    Stream rows to CSV without holding the entire dataset in memory.

    Returns
    -------
    int
        Number of rows written.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_COLUMNS,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))
            count += 1

    return count
