from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from .audio import load_audio_segment, resample_audio


REQUIRED_COLUMNS = {
    "path",
    "label",
    "dataset",
    "split",
    "utterance_id",
}


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """
    Lightweight training representation.

    We intentionally retain only fields needed by the common
    training layer. Rich attack/source metadata remains available
    in the canonical CSV manifests for later analysis.
    """

    path: str
    label: int
    dataset: str
    split: str
    utterance_id: str


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest_paths: (
            str
            | Path
            | Sequence[str | Path]
        ),
        *,
        load_audio: bool = False,
        max_seconds: float | None = None,
        random_crop: bool = False,
        target_sample_rate: int | None = None,
    ) -> None:

        if isinstance(manifest_paths, (str, Path)):
            paths = [Path(manifest_paths)]
        else:
            paths = [
                Path(path)
                for path in manifest_paths
            ]

        if not paths:
            raise ValueError(
                "At least one manifest path is required"
            )

        self.records: list[ManifestRecord] = []

        for path in paths:
            self._read_manifest(path)

        if not self.records:
            raise ValueError(
                "ManifestDataset contains no rows"
            )

        self.load_audio = load_audio
        self.max_seconds = max_seconds
        self.random_crop = random_crop
        self.target_sample_rate = target_sample_rate

    def _read_manifest(
        self,
        path: Path,
    ) -> None:

        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:

            reader = csv.DictReader(handle)

            if reader.fieldnames is None:
                raise ValueError(
                    f"{path}: missing CSV header"
                )

            missing = (
                REQUIRED_COLUMNS
                - set(reader.fieldnames)
            )

            if missing:
                raise ValueError(
                    f"{path}: missing required columns "
                    f"{sorted(missing)}"
                )

            for line_number, row in enumerate(
                reader,
                start=2,
            ):
                label_text = row["label"]

                try:
                    label = int(label_text)
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: "
                        f"invalid label {label_text!r}"
                    ) from exc

                if label not in (0, 1):
                    raise ValueError(
                        f"{path}:{line_number}: "
                        f"label must be 0 or 1"
                    )

                self.records.append(
                    ManifestRecord(
                        path=row["path"],
                        label=label,
                        dataset=row["dataset"],
                        split=row["split"],
                        utterance_id=row["utterance_id"],
                    )
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: int | tuple[int, int],
    ) -> dict:

        crop_seed: int | None = None

        if isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError(
                    "Sampler index tuple must contain "
                    "(record_index, crop_seed)"
                )

            index, crop_seed = index
            index = int(index)
            crop_seed = int(crop_seed)

        record = self.records[index]

        item = {
            "path": record.path,
            "label": record.label,
            "dataset": record.dataset,
            "split": record.split,
            "utterance_id": record.utterance_id,
        }

        if not self.load_audio:
            return item

        generator = None

        if crop_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(crop_seed)

        waveform, sample_rate = load_audio_segment(
            record.path,
            max_seconds=self.max_seconds,
            random_crop=self.random_crop,
            generator=generator,
        )

        if self.target_sample_rate is not None:
            waveform = resample_audio(
                waveform,
                source_sample_rate=sample_rate,
                target_sample_rate=self.target_sample_rate,
            )

            sample_rate = self.target_sample_rate

        item["waveform"] = waveform
        item["sample_rate"] = sample_rate

        return item
