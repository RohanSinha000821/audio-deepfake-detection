import csv

import numpy as np
import soundfile as sf

from audio_deepfake_detection.data.datasets import (
    ManifestDataset,
)


FIELDS = [
    "path",
    "label",
    "dataset",
    "split",
    "utterance_id",
]


def write_manifest(path, rows):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_dataset_metadata(tmp_path):
    manifest = tmp_path / "manifest.csv"

    write_manifest(
        manifest,
        [
            {
                "path": "/tmp/a.wav",
                "label": "0",
                "dataset": "a",
                "split": "train",
                "utterance_id": "a1",
            },
            {
                "path": "/tmp/b.wav",
                "label": "1",
                "dataset": "b",
                "split": "train",
                "utterance_id": "b1",
            },
        ],
    )

    dataset = ManifestDataset(manifest)

    assert len(dataset) == 2
    assert dataset[0]["label"] == 0
    assert dataset[1]["dataset"] == "b"


def test_manifest_dataset_loads_audio(tmp_path):
    audio = tmp_path / "audio.wav"
    manifest = tmp_path / "manifest.csv"

    sf.write(
        audio,
        np.zeros(8000, dtype=np.float32),
        8000,
    )

    write_manifest(
        manifest,
        [
            {
                "path": str(audio),
                "label": "1",
                "dataset": "example",
                "split": "train",
                "utterance_id": "utt1",
            },
        ],
    )

    dataset = ManifestDataset(
        manifest,
        load_audio=True,
        target_sample_rate=16000,
    )

    item = dataset[0]

    assert item["sample_rate"] == 16000
    assert item["waveform"].shape[0] == 16000
