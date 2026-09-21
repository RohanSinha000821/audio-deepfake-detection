import csv

import pytest

from audio_deepfake_detection.data.validation import (
    validate_manifest,
)


FIELDS = [
    "path",
    "label",
    "dataset",
    "split",
    "utterance_id",
]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_validate_manifest(tmp_path):
    path = tmp_path / "manifest.csv"

    write_csv(
        path,
        [
            {
                "path": "/tmp/a.wav",
                "label": "0",
                "dataset": "example",
                "split": "train",
                "utterance_id": "a",
            },
            {
                "path": "/tmp/b.wav",
                "label": "1",
                "dataset": "example",
                "split": "train",
                "utterance_id": "b",
            },
        ],
    )

    stats = validate_manifest(path)

    assert stats["total"] == 2
    assert stats["bonafide"] == 1
    assert stats["spoof"] == 1


def test_duplicate_utterance_id_rejected(tmp_path):
    path = tmp_path / "manifest.csv"

    row = {
        "path": "/tmp/a.wav",
        "label": "0",
        "dataset": "example",
        "split": "train",
        "utterance_id": "a",
    }

    write_csv(path, [row, row])

    with pytest.raises(ValueError):
        validate_manifest(path)
