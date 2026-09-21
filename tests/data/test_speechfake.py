import csv
from pathlib import Path

import pytest

from audio_deepfake_detection.data.speechfake import (
    build_split_manifest,
    iter_unique_split,
)


FIELDS = [
    "file",
    "label",
    "generator",
    "model",
    "speaker",
    "language",
]


def write_metadata(
    root: Path,
    split: str,
    rows: list[dict[str, str]],
) -> None:
    base = (
        root
        / "metadata"
        / "experiments"
        / "baseline"
    )
    base.mkdir(parents=True, exist_ok=True)

    filename = {
        "train": "train_all.csv",
        "dev": "dev_all.csv",
        "test": "test_all.csv",
    }[split]

    path = base / filename

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


def test_speechfake_parser(tmp_path):
    root = tmp_path / "extracted"

    write_metadata(
        root,
        "train",
        [
            {
                "file": "Real/LibriTTS/a.wav",
                "label": "bonafide",
                "generator": "-",
                "model": "-",
                "speaker": "spk1",
                "language": "en",
            },
            {
                "file": "BD/CosyVoice/b.wav",
                "label": "spoof",
                "generator": "TTS",
                "model": "CosyVoice",
                "speaker": "spk2",
                "language": "en",
            },
        ],
    )

    rows = list(iter_unique_split(root, "train"))

    assert len(rows) == 2

    real = rows[0]
    fake = rows[1]

    assert real.label == 0
    assert real.source_id == "LibriTTS"
    assert real.condition == "Real"
    assert real.speaker_id == "spk1"

    assert fake.label == 1
    assert fake.attack_id == "CosyVoice"
    assert fake.attack_family == "tts"
    assert fake.condition == "BD"


def test_identical_duplicate_is_deduplicated(tmp_path):
    root = tmp_path / "extracted"

    row = {
        "file": "BD/CosyVoice/b.wav",
        "label": "spoof",
        "generator": "TTS",
        "model": "CosyVoice",
        "speaker": "spk2",
        "language": "en",
    }

    write_metadata(
        root,
        "train",
        [row, row],
    )

    rows = list(iter_unique_split(root, "train"))

    assert len(rows) == 1


def test_conflicting_duplicate_is_rejected(tmp_path):
    root = tmp_path / "extracted"

    write_metadata(
        root,
        "train",
        [
            {
                "file": "BD/example.wav",
                "label": "spoof",
                "generator": "TTS",
                "model": "ModelA",
                "speaker": "spk1",
                "language": "en",
            },
            {
                "file": "BD/example.wav",
                "label": "bonafide",
                "generator": "-",
                "model": "-",
                "speaker": "spk1",
                "language": "en",
            },
        ],
    )

    with pytest.raises(ValueError):
        list(iter_unique_split(root, "train"))


def test_train_excludes_dev_overlap(tmp_path):
    root = tmp_path / "extracted"

    write_metadata(
        root,
        "train",
        [
            {
                "file": "BD/a.wav",
                "label": "spoof",
                "generator": "TTS",
                "model": "A",
                "speaker": "s1",
                "language": "en",
            },
            {
                "file": "BD/b.wav",
                "label": "spoof",
                "generator": "TTS",
                "model": "B",
                "speaker": "s2",
                "language": "en",
            },
        ],
    )

    write_metadata(
        root,
        "dev",
        [
            {
                "file": "BD/b.wav",
                "label": "spoof",
                "generator": "TTS",
                "model": "B",
                "speaker": "s2",
                "language": "en",
            },
        ],
    )

    output = tmp_path / "train.csv"

    count = build_split_manifest(
        root,
        "train",
        output,
    )

    assert count == 1
