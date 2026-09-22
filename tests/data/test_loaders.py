import csv

import numpy as np
import soundfile as sf

from audio_deepfake_detection.data.loaders import (
    create_eval_loader,
    create_train_loader,
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


def make_audio(path):
    sf.write(
        path,
        np.zeros(1600, dtype=np.float32),
        16000,
    )


def test_train_loader(tmp_path):
    rows = []

    for dataset_name in ("a", "b"):
        for label in (0, 1):
            audio = (
                tmp_path
                / f"{dataset_name}_{label}.wav"
            )

            make_audio(audio)

            rows.append(
                {
                    "path": str(audio),
                    "label": str(label),
                    "dataset": dataset_name,
                    "split": "train",
                    "utterance_id": (
                        f"{dataset_name}_{label}"
                    ),
                }
            )

    manifest = tmp_path / "train.csv"
    write_manifest(manifest, rows)

    loader, dataset, sampler = create_train_loader(
        [manifest],
        batch_size=2,
        seed=42,
        max_seconds=1.0,
        num_workers=0,
        num_samples=4,
        pin_memory=False,
    )

    batch = next(iter(loader))

    assert len(dataset) == 4
    assert len(sampler) == 4

    assert batch["waveform"].shape[0] == 2
    assert batch["label"].shape[0] == 2
    assert batch["sample_rate"] == 16000


def test_eval_loader_is_sequential(tmp_path):
    rows = []

    for i in range(3):
        audio = tmp_path / f"{i}.wav"
        make_audio(audio)

        rows.append(
            {
                "path": str(audio),
                "label": str(i % 2),
                "dataset": "example",
                "split": "dev",
                "utterance_id": str(i),
            }
        )

    manifest = tmp_path / "dev.csv"
    write_manifest(manifest, rows)

    loader, dataset = create_eval_loader(
        manifest,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )

    batches = list(loader)

    assert len(dataset) == 3

    ids = (
        batches[0]["utterance_id"]
        + batches[1]["utterance_id"]
    )

    assert ids == ["0", "1", "2"]
