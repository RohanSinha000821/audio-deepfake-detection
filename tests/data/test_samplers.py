import csv
from collections import Counter

from audio_deepfake_detection.data.datasets import (
    ManifestDataset,
)
from audio_deepfake_detection.data.samplers import (
    DatasetClassBalancedSampler,
)


FIELDS = [
    "path",
    "label",
    "dataset",
    "split",
    "utterance_id",
]


def test_dataset_class_balanced_sampler(tmp_path):
    manifest = tmp_path / "manifest.csv"

    rows = []

    # Deliberately very imbalanced raw dataset sizes.
    for dataset_name, count in [
        ("small", 10),
        ("large", 100),
    ]:
        for label in (0, 1):
            for i in range(count):
                rows.append(
                    {
                        "path": f"/tmp/{dataset_name}_{label}_{i}.wav",
                        "label": str(label),
                        "dataset": dataset_name,
                        "split": "train",
                        "utterance_id": (
                            f"{dataset_name}_{label}_{i}"
                        ),
                    }
                )

    with manifest.open(
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

    dataset = ManifestDataset(manifest)

    sampler = DatasetClassBalancedSampler(
        dataset,
        num_samples=8000,
        seed=1234,
    )

    counts = Counter()

    for index, _crop_seed in sampler:
        record = dataset.records[index]

        counts[
            (record.dataset, record.label)
        ] += 1

    # Four dataset/class buckets should each receive
    # approximately 25% of samples.
    for key in [
        ("small", 0),
        ("small", 1),
        ("large", 0),
        ("large", 1),
    ]:
        assert 1700 <= counts[key] <= 2300


def test_sampler_is_reproducible(tmp_path):
    manifest = tmp_path / "manifest.csv"

    rows = []

    for dataset_name in ("a", "b"):
        for label in (0, 1):
            for i in range(3):
                rows.append(
                    {
                        "path": f"/tmp/{i}.wav",
                        "label": str(label),
                        "dataset": dataset_name,
                        "split": "train",
                        "utterance_id": (
                            f"{dataset_name}_{label}_{i}"
                        ),
                    }
                )

    with manifest.open(
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

    dataset = ManifestDataset(manifest)

    sampler1 = DatasetClassBalancedSampler(
        dataset,
        num_samples=100,
        seed=42,
    )

    sampler2 = DatasetClassBalancedSampler(
        dataset,
        num_samples=100,
        seed=42,
    )

    assert list(sampler1) == list(sampler2)

    sampler2.set_epoch(1)

    assert list(sampler1) != list(sampler2)


def test_sampler_crop_seeds_are_epoch_reproducible(tmp_path):
    manifest = tmp_path / "manifest.csv"
    rows = []

    for label in (0, 1):
        for i in range(3):
            rows.append(
                {
                    "path": f"/tmp/{label}_{i}.wav",
                    "label": str(label),
                    "dataset": "source",
                    "split": "train",
                    "utterance_id": f"{label}_{i}",
                }
            )

    with manifest.open(
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

    dataset = ManifestDataset(manifest)
    sampler1 = DatasetClassBalancedSampler(
        dataset,
        num_samples=50,
        seed=91,
    )
    sampler2 = DatasetClassBalancedSampler(
        dataset,
        num_samples=50,
        seed=91,
    )

    sampler1.set_epoch(7)
    sampler2.set_epoch(7)
    epoch7_first = list(sampler1)
    epoch7_second = list(sampler2)

    assert epoch7_first == epoch7_second
    assert all(
        isinstance(index, int) and isinstance(seed, int)
        for index, seed in epoch7_first
    )

    sampler2.set_epoch(8)
    assert epoch7_first != list(sampler2)
