import csv
from collections import Counter

from audio_deepfake_detection.data.datasets import (
    ManifestDataset,
)
from audio_deepfake_detection.data.samplers import (
    DatasetBalancedSampler,
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


def write_rows(path, rows):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_naturally_imbalanced_dataset(tmp_path):
    manifest = tmp_path / "domain_balanced.csv"
    rows = []

    for dataset_name, class_counts in {
        "mostly_bona": {0: 90, 1: 10},
        "mostly_spoof": {0: 100, 1: 900},
    }.items():
        for label, count in class_counts.items():
            for index in range(count):
                rows.append(
                    {
                        "path": (
                            f"/tmp/{dataset_name}_{label}_{index}.wav"
                        ),
                        "label": str(label),
                        "dataset": dataset_name,
                        "split": "train",
                        "utterance_id": (
                            f"{dataset_name}_{label}_{index}"
                        ),
                    }
                )

    write_rows(manifest, rows)

    return ManifestDataset(manifest)


def test_dataset_balanced_sampler_is_domain_uniform(tmp_path):
    dataset = make_naturally_imbalanced_dataset(tmp_path)
    sampler = DatasetBalancedSampler(
        dataset,
        num_samples=20000,
        seed=123,
    )
    counts = Counter(
        dataset.records[index].dataset
        for index, _crop_seed in sampler
    )

    assert 9000 <= counts["mostly_bona"] <= 11000
    assert 9000 <= counts["mostly_spoof"] <= 11000


def test_dataset_balanced_sampler_preserves_class_ratios(tmp_path):
    dataset = make_naturally_imbalanced_dataset(tmp_path)
    sampler = DatasetBalancedSampler(
        dataset,
        num_samples=20000,
        seed=456,
    )
    counts = Counter()

    for index, _crop_seed in sampler:
        record = dataset.records[index]
        counts[(record.dataset, record.label)] += 1

    mostly_bona_spoof_fraction = (
        counts[("mostly_bona", 1)]
        / (
            counts[("mostly_bona", 0)]
            + counts[("mostly_bona", 1)]
        )
    )
    mostly_spoof_fraction = (
        counts[("mostly_spoof", 1)]
        / (
            counts[("mostly_spoof", 0)]
            + counts[("mostly_spoof", 1)]
        )
    )

    assert 0.07 <= mostly_bona_spoof_fraction <= 0.13
    assert 0.87 <= mostly_spoof_fraction <= 0.93


def test_dataset_balanced_sampler_epoch_reproducibility(tmp_path):
    dataset = make_naturally_imbalanced_dataset(tmp_path)
    sampler1 = DatasetBalancedSampler(
        dataset,
        num_samples=100,
        seed=789,
    )
    sampler2 = DatasetBalancedSampler(
        dataset,
        num_samples=100,
        seed=789,
    )

    sampler1.set_epoch(4)
    sampler2.set_epoch(4)
    epoch4 = list(sampler1)

    assert epoch4 == list(sampler2)

    sampler2.set_epoch(5)
    assert epoch4 != list(sampler2)
