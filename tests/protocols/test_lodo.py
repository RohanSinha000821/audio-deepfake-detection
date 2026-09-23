import csv
from pathlib import Path

import pytest

from audio_deepfake_detection.protocols.lodo import (
    LODOFold,
    ManifestRef,
    load_lodo_config,
    validate_lodo_fold,
)


CONFIG_ROOT = Path("configs/lodo")


EXPECTED = {
    "F1": {
        "held_out": "speechfake",
        "sources": {
            "asvspoof2019",
            "asvspoof5",
            "cfad",
        },
        "target": ("speechfake", "test"),
    },
    "F2": {
        "held_out": "cfad",
        "sources": {
            "asvspoof2019",
            "asvspoof5",
            "speechfake",
        },
        "target": ("cfad", "test_unseen"),
    },
    "F3": {
        "held_out": "asvspoof5",
        "sources": {
            "asvspoof2019",
            "cfad",
            "speechfake",
        },
        "target": ("asvspoof5", "eval"),
    },
    "F4": {
        "held_out": "asvspoof2019",
        "sources": {
            "asvspoof5",
            "cfad",
            "speechfake",
        },
        "target": ("asvspoof2019", "eval"),
    },
}


@pytest.mark.parametrize(
    "filename",
    [
        "f1.yaml",
        "f2.yaml",
        "f3.yaml",
        "f4.yaml",
    ],
)
def test_lodo_configs(filename):
    fold = load_lodo_config(
        CONFIG_ROOT / filename
    )

    expected = EXPECTED[fold.fold]

    assert fold.held_out_dataset == expected["held_out"]

    assert {
        ref.dataset
        for ref in fold.source_train
    } == expected["sources"]

    assert {
        ref.dataset
        for ref in fold.source_dev
    } == expected["sources"]

    assert (
        fold.target_primary.dataset,
        fold.target_primary.split,
    ) == expected["target"]


def test_f2_has_seen_secondary_target():
    fold = load_lodo_config(
        CONFIG_ROOT / "f2.yaml"
    )

    assert fold.target_secondary is not None
    assert fold.target_secondary.dataset == "cfad"
    assert fold.target_secondary.split == "test_seen"


def test_f3_never_uses_asvspoof5_dev():
    fold = load_lodo_config(
        CONFIG_ROOT / "f3.yaml"
    )

    assert "asvspoof5" not in {
        ref.dataset
        for ref in fold.source_dev
    }


def test_held_out_dataset_rejected_as_source():
    fold = LODOFold(
        fold="BAD",
        held_out_dataset="speechfake",
        source_train=(
            ManifestRef(
                "speechfake",
                "train",
                "x.csv",
            ),
        ),
        source_dev=(
            ManifestRef(
                "speechfake",
                "dev",
                "y.csv",
            ),
        ),
        target_primary=ManifestRef(
            "speechfake",
            "test",
            "z.csv",
        ),
    )

    with pytest.raises(ValueError):
        validate_lodo_fold(fold)


def write_manifest(path, dataset, split):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "split"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": dataset,
                "split": split,
            }
        )


def make_content_fold(tmp_path):
    train = tmp_path / "train.csv"
    dev = tmp_path / "dev.csv"
    target = tmp_path / "target.csv"

    write_manifest(train, "source", "train")
    write_manifest(dev, "source", "dev")
    write_manifest(target, "target", "test")

    fold = LODOFold(
        fold="TEST",
        held_out_dataset="target",
        source_train=(
            ManifestRef("source", "train", str(train)),
        ),
        source_dev=(
            ManifestRef("source", "dev", str(dev)),
        ),
        target_primary=ManifestRef(
            "target",
            "test",
            str(target),
        ),
    )

    return fold, train, dev


def test_manifest_content_dataset_mismatch_rejected(tmp_path):
    fold, train, _dev = make_content_fold(tmp_path)
    write_manifest(train, "wrong", "train")

    with pytest.raises(
        ValueError,
        match=(
            r"train\.csv:2: expected dataset='source', "
            r"split='train'; got dataset='wrong', "
            r"split='train'"
        ),
    ):
        validate_lodo_fold(
            fold,
            check_manifest_contents=True,
        )


def test_manifest_content_split_mismatch_rejected(tmp_path):
    fold, _train, dev = make_content_fold(tmp_path)
    write_manifest(dev, "source", "train")

    with pytest.raises(
        ValueError,
        match=(
            r"dev\.csv:2: expected dataset='source', "
            r"split='dev'; got dataset='source', "
            r"split='train'"
        ),
    ):
        validate_lodo_fold(
            fold,
            check_manifest_contents=True,
        )


def test_source_train_dev_same_file_rejected(tmp_path):
    shared = tmp_path / "shared.csv"

    fold = LODOFold(
        fold="TEST",
        held_out_dataset="target",
        source_train=(
            ManifestRef("source", "train", str(shared)),
        ),
        source_dev=(
            ManifestRef("source", "dev", str(shared)),
        ),
        target_primary=ManifestRef(
            "target",
            "test",
            str(tmp_path / "target.csv"),
        ),
    )

    with pytest.raises(ValueError, match="same manifest file"):
        validate_lodo_fold(fold)


def test_target_source_same_file_rejected(tmp_path):
    shared = tmp_path / "shared.csv"

    fold = LODOFold(
        fold="TEST",
        held_out_dataset="target",
        source_train=(
            ManifestRef("source", "train", str(shared)),
        ),
        source_dev=(
            ManifestRef(
                "source",
                "dev",
                str(tmp_path / "dev.csv"),
            ),
        ),
        target_primary=ManifestRef(
            "target",
            "test",
            str(shared),
        ),
    )

    with pytest.raises(ValueError, match="also used as a source"):
        validate_lodo_fold(fold)
