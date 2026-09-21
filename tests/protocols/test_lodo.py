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
