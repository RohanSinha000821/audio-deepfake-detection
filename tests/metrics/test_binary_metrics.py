import numpy as np
import pytest

from audio_deepfake_detection.metrics.binary_metrics import (
    compute_apcer_bpcer,
    compute_auroc,
)


def test_perfect_separation():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]

    assert compute_auroc(
        labels,
        scores,
    ) == pytest.approx(1.0)

    metrics = compute_apcer_bpcer(
        labels,
        scores,
        threshold=0.5,
    )

    assert metrics["apcer"] == 0.0
    assert metrics["bpcer"] == 0.0
    assert metrics["acer"] == 0.0


def test_score_direction_is_spoof_high():
    labels = [0, 0, 1, 1]
    scores = [0.9, 0.8, 0.2, 0.1]

    assert compute_auroc(
        labels,
        scores,
    ) == pytest.approx(0.0)


def test_integral_float_labels_are_valid():
    assert compute_auroc(
        [0.0, 1.0],
        [0.1, 0.9],
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "labels",
    [
        [0.9, 1.0],
        [0.0, 1.2],
        [-1, 1],
        [0, 2],
        [0, np.nan],
        [0, np.inf],
        ["not-a-label", 1],
    ],
)
def test_invalid_labels_are_rejected(labels):
    with pytest.raises(ValueError):
        compute_auroc(labels, [0.1, 0.9])
