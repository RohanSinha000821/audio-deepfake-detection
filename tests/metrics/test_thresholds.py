import numpy as np
import pytest

from audio_deepfake_detection.metrics.thresholds import (
    evaluate_transferred_threshold,
    select_threshold_for_apcer,
)


def test_select_exact_five_percent_apcer():
    spoof = np.arange(
        100,
        dtype=float,
    ) / 100.0

    bona = np.array(
        [-0.3, -0.2, -0.1],
        dtype=float,
    )

    labels = np.concatenate(
        [
            np.zeros(len(bona)),
            np.ones(len(spoof)),
        ]
    )

    scores = np.concatenate(
        [bona, spoof]
    )

    selection = select_threshold_for_apcer(
        labels,
        scores,
        0.05,
    )

    assert selection.threshold == pytest.approx(
        0.05
    )

    assert selection.achieved_apcer == pytest.approx(
        0.05
    )


def test_threshold_is_transferred_unchanged():
    dev_labels = [0, 0, 1, 1]
    dev_scores = [0.0, 0.1, 0.8, 0.9]

    selection = select_threshold_for_apcer(
        dev_labels,
        dev_scores,
        0.10,
    )

    target_labels = [0, 0, 1, 1]
    target_scores = [0.2, 0.7, 0.3, 0.8]

    result = evaluate_transferred_threshold(
        target_labels,
        target_scores,
        selection,
    )

    assert result["threshold"] == (
        selection.threshold
    )


def test_invalid_apcer_target_rejected():
    with pytest.raises(ValueError):
        select_threshold_for_apcer(
            [0, 1],
            [0.1, 0.9],
            1.0,
        )
