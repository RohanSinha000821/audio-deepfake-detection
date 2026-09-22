import pytest

from audio_deepfake_detection.metrics.evaluation import (
    evaluate_scores,
    select_source_operating_points,
)


def test_evaluate_scores_basic():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]

    result = evaluate_scores(
        labels,
        scores,
    )

    assert result["eer"] == pytest.approx(0.0)
    assert result["auroc"] == pytest.approx(1.0)
    assert "eer_threshold_diagnostic" in result
    assert "transferred" not in result


def test_source_thresholds_transfer_unchanged():
    dev_labels = (
        [0] * 20
        + [1] * 100
    )

    dev_scores = (
        [-0.5] * 20
        + [i / 100 for i in range(100)]
    )

    operating_points = (
        select_source_operating_points(
            dev_labels,
            dev_scores,
        )
    )

    target_labels = [
        0, 0, 0,
        1, 1, 1,
    ]

    target_scores = [
        0.2, 0.6, 0.8,
        0.3, 0.7, 0.9,
    ]

    result = evaluate_scores(
        target_labels,
        target_scores,
        transferred_thresholds=operating_points,
    )

    assert set(result["transferred"]) == {
        0.01,
        0.05,
        0.10,
    }

    for apcer_target, selection in (
        operating_points.items()
    ):
        assert (
            result["transferred"]
            [apcer_target]
            ["threshold"]
            == selection.threshold
        )
