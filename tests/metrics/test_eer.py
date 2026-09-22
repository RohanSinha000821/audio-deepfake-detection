import pytest

from audio_deepfake_detection.metrics.eer import (
    compute_eer,
)


def test_perfect_eer_is_zero():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]

    eer, threshold = compute_eer(
        labels,
        scores,
    )

    assert eer == pytest.approx(0.0)
    assert 0.2 < threshold <= 0.8


def test_half_eer_case():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.6, 0.4, 0.9]

    eer, _ = compute_eer(
        labels,
        scores,
    )

    assert eer == pytest.approx(0.5)
