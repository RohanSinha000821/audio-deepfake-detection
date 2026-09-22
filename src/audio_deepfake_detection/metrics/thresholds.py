from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .binary_metrics import (
    compute_apcer_bpcer,
    validate_binary_inputs,
)


@dataclass(frozen=True)
class APCERThreshold:
    target_apcer: float
    threshold: float

    achieved_apcer: float
    bpcer: float
    acer: float

    num_bonafide: int
    num_spoof: int


def select_threshold_for_apcer(
    labels,
    scores,
    target_apcer: float,
) -> APCERThreshold:
    """
    Select the highest threshold whose source-dev APCER
    does not exceed the requested operating point.

    Because higher scores indicate spoof:

        APCER(t) = P(score < t | spoof)

    Increasing the threshold increases APCER but decreases
    BPCER. Therefore, among thresholds satisfying the APCER
    constraint, the highest threshold gives the lowest
    attainable BPCER.

    Ties are handled exactly according to the decision rule
    score >= threshold -> spoof.
    """

    labels, scores = validate_binary_inputs(
        labels,
        scores,
    )

    target_apcer = float(target_apcer)

    if not 0.0 <= target_apcer < 1.0:
        raise ValueError(
            "target_apcer must satisfy "
            "0 <= target_apcer < 1"
        )

    spoof_scores = np.sort(
        scores[labels == 1]
    )

    unique_scores, first_indices = np.unique(
        spoof_scores,
        return_index=True,
    )

    # At threshold t, spoof samples strictly below t
    # contribute to APCER.
    achieved = (
        first_indices.astype(np.float64)
        / len(spoof_scores)
    )

    valid = achieved <= (
        target_apcer + 1e-15
    )

    if not np.any(valid):
        raise RuntimeError(
            "No valid APCER threshold found"
        )

    index = int(
        np.flatnonzero(valid)[-1]
    )

    threshold = float(
        unique_scores[index]
    )

    metrics = compute_apcer_bpcer(
        labels,
        scores,
        threshold,
    )

    return APCERThreshold(
        target_apcer=target_apcer,
        threshold=threshold,
        achieved_apcer=metrics["apcer"],
        bpcer=metrics["bpcer"],
        acer=metrics["acer"],
        num_bonafide=int(
            np.sum(labels == 0)
        ),
        num_spoof=int(
            np.sum(labels == 1)
        ),
    )


def select_standard_apcer_thresholds(
    labels,
    scores,
) -> dict[float, APCERThreshold]:
    """
    Project-standard source-dev operating points.
    """

    return {
        target: select_threshold_for_apcer(
            labels,
            scores,
            target,
        )
        for target in (
            0.01,
            0.05,
            0.10,
        )
    }


def evaluate_transferred_threshold(
    labels,
    scores,
    selection: APCERThreshold,
) -> dict[str, float]:
    """
    Apply a previously selected source-dev threshold
    unchanged to another partition, typically the held-out
    target.
    """

    metrics = compute_apcer_bpcer(
        labels,
        scores,
        selection.threshold,
    )

    return {
        "threshold": selection.threshold,
        "source_target_apcer":
            selection.target_apcer,
        "apcer": metrics["apcer"],
        "bpcer": metrics["bpcer"],
        "acer": metrics["acer"],
    }
