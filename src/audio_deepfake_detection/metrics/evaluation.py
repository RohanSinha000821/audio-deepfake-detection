from __future__ import annotations

from collections.abc import Mapping

from .binary_metrics import compute_auroc
from .eer import compute_eer
from .thresholds import (
    APCERThreshold,
    evaluate_transferred_threshold,
    select_standard_apcer_thresholds,
)


def select_source_operating_points(
    labels,
    scores,
) -> dict[float, APCERThreshold]:
    """
    Select project-standard operating thresholds from source dev.

    These thresholds may be selected ONLY from source validation
    data and must then be transferred unchanged to the held-out
    target.
    """

    return select_standard_apcer_thresholds(
        labels,
        scores,
    )


def evaluate_scores(
    labels,
    scores,
    *,
    transferred_thresholds: (
        Mapping[float, APCERThreshold] | None
    ) = None,
) -> dict:
    """
    Evaluate one score vector under the global project convention:

        label 0 = bonafide
        label 1 = spoof
        higher score = more spoof-like

    EER and AUROC are threshold-independent summary metrics.

    The returned EER threshold is diagnostic only. It must never
    be used as a deployment threshold for a held-out target.

    If source-dev APCER operating points are supplied, they are
    applied unchanged to this score vector.
    """

    eer, eer_threshold = compute_eer(
        labels,
        scores,
    )

    result = {
        "eer": eer,
        "auroc": compute_auroc(
            labels,
            scores,
        ),
        "eer_threshold_diagnostic": eer_threshold,
    }

    if transferred_thresholds is not None:
        transferred = {}

        for target_apcer in sorted(
            transferred_thresholds
        ):
            selection = transferred_thresholds[
                target_apcer
            ]

            transferred[target_apcer] = (
                evaluate_transferred_threshold(
                    labels,
                    scores,
                    selection,
                )
            )

        result["transferred"] = transferred

    return result
