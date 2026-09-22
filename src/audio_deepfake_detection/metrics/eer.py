from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve

from .binary_metrics import validate_binary_inputs


def compute_eer(
    labels,
    scores,
) -> tuple[float, float]:
    """
    Compute Equal Error Rate using linear interpolation
    between neighbouring ROC operating points.

    Returns
    -------
    eer:
        Equal error rate in [0, 1].

    threshold:
        Target-derived EER threshold.

        This threshold is diagnostic only and MUST NOT be
        used as a deployable threshold on held-out targets.
    """

    labels, scores = validate_binary_inputs(
        labels,
        scores,
    )

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
        pos_label=1,
    )

    fnr = 1.0 - tpr
    difference = fpr - fnr

    exact = np.flatnonzero(
        np.isclose(difference, 0.0)
    )

    if len(exact):
        index = int(exact[0])

        return (
            float(fpr[index]),
            float(thresholds[index]),
        )

    crossing = np.flatnonzero(
        difference >= 0.0
    )

    if len(crossing) == 0:
        index = int(
            np.argmin(np.abs(difference))
        )

        return (
            float(
                (fpr[index] + fnr[index])
                / 2.0
            ),
            float(thresholds[index]),
        )

    upper = int(crossing[0])

    if upper == 0:
        return (
            float(
                (fpr[0] + fnr[0])
                / 2.0
            ),
            float(thresholds[0]),
        )

    lower = upper - 1

    d0 = difference[lower]
    d1 = difference[upper]

    weight = float(
        -d0 / (d1 - d0)
    )

    eer = float(
        fpr[lower]
        + weight
        * (fpr[upper] - fpr[lower])
    )

    t0 = thresholds[lower]
    t1 = thresholds[upper]

    if np.isfinite(t0) and np.isfinite(t1):
        threshold = float(
            t0 + weight * (t1 - t0)
        )
    else:
        threshold = float(t1)

    return eer, threshold
