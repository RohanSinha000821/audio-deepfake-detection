from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def validate_binary_inputs(
    labels,
    scores,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)

    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError(
            "labels and scores must be 1-D arrays"
        )

    if len(labels) != len(scores):
        raise ValueError(
            "labels and scores must have the same length"
        )

    if len(labels) == 0:
        raise ValueError("inputs must not be empty")

    if not np.all(np.isfinite(scores)):
        raise ValueError(
            "scores must contain only finite values"
        )

    try:
        labels = labels.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "labels must be binary integers"
        ) from exc

    unique = set(np.unique(labels).tolist())

    if not unique.issubset({0, 1}):
        raise ValueError(
            f"labels must be 0/1, got {sorted(unique)}"
        )

    if unique != {0, 1}:
        raise ValueError(
            "both bonafide (0) and spoof (1) "
            "samples are required"
        )

    return labels, scores


def compute_auroc(
    labels,
    scores,
) -> float:
    """
    AUROC under the global convention:
        label 1 = spoof
        higher score = more spoof-like
    """

    labels, scores = validate_binary_inputs(
        labels,
        scores,
    )

    return float(
        roc_auc_score(labels, scores)
    )


def compute_apcer_bpcer(
    labels,
    scores,
    threshold: float,
) -> dict[str, float]:
    """
    Compute ISO-style class error rates.

    Decision rule:
        score >= threshold -> spoof
        score <  threshold -> bonafide

    APCER:
        spoof incorrectly accepted as bonafide.

    BPCER:
        bonafide incorrectly rejected as spoof.
    """

    labels, scores = validate_binary_inputs(
        labels,
        scores,
    )

    threshold = float(threshold)

    if not np.isfinite(threshold):
        raise ValueError(
            "threshold must be finite"
        )

    bona = labels == 0
    spoof = labels == 1

    apcer = float(
        np.mean(scores[spoof] < threshold)
    )

    bpcer = float(
        np.mean(scores[bona] >= threshold)
    )

    return {
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": (apcer + bpcer) / 2.0,
    }
