from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def validate_binary_inputs(
    labels,
    scores,
) -> tuple[np.ndarray, np.ndarray]:
    raw_labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)

    if raw_labels.ndim != 1 or scores.ndim != 1:
        raise ValueError(
            "labels and scores must be 1-D arrays"
        )

    try:
        numeric_labels = np.asarray(
            labels,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "labels must contain only numeric 0/1 values"
        ) from exc

    if len(numeric_labels) != len(scores):
        raise ValueError(
            "labels and scores must have the same length"
        )

    if len(numeric_labels) == 0:
        raise ValueError("inputs must not be empty")

    if not np.all(np.isfinite(scores)):
        raise ValueError(
            "scores must contain only finite values"
        )

    if not np.all(np.isfinite(numeric_labels)):
        raise ValueError(
            "labels must contain only finite values"
        )

    valid_labels = (
        (numeric_labels == 0.0)
        | (numeric_labels == 1.0)
    )

    if not np.all(valid_labels):
        invalid = np.unique(
            numeric_labels[~valid_labels]
        ).tolist()
        raise ValueError(
            f"labels must be exactly 0 or 1, got {invalid}"
        )

    labels = numeric_labels.astype(np.int64)
    unique = set(np.unique(labels).tolist())

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
