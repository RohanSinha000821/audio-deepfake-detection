from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_audio_batch(
    batch: Sequence[dict],
) -> dict:
    """
    Collate variable-length mono waveforms.

    Returns
    -------
    waveform:
        Float tensor [batch, time], zero padded.

    attention_mask:
        Boolean tensor [batch, time].
        True marks real audio, False marks padding.

    lengths:
        Original waveform lengths before padding.

    label:
        Long tensor [batch].

    sample_rate:
        Shared integer sample rate for the batch.

    Other metadata fields are returned as Python lists.
    """

    if not batch:
        raise ValueError("Cannot collate an empty batch")

    required = {
        "waveform",
        "sample_rate",
        "label",
        "path",
        "dataset",
        "split",
        "utterance_id",
    }

    for index, item in enumerate(batch):
        missing = required - set(item)

        if missing:
            raise ValueError(
                f"Batch item {index} is missing fields: "
                f"{sorted(missing)}"
            )

        waveform = item["waveform"]

        if not isinstance(waveform, torch.Tensor):
            raise TypeError(
                f"Batch item {index}: waveform must be a tensor"
            )

        if waveform.ndim != 1:
            raise ValueError(
                f"Batch item {index}: expected 1-D waveform, "
                f"got shape {tuple(waveform.shape)}"
            )

    sample_rates = {
        int(item["sample_rate"])
        for item in batch
    }

    if len(sample_rates) != 1:
        raise ValueError(
            "All examples in one batch must have the same "
            f"sample rate, got {sorted(sample_rates)}"
        )

    waveforms = [
        item["waveform"].to(dtype=torch.float32)
        for item in batch
    ]

    lengths = torch.tensor(
        [waveform.numel() for waveform in waveforms],
        dtype=torch.long,
    )

    padded = pad_sequence(
        waveforms,
        batch_first=True,
        padding_value=0.0,
    )

    positions = torch.arange(
        padded.shape[1],
        dtype=torch.long,
    ).unsqueeze(0)

    attention_mask = positions < lengths.unsqueeze(1)

    labels = torch.tensor(
        [int(item["label"]) for item in batch],
        dtype=torch.long,
    )

    return {
        "waveform": padded,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "label": labels,
        "sample_rate": sample_rates.pop(),
        "path": [item["path"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
        "split": [item["split"] for item in batch],
        "utterance_id": [
            item["utterance_id"]
            for item in batch
        ],
    }
