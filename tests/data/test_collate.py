import pytest
import torch

from audio_deepfake_detection.data.collate import (
    collate_audio_batch,
)


def make_item(
    waveform,
    label,
    utterance_id,
    sample_rate=16000,
):
    return {
        "waveform": torch.tensor(
            waveform,
            dtype=torch.float32,
        ),
        "sample_rate": sample_rate,
        "label": label,
        "path": f"/tmp/{utterance_id}.wav",
        "dataset": "example",
        "split": "train",
        "utterance_id": utterance_id,
    }


def test_collate_variable_length_audio():
    batch = [
        make_item(
            [1.0, 2.0, 3.0],
            0,
            "a",
        ),
        make_item(
            [4.0, 5.0, 6.0, 7.0, 8.0],
            1,
            "b",
        ),
    ]

    result = collate_audio_batch(batch)

    assert result["waveform"].shape == (2, 5)

    assert torch.equal(
        result["lengths"],
        torch.tensor([3, 5]),
    )

    assert torch.equal(
        result["attention_mask"],
        torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, True],
            ]
        ),
    )

    assert torch.equal(
        result["label"],
        torch.tensor([0, 1]),
    )

    assert result["sample_rate"] == 16000


def test_collate_rejects_mixed_sample_rates():
    batch = [
        make_item(
            [1.0],
            0,
            "a",
            sample_rate=16000,
        ),
        make_item(
            [1.0],
            1,
            "b",
            sample_rate=48000,
        ),
    ]

    with pytest.raises(ValueError):
        collate_audio_batch(batch)


def test_collate_rejects_empty_batch():
    with pytest.raises(ValueError):
        collate_audio_batch([])
