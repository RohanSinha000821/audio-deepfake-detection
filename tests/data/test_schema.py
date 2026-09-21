import pytest

from audio_deepfake_detection.data.schema import ManifestRow


def test_valid_manifest_row():
    row = ManifestRow(
        path="/tmp/example.wav",
        label=1,
        dataset="example",
        split="train",
        utterance_id="utt001",
    )

    assert row.label == 1
    assert row.dataset == "example"


def test_invalid_label():
    with pytest.raises(ValueError):
        ManifestRow(
            path="/tmp/example.wav",
            label=2,
            dataset="example",
            split="train",
            utterance_id="utt001",
        )


def test_required_field_cannot_be_empty():
    with pytest.raises(ValueError):
        ManifestRow(
            path="",
            label=0,
            dataset="example",
            split="train",
            utterance_id="utt001",
        )
