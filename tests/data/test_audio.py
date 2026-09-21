import numpy as np
import soundfile as sf
import torch

from audio_deepfake_detection.data.audio import (
    load_audio_segment,
    resample_audio,
)


def test_load_audio_segment(tmp_path):
    path = tmp_path / "test.wav"

    sample_rate = 8000

    # 1 second, stereo
    signal = np.stack(
        [
            np.ones(sample_rate),
            np.zeros(sample_rate),
        ],
        axis=1,
    ).astype(np.float32)

    sf.write(
        path,
        signal,
        sample_rate,
    )

    waveform, sr = load_audio_segment(
        path,
        max_seconds=0.25,
        random_crop=False,
    )

    assert sr == 8000
    assert waveform.ndim == 1
    assert waveform.shape[0] == 2000

    # Stereo average: (1 + 0) / 2
    assert torch.allclose(
        waveform,
        torch.full_like(waveform, 0.5),
        atol=1e-4,
    )


def test_resample_audio():
    waveform = torch.zeros(8000)

    output = resample_audio(
        waveform,
        source_sample_rate=8000,
        target_sample_rate=16000,
    )

    assert output.shape[0] == 16000
