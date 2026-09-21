from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
import torchaudio


def load_audio_segment(
    path: str | Path,
    *,
    max_seconds: float | None = None,
    random_crop: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, int]:
    """
    Load a mono floating-point audio segment.

    If max_seconds is given and the utterance is longer,
    choose one continuous segment before any resampling.

    Training:
        random_crop=True

    Evaluation:
        random_crop=False -> deterministic centre crop

    Returns
    -------
    waveform:
        1-D float32 tensor [time]

    sample_rate:
        Native sample rate of the file.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive")

    with sf.SoundFile(path, mode="r") as audio:
        sample_rate = int(audio.samplerate)
        total_frames = int(len(audio))

        if max_seconds is None:
            start_frame = 0
            frames_to_read = total_frames

        else:
            max_frames = max(
                1,
                int(round(max_seconds * sample_rate)),
            )

            if total_frames <= max_frames:
                start_frame = 0
                frames_to_read = total_frames

            else:
                frames_to_read = max_frames
                available_start = total_frames - max_frames

                if random_crop:
                    start_frame = int(
                        torch.randint(
                            low=0,
                            high=available_start + 1,
                            size=(1,),
                            generator=generator,
                        ).item()
                    )
                else:
                    start_frame = available_start // 2

        audio.seek(start_frame)

        data = audio.read(
            frames=frames_to_read,
            dtype="float32",
            always_2d=True,
        )

    # soundfile -> [time, channels]
    waveform = torch.from_numpy(data)

    # Convert to mono.
    waveform = waveform.mean(dim=1)

    return waveform.contiguous(), sample_rate


def resample_audio(
    waveform: torch.Tensor,
    source_sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    """
    Resample a waveform while preserving the already-selected
    temporal segment.
    """

    if source_sample_rate <= 0:
        raise ValueError(
            "source_sample_rate must be positive"
        )

    if target_sample_rate <= 0:
        raise ValueError(
            "target_sample_rate must be positive"
        )

    if source_sample_rate == target_sample_rate:
        return waveform

    return torchaudio.functional.resample(
        waveform,
        orig_freq=source_sample_rate,
        new_freq=target_sample_rate,
    )
