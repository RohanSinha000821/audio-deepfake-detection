from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from .schema import ManifestRow, write_manifest


DATASET_NAME = "cfad"


SPLITS = {
    "train": "train_clean",
    "dev": "dev_clean",
    "test_seen": "test_seen_clean",
    "test_unseen": "test_unseen_clean",
}


REAL_SOURCES = {
    "aishell1": ("R01", "AISHELL1"),
    "aishell3": ("R02", "AISHELL3"),
    "thchs30": ("R03", "THCHS-30"),
    "magicread": ("R04", "MAGICDATA-read"),
    "magicconversa": ("R05", "MAGICDATA-conversation"),
    "selfrecording": ("R06", "self-recording"),
}


FAKE_TYPES = {
    "straight": ("F01", "traditional_vocoder"),
    "gl": ("F02", "traditional_vocoder"),
    "lpcnet": ("F03", "neural_vocoder"),
    "wavenet": ("F04", "neural_vocoder"),
    "pwg": ("F05", "neural_vocoder"),
    "hifigan": ("F06", "neural_vocoder"),
    "mbmelgan": ("F07", "neural_vocoder"),
    "stylegan": ("F08", "neural_vocoder"),
    "world": ("F09", "traditional_vocoder"),
    "fasthifigan": ("F10", "tts"),
    "tacohifigan": ("F11", "tts"),
    "partiallyfake": ("F12", "partially_fake"),
}


EXPECTED_REAL_BY_SPLIT = {
    "train": {"aishell1", "aishell3", "thchs30", "magicread"},
    "dev": {"aishell1", "aishell3", "thchs30", "magicread"},
    "test_seen": {"aishell1", "aishell3", "thchs30", "magicread"},
    "test_unseen": {"magicconversa", "selfrecording"},
}


EXPECTED_FAKE_BY_SPLIT = {
    "train": {
        "straight",
        "gl",
        "lpcnet",
        "wavenet",
        "pwg",
        "hifigan",
        "mbmelgan",
        "stylegan",
    },
    "dev": {
        "straight",
        "gl",
        "lpcnet",
        "wavenet",
        "pwg",
        "hifigan",
        "mbmelgan",
        "stylegan",
    },
    "test_seen": {
        "straight",
        "gl",
        "lpcnet",
        "wavenet",
        "pwg",
        "hifigan",
        "mbmelgan",
        "stylegan",
    },
    "test_unseen": {
        "world",
        "fasthifigan",
        "tacohifigan",
        "partiallyfake",
    },
}


def _list_subdirectories(path: Path) -> set[str]:
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")

    result: set[str] = set()

    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                result.add(entry.name)

    return result


def _iter_wav_files(path: Path) -> Iterator[Path]:
    """
    Enumerate only immediate WAV files in one known CFAD leaf
    directory. No recursive filesystem walk is performed.
    """
    with os.scandir(path) as entries:
        for entry in entries:
            if (
                entry.is_file(follow_symlinks=False)
                and entry.name.lower().endswith(".wav")
            ):
                yield Path(entry.path)


def _check_structure(
    real_root: Path,
    fake_root: Path,
    split: str,
) -> None:
    actual_real = _list_subdirectories(real_root)
    actual_fake = _list_subdirectories(fake_root)

    expected_real = EXPECTED_REAL_BY_SPLIT[split]
    expected_fake = EXPECTED_FAKE_BY_SPLIT[split]

    if actual_real != expected_real:
        raise ValueError(
            f"{real_root}: unexpected source directories. "
            f"Expected {sorted(expected_real)}, "
            f"found {sorted(actual_real)}"
        )

    if actual_fake != expected_fake:
        raise ValueError(
            f"{fake_root}: unexpected fake directories. "
            f"Expected {sorted(expected_fake)}, "
            f"found {sorted(actual_fake)}"
        )


def iter_split(
    extracted_root: str | Path,
    split: str,
) -> Iterator[ManifestRow]:
    if split not in SPLITS:
        raise ValueError(
            f"Unknown split {split!r}. "
            f"Expected one of: {sorted(SPLITS)}"
        )

    root = Path(extracted_root)

    split_root = (
        root
        / "clean_version"
        / SPLITS[split]
    )

    real_root = split_root / "real_clean"
    fake_root = split_root / "fake_clean"

    _check_structure(real_root, fake_root, split)

    # Bona fide
    for source_dir in sorted(EXPECTED_REAL_BY_SPLIT[split]):
        source_id, _source_name = REAL_SOURCES[source_dir]
        leaf = real_root / source_dir

        for audio_path in _iter_wav_files(leaf):
            relative_id = (
                Path("real_clean")
                / source_dir
                / audio_path.stem
            )

            yield ManifestRow(
                path=str(audio_path),
                label=0,
                dataset=DATASET_NAME,
                split=split,
                utterance_id=relative_id.as_posix(),
                source_id=source_id,
                language="zh",
                condition="clean",
            )

    # Spoof
    for fake_dir in sorted(EXPECTED_FAKE_BY_SPLIT[split]):
        attack_id, attack_family = FAKE_TYPES[fake_dir]
        leaf = fake_root / fake_dir

        for audio_path in _iter_wav_files(leaf):
            relative_id = (
                Path("fake_clean")
                / fake_dir
                / audio_path.stem
            )

            yield ManifestRow(
                path=str(audio_path),
                label=1,
                dataset=DATASET_NAME,
                split=split,
                utterance_id=relative_id.as_posix(),
                attack_id=attack_id,
                attack_family=attack_family,
                language="zh",
                condition="clean",
            )


def build_split_manifest(
    extracted_root: str | Path,
    split: str,
    output_path: str | Path,
) -> int:
    return write_manifest(
        iter_split(extracted_root, split),
        output_path,
    )
