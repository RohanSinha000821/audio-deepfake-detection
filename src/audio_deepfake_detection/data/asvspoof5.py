from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .schema import ManifestRow, write_manifest


DATASET_NAME = "asvspoof5"


SPLITS = {
    "train": {
        "protocol": "ASVspoof5.train.tsv",
        "audio_dir": "flac_T",
    },
    "dev": {
        "protocol": "ASVspoof5.dev.track_1.tsv",
        "audio_dir": "flac_D",
    },
    "eval": {
        "protocol": "ASVspoof5.eval.track_1.tsv",
        "audio_dir": "flac_E_eval",
    },
}


# ASVspoof 5 attack categories.
#
# Train:
#   A01-A08: TTS
#
# Development:
#   A09,A10,A11,A12,A14: TTS
#   A13,A15,A16: VC
#
# Evaluation:
#   A17,A19,A21,A22,A28,A29: TTS
#   A24,A25,A26: VC
#   A18,A20,A23,A27,A30,A31,A32: adversarial
#
# Evaluation therefore contains 6 TTS + 3 VC + 7 adversarial
# attacks.
ATTACK_FAMILY = {
    **{f"A{i:02d}": "tts" for i in range(1, 9)},

    "A09": "tts",
    "A10": "tts",
    "A11": "tts",
    "A12": "tts",
    "A13": "vc",
    "A14": "tts",
    "A15": "vc",
    "A16": "vc",

    "A17": "tts",
    "A18": "adversarial",
    "A19": "tts",
    "A20": "adversarial",
    "A21": "tts",
    "A22": "tts",
    "A23": "adversarial",
    "A24": "vc",
    "A25": "vc",
    "A26": "vc",
    "A27": "adversarial",
    "A28": "tts",
    "A29": "tts",
    "A30": "adversarial",
    "A31": "adversarial",
    "A32": "adversarial",
}


def clean_optional(value: str) -> str:
    return "" if value == "-" else value


def iter_split(
    extracted_root: str | Path,
    split: str,
) -> Iterator[ManifestRow]:
    """
    Parse an official ASVspoof 5 Track-1 protocol.

    The protocol contains ten whitespace-separated fields:

        speaker_id
        utterance_id
        gender
        codec
        codec_quality
        codec_seed
        attack_config
        attack_id
        key
        tmp

    Only official Track-1 protocol rows are emitted.
    """

    if split not in SPLITS:
        raise ValueError(
            f"Unknown split {split!r}. "
            f"Expected one of: {sorted(SPLITS)}"
        )

    root = Path(extracted_root)
    split_info = SPLITS[split]

    protocol_path = (
        root / "protocols" / split_info["protocol"]
    )

    audio_root = root / split_info["audio_dir"]

    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"Protocol not found: {protocol_path}"
        )

    with protocol_path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 10:
                raise ValueError(
                    f"{protocol_path}:{line_number}: "
                    f"expected 10 fields, got {len(parts)}: {parts}"
                )

            (
                speaker_id,
                utterance_id,
                gender,
                codec,
                codec_quality,
                codec_seed,
                attack_config,
                attack_id,
                key,
                _tmp,
            ) = parts

            key = key.lower()

            if key == "bonafide":
                label = 0
                canonical_attack_id = ""
                attack_family = ""
                canonical_attack_config = ""

            elif key == "spoof":
                label = 1
                canonical_attack_id = attack_id
                canonical_attack_config = clean_optional(
                    attack_config
                )

                if attack_id not in ATTACK_FAMILY:
                    raise ValueError(
                        f"{protocol_path}:{line_number}: "
                        f"unknown attack ID {attack_id!r}"
                    )

                attack_family = ATTACK_FAMILY[attack_id]

            else:
                raise ValueError(
                    f"{protocol_path}:{line_number}: "
                    f"unexpected key {key!r}"
                )

            canonical_codec = clean_optional(codec)

            audio_path = audio_root / f"{utterance_id}.flac"

            yield ManifestRow(
                path=str(audio_path),
                label=label,
                dataset=DATASET_NAME,
                split=split,
                utterance_id=utterance_id,
                speaker_id=speaker_id,
                gender=clean_optional(gender),
                attack_id=canonical_attack_id,
                attack_family=attack_family,
                attack_config=canonical_attack_config,
                language="en",
                condition=(
                    "codec"
                    if canonical_codec
                    else "original"
                ),
                codec=canonical_codec,
                codec_quality=clean_optional(codec_quality),
                codec_seed=clean_optional(codec_seed),
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
