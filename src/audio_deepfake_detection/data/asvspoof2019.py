from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .schema import ManifestRow, write_manifest


DATASET_NAME = "asvspoof2019"

SPLITS = {
    "train": {
        "protocol": "ASVspoof2019.LA.cm.train.trn.txt",
        "audio_dir": "ASVspoof2019_LA_train/flac",
    },
    "dev": {
        "protocol": "ASVspoof2019.LA.cm.dev.trl.txt",
        "audio_dir": "ASVspoof2019_LA_dev/flac",
    },
    "eval": {
        "protocol": "ASVspoof2019.LA.cm.eval.trl.txt",
        "audio_dir": "ASVspoof2019_LA_eval/flac",
    },
}


# Attack categories from the ASVspoof 2019 LA dataset paper.
ATTACK_FAMILY = {
    "A01": "tts",
    "A02": "tts",
    "A03": "tts",
    "A04": "tts",
    "A05": "vc",
    "A06": "vc",
    "A07": "tts",
    "A08": "tts",
    "A09": "tts",
    "A10": "tts",
    "A11": "tts",
    "A12": "tts",
    "A13": "tts_vc",
    "A14": "tts_vc",
    "A15": "tts_vc",
    "A16": "tts",
    "A17": "vc",
    "A18": "vc",
    "A19": "vc",
}


def iter_split(
    extracted_root: str | Path,
    split: str,
) -> Iterator[ManifestRow]:
    """
    Parse one official ASVspoof2019 LA CM protocol.

    Protocol format:
        speaker_id utterance_id - attack_id key
    """

    if split not in SPLITS:
        raise ValueError(
            f"Unknown split {split!r}. "
            f"Expected one of: {sorted(SPLITS)}"
        )

    root = Path(extracted_root)

    split_info = SPLITS[split]

    protocol_path = (
        root
        / "ASVspoof2019_LA_cm_protocols"
        / split_info["protocol"]
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

            if len(parts) != 5:
                raise ValueError(
                    f"{protocol_path}:{line_number}: "
                    f"expected 5 fields, got {len(parts)}: {parts}"
                )

            (
                speaker_id,
                utterance_id,
                _,
                attack_id,
                key,
            ) = parts

            key = key.lower()

            if key == "bonafide":
                label = 0
                canonical_attack_id = ""
                attack_family = ""

            elif key == "spoof":
                label = 1
                canonical_attack_id = attack_id

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

            audio_path = audio_root / f"{utterance_id}.flac"

            yield ManifestRow(
                path=str(audio_path),
                label=label,
                dataset=DATASET_NAME,
                split=split,
                utterance_id=utterance_id,
                speaker_id=speaker_id,
                attack_id=canonical_attack_id,
                attack_family=attack_family,
                language="en",
                condition="original",
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
