#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from audio_deepfake_detection.data.asvspoof2019 import (
    build_split_manifest as build_asv19_split,
)
from audio_deepfake_detection.data.asvspoof5 import (
    build_split_manifest as build_asv5_split,
)
from audio_deepfake_detection.data.cfad import (
    build_split_manifest as build_cfad_split,
)


DEFAULT_DATA_ROOT = Path(
    "/mnt/salt/datasets/audio-deepfake"
)

DEFAULT_OUTPUT_ROOT = Path("manifests")


def build_asvspoof2019(
    data_root: Path,
    output_root: Path,
) -> None:
    extracted_root = (
        data_root
        / "asvspoof2019"
        / "extracted"
    )

    output_dir = output_root / "asvspoof2019"

    for split in ("train", "dev", "eval"):
        output_path = output_dir / f"{split}.csv"

        count = build_asv19_split(
            extracted_root=extracted_root,
            split=split,
            output_path=output_path,
        )

        print(
            f"ASVspoof2019 {split:11s}: "
            f"{count:>8,d} rows -> {output_path}"
        )


def build_asvspoof5(
    data_root: Path,
    output_root: Path,
) -> None:
    extracted_root = (
        data_root
        / "asvspoof5"
        / "extracted"
    )

    output_dir = output_root / "asvspoof5"

    for split in ("train", "dev", "eval"):
        output_path = output_dir / f"{split}.csv"

        count = build_asv5_split(
            extracted_root=extracted_root,
            split=split,
            output_path=output_path,
        )

        print(
            f"ASVspoof5     {split:11s}: "
            f"{count:>8,d} rows -> {output_path}"
        )


def build_cfad(
    data_root: Path,
    output_root: Path,
) -> None:
    extracted_root = (
        data_root
        / "cfad"
        / "extracted"
    )

    output_dir = output_root / "cfad"

    for split in (
        "train",
        "dev",
        "test_seen",
        "test_unseen",
    ):
        output_path = output_dir / f"{split}.csv"

        count = build_cfad_split(
            extracted_root=extracted_root,
            split=split,
            output_path=output_path,
        )

        print(
            f"CFAD          {split:11s}: "
            f"{count:>8,d} rows -> {output_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=[
            "asvspoof2019",
            "asvspoof5",
            "cfad",
        ],
        required=True,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    args = parser.parse_args()

    if args.dataset == "asvspoof2019":
        build_asvspoof2019(
            args.data_root,
            args.output_root,
        )

    elif args.dataset == "asvspoof5":
        build_asvspoof5(
            args.data_root,
            args.output_root,
        )

    elif args.dataset == "cfad":
        build_cfad(
            args.data_root,
            args.output_root,
        )


if __name__ == "__main__":
    main()
