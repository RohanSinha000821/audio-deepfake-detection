from __future__ import annotations

import csv
from pathlib import Path


VALID_LABELS = {"0", "1"}


def validate_manifest(
    manifest_path: str | Path,
    *,
    check_files: bool = False,
) -> dict[str, int]:
    path = Path(manifest_path)

    if not path.is_file():
        raise FileNotFoundError(path)

    required = {
        "path",
        "label",
        "dataset",
        "split",
        "utterance_id",
    }

    total = 0
    bonafide = 0
    spoof = 0
    missing_files = 0
    seen_ids: set[str] = set()

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")

        missing_columns = required - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                f"{path}: missing required columns: "
                f"{sorted(missing_columns)}"
            )

        for line_number, row in enumerate(reader, start=2):
            total += 1

            for column in required:
                if row[column] is None or not row[column].strip():
                    raise ValueError(
                        f"{path}:{line_number}: "
                        f"empty required field {column!r}"
                    )

            label = row["label"]

            if label not in VALID_LABELS:
                raise ValueError(
                    f"{path}:{line_number}: "
                    f"invalid label {label!r}"
                )

            if label == "0":
                bonafide += 1
            else:
                spoof += 1

            utterance_id = row["utterance_id"]

            if utterance_id in seen_ids:
                raise ValueError(
                    f"{path}:{line_number}: "
                    f"duplicate utterance_id {utterance_id!r}"
                )

            seen_ids.add(utterance_id)

            if check_files and not Path(row["path"]).is_file():
                missing_files += 1

    return {
        "total": total,
        "bonafide": bonafide,
        "spoof": spoof,
        "missing_files": missing_files,
    }
