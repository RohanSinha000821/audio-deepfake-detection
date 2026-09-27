#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

import torch
import yaml
from torch.utils.data import DataLoader

from audio_deepfake_detection.data.collate import collate_audio_batch
from audio_deepfake_detection.data.datasets import ManifestDataset
from audio_deepfake_detection.metrics.evaluation import (
    evaluate_scores,
    select_source_operating_points,
)
from audio_deepfake_detection.protocols.lodo import load_lodo_config
from audio_deepfake_detection.sota.wavlm_backend.model import WavLMWA
from audio_deepfake_detection.sota.wavlm_backend.training import (
    load_checkpoint,
    score_loader_with_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a source-selected WavLM WA checkpoint",
    )
    parser.add_argument("--fold", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sota/wavlm_wa.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    return parser.parse_args()


def make_eval_loader(ref, config: dict) -> DataLoader:
    dataset = ManifestDataset(
        ref.path,
        load_audio=True,
        max_seconds=None,
        random_crop=False,
        target_sample_rate=int(config["sample_rate"]),
    )
    num_workers = int(config["num_workers"])

    return DataLoader(
        dataset,
        batch_size=int(config["eval_batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_audio_batch,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)

        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _write_score_tsv(
    path: Path,
    records: list[dict[str, str | int | float]],
) -> None:
    rows = ["utterance_id	dataset	split	label	score"]
    rows.extend(
        "	".join(
            (
                str(record["utterance_id"]),
                str(record["dataset"]),
                str(record["split"]),
                str(record["label"]),
                repr(float(record["score"])),
            )
        )
        for record in records
    )
    _atomic_write_text(path, "\n".join(rows) + "\n")


def _artifact_name(role: str, ref) -> str:
    return f"{role}_{ref.dataset}_{ref.split}.tsv"


def evaluate_checkpoint(
    *,
    model: WavLMWA,
    fold,
    config: dict,
    checkpoint_path: Path,
    checkpoint_epoch: int,
    artifact_dir: Path,
    device: torch.device,
    bf16: bool,
) -> dict:
    scores_dir = artifact_dir / "scores"
    source_report: dict[str, dict] = {}
    source_score_files: dict[str, str] = {}
    pooled_labels: list[int] = []
    pooled_scores: list[float] = []

    # Source development is scored and calibration is frozen and persisted
    # before any target dataset is constructed or read.
    for ref in fold.source_dev:
        partition = f"source_dev:{ref.dataset}:{ref.split}"
        labels, scores, records = score_loader_with_metadata(
            model,
            make_eval_loader(ref, config),
            device=device,
            bf16=bf16,
            partition=partition,
        )
        score_path = scores_dir / _artifact_name("source_dev", ref)
        _write_score_tsv(score_path, records)
        source_score_files[f"{ref.dataset}:{ref.split}"] = str(score_path)
        source_report[ref.dataset] = evaluate_scores(labels, scores)
        pooled_labels.extend(labels)
        pooled_scores.extend(scores)

    operating_points = select_source_operating_points(
        pooled_labels,
        pooled_scores,
    )
    thresholds_path = artifact_dir / "thresholds.json"
    thresholds_report = {
        "fold": fold.fold,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint_epoch),
        "source_apcer_thresholds": {
            str(target): asdict(selection)
            for target, selection in operating_points.items()
        },
    }
    _atomic_write_text(
        thresholds_path,
        json.dumps(thresholds_report, indent=2, sort_keys=True) + "\n",
    )

    target_refs = [fold.target_primary]

    if fold.target_secondary is not None:
        target_refs.append(fold.target_secondary)

    target_report: dict[str, dict] = {}
    target_score_files: dict[str, str] = {}

    for ref in target_refs:
        key = f"{ref.dataset}:{ref.split}"
        labels, scores, records = score_loader_with_metadata(
            model,
            make_eval_loader(ref, config),
            device=device,
            bf16=bf16,
            partition=f"target:{key}",
        )
        score_path = scores_dir / _artifact_name("target", ref)
        _write_score_tsv(score_path, records)
        target_score_files[key] = str(score_path)
        target_report[key] = evaluate_scores(
            labels,
            scores,
            transferred_thresholds=operating_points,
        )

    return {
        "fold": fold.fold,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint_epoch),
        "source_dev": source_report,
        "source_dev_macro_eer": sum(
            float(metrics["eer"])
            for metrics in source_report.values()
        ) / len(source_report),
        "source_pooled_apcer_thresholds": {
            str(target): asdict(selection)
            for target, selection in operating_points.items()
        },
        "targets": target_report,
        "score_files": {
            "source_dev": source_score_files,
            "targets": target_score_files,
        },
        "thresholds_file": str(thresholds_path),
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(
        args.config.read_text(encoding="utf-8")
    )
    fold = load_lodo_config(args.fold)
    config = dict(config)
    config["lodo_fold"] = fold.fold
    config["held_out_dataset"] = fold.held_out_dataset
    device = torch.device(args.device)
    local_files_only = (
        bool(config.get("local_files_only", False))
        if args.local_files_only is None
        else args.local_files_only
    )
    bf16 = bool(config["bf16"])

    model = WavLMWA.from_pretrained(
        str(config["model_name"]),
        revision=config.get("revision"),
        local_files_only=local_files_only,
        layer_weight_normalization=str(
            config["layer_weight_normalization"]
        ),
        input_normalization=bool(
            config.get("input_normalization", False)
        ),
        apply_spec_augment=bool(
            config.get("apply_spec_augment", False)
        ),
    ).to(device)
    checkpoint = load_checkpoint(
        args.checkpoint,
        model=model,
        map_location=device,
        expected_config=config,
        expected_checkpoint_role="best",
    )
    artifact_dir = (
        args.output.parent if args.output is not None else Path.cwd()
    )
    report = evaluate_checkpoint(
        model=model,
        fold=fold,
        config=config,
        checkpoint_path=args.checkpoint,
        checkpoint_epoch=int(checkpoint["epoch"]),
        artifact_dir=artifact_dir,
        device=device,
        bf16=bf16,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)

    if args.output is not None:
        _atomic_write_text(args.output, rendered + "\n")

    print(rendered)


if __name__ == "__main__":
    main()
