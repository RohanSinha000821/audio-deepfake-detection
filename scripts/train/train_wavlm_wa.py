#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from audio_deepfake_detection.data.collate import collate_audio_batch
from audio_deepfake_detection.data.datasets import ManifestDataset
from audio_deepfake_detection.data.samplers import DatasetBalancedSampler
from audio_deepfake_detection.protocols.lodo import load_lodo_config
from audio_deepfake_detection.sota.wavlm_backend.model import WavLMWA
from audio_deepfake_detection.sota.wavlm_backend.training import (
    EarlyStoppingState,
    create_optimizer_and_scheduler,
    create_weighted_cross_entropy,
    load_checkpoint,
    macro_source_dev_eer,
    save_checkpoint,
    score_source_dev_domains,
    train_one_epoch,
)
from audio_deepfake_detection.training.reproducibility import (
    make_generator,
    seed_everything,
    seed_worker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the WavLM Base weighted-average baseline",
    )
    parser.add_argument("--fold", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sota/wavlm_wa.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume",
        nargs="?",
        type=Path,
        const=None,
        default=False,
        help=(
            "Resume from a supplied checkpoint; without a path, use "
            "OUTPUT_DIR/last.pt"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    return parser.parse_args()


def make_train_loader(
    manifest_paths: list[str],
    config: dict,
) -> tuple[DataLoader, DatasetBalancedSampler]:
    dataset = ManifestDataset(
        manifest_paths,
        load_audio=True,
        max_seconds=float(config["train_crop_seconds"]),
        random_crop=True,
        target_sample_rate=int(config["sample_rate"]),
    )
    sampler = DatasetBalancedSampler(
        dataset,
        num_samples=config.get("num_samples"),
        seed=int(config["seed"]),
    )
    num_workers = int(config["num_workers"])
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_audio_batch,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=make_generator(int(config["seed"])),
        persistent_workers=(num_workers > 0),
    )

    return loader, sampler


def make_source_dev_loaders(
    source_dev_refs,
    config: dict,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    num_workers = int(config["num_workers"])

    for ref in source_dev_refs:
        dataset = ManifestDataset(
            ref.path,
            load_audio=True,
            max_seconds=None,
            random_crop=False,
            target_sample_rate=int(config["sample_rate"]),
        )
        loaders[ref.dataset] = DataLoader(
            dataset,
            batch_size=int(config["eval_batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_audio_batch,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    return loaders


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(
        args.config.read_text(encoding="utf-8")
    )

    if config.get("initial_weight_l2_lambda") is not None:
        raise NotImplementedError(
            "L2-to-initial-WavLM is disabled until its coefficient "
            "and reduction are fixed as a reproduction assumption"
        )

    seed_everything(int(config["seed"]), deterministic=True)
    # Structural YAML validation only: the training process never
    # opens, stats, validates, or constructs a held-out target dataset.
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

    train_loader, sampler = make_train_loader(
        [ref.path for ref in fold.source_train],
        config,
    )
    source_dev_loaders = make_source_dev_loaders(
        fold.source_dev,
        config,
    )

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
    loss_function = create_weighted_cross_entropy(
        tuple(float(value) for value in config["class_weights"]),
        device=device,
    )
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        encoder_lr=float(config["encoder_lr"]),
        backend_lr=float(config["backend_lr"]),
        lr_gamma=float(config["lr_gamma"]),
        layerwise_lr_decay=config.get("layerwise_lr_decay"),
    )

    early_stopping = EarlyStoppingState(
        patience=int(config["early_stopping_patience"]),
    )
    start_epoch = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = args.output_dir / "best.pt"
    last_checkpoint = args.output_dir / "last.pt"

    if args.resume is not False:
        resume_path = (
            last_checkpoint
            if args.resume is None
            else args.resume
        )
        checkpoint = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
            expected_config=config,
            expected_checkpoint_role=(
                "last" if args.resume is None else None
            ),
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        early_stopping = EarlyStoppingState(
            **checkpoint["early_stopping"]
        )
    bf16 = bool(config["bf16"])

    for epoch in range(start_epoch, int(config["max_epochs"])):
        sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device=device,
            gradient_accumulation_steps=int(
                config["gradient_accumulation_steps"]
            ),
            bf16=bf16,
        )
        domain_results = score_source_dev_domains(
            model,
            source_dev_loaders,
            device=device,
            bf16=bf16,
        )
        macro_eer = macro_source_dev_eer(domain_results)
        improved = early_stopping.update(macro_eer)

        # Paper schedule: both learning rates are reduced 5% after
        # every completed epoch. Save the post-step state for resume.
        scheduler.step()

        if improved:
            save_checkpoint(
                best_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                early_stopping=early_stopping,
                config=config,
                checkpoint_role="best",
            )

        save_checkpoint(
            last_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            early_stopping=early_stopping,
            config=config,
            checkpoint_role="last",
        )

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "source_dev_eer": {
                        dataset: result["eer"]
                        for dataset, result in domain_results.items()
                    },
                    "macro_source_dev_eer": macro_eer,
                    "best_macro_source_dev_eer": (
                        early_stopping.best_metric
                    ),
                    "epochs_without_improvement": (
                        early_stopping.epochs_without_improvement
                    ),
                    "saved_best": improved,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        if early_stopping.should_stop:
            break


if __name__ == "__main__":
    main()
