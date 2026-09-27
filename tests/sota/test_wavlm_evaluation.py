import json
from types import SimpleNamespace

import pytest
import torch

from scripts.evaluate import evaluate_wavlm_wa


def test_thresholds_persist_before_target_construction(monkeypatch, tmp_path):
    source_ref = SimpleNamespace(
        dataset="source-domain",
        split="dev",
        path="source.csv",
    )
    target_ref = SimpleNamespace(
        dataset="held-out",
        split="test",
        path="target.csv",
    )
    fold = SimpleNamespace(
        fold="F-test",
        source_dev=(source_ref,),
        target_primary=target_ref,
        target_secondary=None,
    )
    events = []

    def fake_make_eval_loader(ref, config):
        del config
        events.append(f"construct:{ref.dataset}:{ref.split}")

        if ref is target_ref:
            thresholds_path = tmp_path / "thresholds.json"
            assert thresholds_path.is_file()
            persisted = json.loads(thresholds_path.read_text())
            assert set(persisted["source_apcer_thresholds"]) == {
                "0.01",
                "0.05",
                "0.1",
            }

        return ref

    def fake_score_loader_with_metadata(
        model,
        loader,
        *,
        device,
        bf16,
        partition,
    ):
        del model, device, bf16
        events.append(f"score:{partition}")
        labels = [0, 1]
        scores = [0.1, 0.9] if loader is source_ref else [0.2, 0.8]
        records = [
            {
                "utterance_id": f"{loader.dataset}-bona",
                "dataset": loader.dataset,
                "split": loader.split,
                "label": labels[0],
                "score": scores[0],
            },
            {
                "utterance_id": f"{loader.dataset}-spoof",
                "dataset": loader.dataset,
                "split": loader.split,
                "label": labels[1],
                "score": scores[1],
            },
        ]
        return labels, scores, records

    monkeypatch.setattr(
        evaluate_wavlm_wa,
        "make_eval_loader",
        fake_make_eval_loader,
    )
    monkeypatch.setattr(
        evaluate_wavlm_wa,
        "score_loader_with_metadata",
        fake_score_loader_with_metadata,
    )

    report = evaluate_wavlm_wa.evaluate_checkpoint(
        model=object(),
        fold=fold,
        config={},
        checkpoint_path=tmp_path / "best.pt",
        checkpoint_epoch=4,
        artifact_dir=tmp_path,
        device=torch.device("cpu"),
        bf16=False,
    )

    assert events == [
        "construct:source-domain:dev",
        "score:source_dev:source-domain:dev",
        "construct:held-out:test",
        "score:target:held-out:test",
    ]
    assert report["checkpoint_epoch"] == 4
    assert report["thresholds_file"] == str(tmp_path / "thresholds.json")
    source_thresholds = report["source_pooled_apcer_thresholds"]
    transferred = report["targets"]["held-out:test"]["transferred"]

    for target, selection in source_thresholds.items():
        assert transferred[float(target)]["threshold"] == pytest.approx(
            selection["threshold"]
        )

    source_tsv = (
        tmp_path / "scores/source_dev_source-domain_dev.tsv"
    )
    target_tsv = tmp_path / "scores/target_held-out_test.tsv"
    assert source_tsv.read_text().splitlines()[0] == (
        "utterance_id\tdataset\tsplit\tlabel\tscore"
    )
    assert target_tsv.read_text().splitlines()[1].endswith("\t0\t0.2")
