import csv

from audio_deepfake_detection.data.asvspoof2019 import (
    build_split_manifest,
)


def test_asvspoof2019_train_parser(tmp_path):
    root = tmp_path / "extracted"

    protocols = (
        root / "ASVspoof2019_LA_cm_protocols"
    )
    protocols.mkdir(parents=True)

    protocol = (
        protocols
        / "ASVspoof2019.LA.cm.train.trn.txt"
    )

    protocol.write_text(
        "LA_0001 LA_T_0000001 - - bonafide\n"
        "LA_0002 LA_T_0000002 - A05 spoof\n",
        encoding="utf-8",
    )

    output = tmp_path / "train.csv"

    count = build_split_manifest(
        root,
        "train",
        output,
    )

    assert count == 2

    with output.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["label"] == "0"
    assert rows[0]["attack_id"] == ""

    assert rows[1]["label"] == "1"
    assert rows[1]["attack_id"] == "A05"
    assert rows[1]["attack_family"] == "vc"

    assert rows[0]["utterance_id"] == "LA_T_0000001"
