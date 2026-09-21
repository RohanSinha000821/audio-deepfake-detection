from audio_deepfake_detection.data.asvspoof5 import (
    iter_split,
)


def test_asvspoof5_train_parser(tmp_path):
    root = tmp_path / "extracted"

    protocols = root / "protocols"
    protocols.mkdir(parents=True)

    protocol = protocols / "ASVspoof5.train.tsv"

    protocol.write_text(
        "T_0001 T_0000000000 F - - - AC3 A05 spoof -\n"
        "T_0002 T_0000000001 M - - - - bonafide bonafide -\n",
        encoding="utf-8",
    )

    rows = list(iter_split(root, "train"))

    assert len(rows) == 2

    spoof = rows[0]
    bona = rows[1]

    assert spoof.label == 1
    assert spoof.attack_id == "A05"
    assert spoof.attack_family == "tts"
    assert spoof.attack_config == "AC3"
    assert spoof.gender == "F"

    assert bona.label == 0
    assert bona.attack_id == ""
    assert bona.attack_family == ""


def test_asvspoof5_eval_codec_parser(tmp_path):
    root = tmp_path / "extracted"

    protocols = root / "protocols"
    protocols.mkdir(parents=True)

    protocol = protocols / "ASVspoof5.eval.track_1.tsv"

    protocol.write_text(
        "E_1607 E_0009538969 M C05 2 E_0009486171 "
        "AC1 A26 spoof -\n",
        encoding="utf-8",
    )

    rows = list(iter_split(root, "eval"))

    assert len(rows) == 1

    row = rows[0]

    assert row.label == 1
    assert row.attack_id == "A26"
    assert row.attack_family == "vc"

    assert row.codec == "C05"
    assert row.codec_quality == "2"
    assert row.codec_seed == "E_0009486171"
    assert row.condition == "codec"
