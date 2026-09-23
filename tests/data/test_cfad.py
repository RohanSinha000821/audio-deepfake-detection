from pathlib import Path

from audio_deepfake_detection.data.cfad import iter_split


TRAIN_REAL = [
    "aishell1",
    "aishell3",
    "thchs30",
    "magicread",
]

TRAIN_FAKE = [
    "straight",
    "gl",
    "lpcnet",
    "wavenet",
    "pwg",
    "hifigan",
    "mbmelgan",
    "stylegan",
]


def create_split(root: Path, split_dir: str, real_dirs, fake_dirs):
    real_root = root / "clean_version" / split_dir / "real_clean"
    fake_root = root / "clean_version" / split_dir / "fake_clean"

    for name in real_dirs:
        (real_root / name).mkdir(parents=True)

    for name in fake_dirs:
        (fake_root / name).mkdir(parents=True)

    return real_root, fake_root


def test_cfad_train_parser(tmp_path):
    root = tmp_path / "extracted"

    real_root, fake_root = create_split(
        root,
        "train_clean",
        TRAIN_REAL,
        TRAIN_FAKE,
    )

    (real_root / "aishell1" / "BAC001.wav").write_bytes(b"")
    (fake_root / "hifigan" / "SSB0001_hifigan.wav").write_bytes(b"")

    rows = list(iter_split(root, "train"))

    assert len(rows) == 2

    real = next(row for row in rows if row.label == 0)
    fake = next(row for row in rows if row.label == 1)

    assert real.source_id == "R01"
    assert real.language == "zh"
    assert real.condition == "clean"
    assert real.utterance_id == "real_clean/aishell1/BAC001"

    assert fake.attack_id == "F06"
    assert fake.attack_family == "neural_vocoder"
    assert fake.utterance_id == (
        "fake_clean/hifigan/SSB0001_hifigan"
    )


def test_cfad_unseen_parser(tmp_path):
    root = tmp_path / "extracted"

    real_root, fake_root = create_split(
        root,
        "test_unseen_clean",
        ["magicconversa", "selfrecording"],
        [
            "world",
            "fasthifigan",
            "tacohifigan",
            "partiallyfake",
        ],
    )

    (real_root / "selfrecording" / "f001.wav").write_bytes(b"")
    (
        fake_root
        / "partiallyfake"
        / "example_replaceOnce.wav"
    ).write_bytes(b"")

    rows = list(iter_split(root, "test_unseen"))

    assert len(rows) == 2

    real = next(row for row in rows if row.label == 0)
    fake = next(row for row in rows if row.label == 1)

    assert real.source_id == "R06"

    assert fake.attack_id == "F12"
    assert fake.attack_family == "partially_fake"


def test_cfad_files_are_emitted_in_filename_order(tmp_path):
    root = tmp_path / "extracted"
    real_root, _fake_root = create_split(
        root,
        "train_clean",
        TRAIN_REAL,
        TRAIN_FAKE,
    )

    leaf = real_root / "aishell1"

    for filename in ("c.wav", "a.wav", "b.wav"):
        (leaf / filename).write_bytes(b"")

    rows = [
        row
        for row in iter_split(root, "train")
        if row.source_id == "R01"
    ]

    assert [row.utterance_id for row in rows] == [
        "real_clean/aishell1/a",
        "real_clean/aishell1/b",
        "real_clean/aishell1/c",
    ]
