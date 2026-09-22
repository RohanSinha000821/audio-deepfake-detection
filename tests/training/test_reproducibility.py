import random

import numpy as np
import torch

from audio_deepfake_detection.training.reproducibility import (
    make_generator,
    seed_everything,
)


def test_seed_everything_is_repeatable():
    seed_everything(1234)

    first = (
        random.random(),
        np.random.rand(),
        torch.rand(1).item(),
    )

    seed_everything(1234)

    second = (
        random.random(),
        np.random.rand(),
        torch.rand(1).item(),
    )

    assert first == second


def test_generator_is_repeatable():
    generator1 = make_generator(42)
    generator2 = make_generator(42)

    values1 = torch.rand(
        20,
        generator=generator1,
    )

    values2 = torch.rand(
        20,
        generator=generator2,
    )

    assert torch.equal(
        values1,
        values2,
    )
