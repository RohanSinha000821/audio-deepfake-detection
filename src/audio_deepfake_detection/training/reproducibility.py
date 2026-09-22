from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(
    seed: int,
    *,
    deterministic: bool = False,
) -> None:
    """
    Seed Python, NumPy, PyTorch CPU, and CUDA RNGs.
    """

    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )


def seed_worker(
    worker_id: int,
) -> None:
    """
    Seed Python and NumPy inside a DataLoader worker.

    PyTorch assigns each worker a deterministic initial seed.
    """

    del worker_id

    worker_seed = (
        torch.initial_seed()
        % (2**32)
    )

    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(
    seed: int,
) -> torch.Generator:
    """
    Create a deterministically seeded PyTorch generator.
    """

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    return generator
