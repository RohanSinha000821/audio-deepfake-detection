from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler

from .datasets import ManifestDataset


class DatasetClassBalancedSampler(Sampler[int]):
    """
    Implements the project's required sampling strategy:

        1. choose dataset uniformly
        2. choose class uniformly
        3. choose utterance uniformly

    Sampling is with replacement.

    This prevents large datasets such as SpeechFake from
    dominating smaller source domains such as CFAD.
    """

    def __init__(
        self,
        dataset: ManifestDataset,
        *,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:

        self.dataset = dataset

        self.num_samples = (
            len(dataset)
            if num_samples is None
            else int(num_samples)
        )

        if self.num_samples <= 0:
            raise ValueError(
                "num_samples must be positive"
            )

        self.seed = int(seed)
        self.epoch = 0

        buckets: dict[
            tuple[str, int],
            list[int],
        ] = defaultdict(list)

        for index, record in enumerate(
            dataset.records
        ):
            buckets[
                (record.dataset, record.label)
            ].append(index)

        self.datasets = sorted(
            {
                record.dataset
                for record in dataset.records
            }
        )

        if not self.datasets:
            raise ValueError(
                "Dataset contains no dataset domains"
            )

        for dataset_name in self.datasets:
            for label in (0, 1):
                key = (dataset_name, label)

                if not buckets[key]:
                    raise ValueError(
                        "Balanced sampling requires both "
                        f"classes for dataset "
                        f"{dataset_name!r}; "
                        f"class {label} is empty"
                    )

        self.buckets = dict(buckets)

    def set_epoch(
        self,
        epoch: int,
    ) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()

        generator.manual_seed(
            self.seed + self.epoch
        )

        num_datasets = len(self.datasets)

        for _ in range(self.num_samples):

            dataset_index = int(
                torch.randint(
                    low=0,
                    high=num_datasets,
                    size=(1,),
                    generator=generator,
                ).item()
            )

            dataset_name = self.datasets[
                dataset_index
            ]

            label = int(
                torch.randint(
                    low=0,
                    high=2,
                    size=(1,),
                    generator=generator,
                ).item()
            )

            bucket = self.buckets[
                (dataset_name, label)
            ]

            utterance_index = int(
                torch.randint(
                    low=0,
                    high=len(bucket),
                    size=(1,),
                    generator=generator,
                ).item()
            )

            yield bucket[utterance_index]
