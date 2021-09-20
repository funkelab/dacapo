from .helpers import DataSplit
from dacapo.fundamentals.datasets import Dataset

import attr


@attr.s
class DefaultDataSplit(DataSplit):
    train: Dataset = attr.ib(metadata={"help_text": "The dataset to use for training"})
    validate: Dataset = attr.ib(
        metadata={"help_text": "The dataset to use for validation"}
    )
