from dacapo.store import StatsStore, WeightsStore
from dacapo.fundamentals.trainers import Trainer
from dacapo.fundamentals.validators import Validator
from dacapo.fundamentals.dataproviders import DataProvider
from dacapo.fundamentals.datasplits import DataSplit
from dacapo.fundamentals.architectures import Architecture
from dacapo.fundamentals.outputs import Output
from dacapo.fundamentals.optimizers import Optimizer

import attr

from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod


@attr.s
class Validation(ABC):
    experiment_name: str = attr.ib(
        metadata={"help_text": "The experiment to which this run belongs"},
    )
    repitition: int = attr.ib(
        metadata={"help_text": "The repitition for your run"},
    )
    root_dir: Path = attr.ib()
    stats_store: StatsStore = attr.ib()

    trainer: Trainer = attr.ib()
    validator: Validator = attr.ib()
    train_provider: DataProvider = attr.ib()
    validation_provider: DataProvider = attr.ib()
    datasplit: DataSplit = attr.ib()
    architecture: Architecture = attr.ib()
    output: Output = attr.ib()
    optimizer: Optimizer = attr.ib()