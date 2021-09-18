from dacapo.store import StatsStore, WeightsStore
from dacapo.fundamentals.trainers import Trainer
from dacapo.fundamentals.validators import Validator
from dacapo.fundamentals.pipelines import Pipeline
from dacapo.fundamentals.datasplits import DataSplit
from dacapo.fundamentals.architectures import Architecture
from dacapo.fundamentals.outputs import Output

import attr

from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod


@attr.s
class Run(ABC):
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
    pipeline: Pipeline = attr.ib()
    datasplit: DataSplit = attr.ib()
    architecture: Architecture = attr.ib()
    output: Output = attr.ib()

    @property
    @abstractmethod
    def complete(self):
        pass

    @abstractmethod
    def retrieve_training_stats(self):
        pass

    @abstractmethod
    def retrieve_validation_scores(self):
        pass