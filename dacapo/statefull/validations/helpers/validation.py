from dacapo.store import StatsStore, WeightsStore
from dacapo.stateless.dataproviders import DataProvider
from dacapo.stateless.datasets import Dataset
from dacapo.stateless.architectures import Architecture
from dacapo.stateless.outputs import Output
from dacapo.stateless.optimizers import Optimizer

import attr

from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod


@attr.s
class Validation(ABC):
    experiment_name: str = attr.ib(
        metadata={"help_text": "The experiment to which this run belongs"},
    )
    root_dir: Path = attr.ib()
    experiment_name: str = attr.ib()
    repitition: int = attr.ib()
    iteration: int = attr.ib()
    stats_store: StatsStore = attr.ib()
    checkpoint: Path = attr.ib()
    dataset: Dataset = attr.ib()
    architecture: Architecture = attr.ib()
    output: Output = attr.ib()