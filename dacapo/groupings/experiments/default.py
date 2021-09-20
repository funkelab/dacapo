from .helpers import Experiment
from dacapo.fundamentals.datasplits import DataSplit
from dacapo.fundamentals.architectures import Architecture
from dacapo.fundamentals.outputs import Output
from dacapo.fundamentals.dataproviders import DataProvider
from dacapo.fundamentals.trainers import Trainer
from dacapo.fundamentals.validators import Validator
from dacapo.fundamentals.optimizers import Optimizer
from dacapo.store import ConfigStore, StatsStore, weights_store
from dacapo.categoricals.datakeys import ArrayKey
from dacapo.groupings.runs import Run, DefaultRun

import attr

from typing import Optional
from pathlib import Path


@attr.s
class DefaultExperiment(Experiment):

    datasplit: DataSplit = attr.ib(
        metadata={"help_text": "The data split for training and validation"}
    )
    architecture: Architecture = attr.ib(
        metadata={"help_text": "The architecture for your model 'backbone'"}
    )
    output: Output = attr.ib(
        metadata={
            "help_text": "The configuration for your model 'heads' and how to handle them"
        }
    )
    optimizer: Optimizer = attr.ib(
        metadata={"help_text": "The optimizer to use for training your model"}
    )
    dataprovider: DataProvider = attr.ib(
        metadata={"help_text": "The dataprovider to use for moving data around"}
    )
    trainer: Trainer = attr.ib(metadata={"help_text": "Training details go here"})
    validator: Validator = attr.ib(metadata={"help_text": "Validation details go here"})
    config_store: ConfigStore = attr.ib(
        metadata={"help_text": "Where to store your configs"}
    )
    stats_store: StatsStore = attr.ib(
        metadata={"help_text": "Where to store your stats"}
    )

    @property
    def root_dir(self):
        return Path(f"experiments/{self.name}")

    def is_valid(self):
        return True
        return all[
            self.can_train(),
            self.can_validate(),
            self.can_apply(),
            self.can_post_process(),
        ]

    def can_train(self):
        assert self.datasplit.train.provides(
            ArrayKey.RAW
        ), f"Datasplit does not provide raw training data"
        assert self.datasplit.train.raw.provides == self.architecture.takes, (
            f"Model expects input: {self.architecture.takes} but the raw data "
            f"provides {self.datasplit.train.raw.provides}"
        )
        assert self.architecture.provides == self.output.takes, (
            f"Model heads expect input: {self.output.takes}, "
            f"but the backbone provides: {self.architecture.provides}"
        )
        self.output.can_train()

    def can_validate(self):
        raise NotImplementedError()

    def can_apply(self):
        raise NotImplementedError()

    def can_post_process(self):
        raise NotImplementedError()

    def run(self, repitition: Optional[int] = None) -> Run:
        return DefaultRun(
            experiment_name=self.name,
            repitition=repitition,
            root_dir=self.root_dir / f"runs/{repitition}",
            stats_store=self.stats_store,
            trainer=self.trainer,
            validator=self.validator,
            train_provider=self.dataprovider,
            validation_provider=self.dataprovider,
            datasplit=self.datasplit,
            architecture=self.architecture,
            output=self.output,
            optimizer=self.optimizer,
        )
