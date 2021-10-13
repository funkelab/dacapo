from .helpers import Experiment
from dacapo.fundamentals.datasplits import DataSplit
from dacapo.fundamentals.architectures import Architecture
from dacapo.fundamentals.outputs import Output
from dacapo.fundamentals.dataproviders import DataProvider
from dacapo.fundamentals.trainers import Trainer
from dacapo.fundamentals.validators import Validator
from dacapo.fundamentals.optimizers import Optimizer
from dacapo.fundamentals.executers import Executer
from dacapo.store import ConfigStore, StatsStore, weights_store
from dacapo.categoricals.datakeys import ArrayKey
from dacapo.groupings.runs import Run, DefaultRun
from dacapo.groupings.validations import Validation, DefaultValidation

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
    train_provider: DataProvider = attr.ib(
        metadata={"help_text": "The dataprovider to use for fetching training batches"}
    )
    trainer: Trainer = attr.ib(metadata={"help_text": "Training details go here"})
    validator: Validator = attr.ib(metadata={"help_text": "Validation details go here"})
    config_store: ConfigStore = attr.ib(
        metadata={"help_text": "Where to store your configs"}
    )
    stats_store: StatsStore = attr.ib(
        metadata={"help_text": "Where to store your stats"}
    )
    experiments_dir: Optional[Path] = attr.ib(
        metadata={"help_text": "Where to find your experiments"}, default=None
    )
    validation_executer: Optional[Executer] = attr.ib(
        metadata={"help_text": "How to execute your validations"}, default=None
    )

    @property
    def root_dir(self):
        if self.experiments_dir is not None:
            return self.experiments_dir / f"{self.name}"
        else:
            return Path(f"experiments/{self.name}")

    @property
    def num_repititions(self):
        if (self.root_dir / "runs").exists():
            return len(list((self.root_dir / "runs").iterdir()))
        else:
            return 0

    def is_valid(self):
        return True
        return all[
            self.can_train(),
            self.can_validate(),
            self.can_apply(),
            self.can_post_process(),
        ]

    def can_train(self):
        return True
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

    def can_validate(self, repitition=None, iteration=None):
        return True
        # check if validation is generally possible
        assert self.output.provides == self.post_processor.takes
        assert self.post_processor.provides == self.evaluator.takes
        # if validation and iteration are provided, check if necessary state exists:
        assert self.checkpoint_exists(repitition, iteration)

    def can_apply(self):
        raise NotImplementedError()

    def can_post_process(self):
        raise NotImplementedError()

    def run(self, repitition: Optional[int] = None) -> Run:
        repitition = repitition if repitition is not None else self.num_repititions
        return DefaultRun(
            experiment_name=self.name,
            repitition=repitition,
            root_dir=self.root_dir / f"runs/{repitition}",
            stats_store=self.stats_store,
            trainer=self.trainer,
            validator=self.validator,
            dataprovider=self.train_provider,
            dataset=self.datasplit.train,
            architecture=self.architecture,
            output=self.output,
            optimizer=self.optimizer,
        )

    def validate(self, repitition: int, iteration: int) -> Validation:
        run = self.run(repitition)
        checkpoint = run.checkpoint(iteration)
        return DefaultValidation(
            experiment_name=self.name,
            checkpoint=checkpoint,
            repitition=repitition,
            iteration=iteration,
            root_dir=self.root_dir / f"runs/{repitition}",
            stats_store=self.stats_store,
            dataset=self.datasplit.validate,
            architecture=self.architecture,
            output=self.output,
        )
