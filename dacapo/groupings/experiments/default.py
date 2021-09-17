from .helpers import Experiment
from dacapo.fundamentals.datasplits import DataSplit
from dacapo.fundamentals.architectures import Architecture
from dacapo.fundamentals.outputs import Output
from dacapo.fundamentals.pipelines import Pipeline
from dacapo.fundamentals.trainers import Trainer
from dacapo.fundamentals.validators import Validator
from dacapo.fundamentals.executers import Executer
from dacapo.store import ConfigStore, StatsStore

import attr


@attr.s
class Default(Experiment):

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
    pipeline: Pipeline = attr.ib(
        metadata={"help_text": "The pipeline to use for moving data around"}
    )
    trainer: Trainer = attr.ib(metadata={"help_text": "Training details go here"})
    validator: Validator = attr.ib(metadata={"help_text": "Validation details go here"})
    executer: Executer = attr.ib(metadata={"help_text": "How to run your computations"})
    config_store: ConfigStore = attr.ib(
        metadata={"help_text": "Where to store your configs"}
    )
    stats_store: StatsStore = attr.ib(
        metadata={"help_text": "Where to store your stats"}
    )
