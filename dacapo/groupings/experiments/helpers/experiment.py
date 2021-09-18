import attr

from typing import Optional
from abc import ABC, abstractmethod
from pathlib import Path


@attr.s
class Experiment(ABC):
    name: Optional[str] = attr.ib(
        metadata={"help_text": "Name of your experiment for easy search and reuse"},
    )

    @property
    @abstractmethod
    def root_dir(self) -> Path:
        pass

    @abstractmethod
    def is_valid(self):
        """
        Check whether all components of this experiment should work together.
        """
        pass

    @abstractmethod
    def can_train(self):
        """
        Check whether this experiment has the necessary components and
        compatibilities to train.
        """
        pass

    @abstractmethod
    def can_validate(self):
        """
        Check whether this experiment has the necessary components and
        compatibilities to validate.
        """
        pass

    @abstractmethod
    def can_apply(self):
        """
        Check whether this experiment has the necessary components and
        compatibilities to apply to new datasets.
        """
        pass

    @abstractmethod
    def can_post_process(self):
        """
        Check whether this experiment has the necessary components and
        compatibilities to apply to post process.
        """
        pass
