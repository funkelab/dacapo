from dacapo.fundamentals.training_stats import TrainingStats
from dacapo.fundamentals.validation_scores import ValidationScores

import attr

from abc import ABC, abstractmethod


@attr.s
class Validator(ABC):
    name: str = attr.ib(
        metadata={"help_text": "Name of your validator for easy search and reuse"}
    )

    @abstractmethod
    def validate(training_stats: TrainingStats, validation_scores: ValidationScores) -> bool:
        """
        Whether or not to validate now.
        Takes in the existing training_stats.
        """
        pass
