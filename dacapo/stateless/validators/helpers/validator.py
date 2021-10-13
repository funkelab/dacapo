import attr

from abc import ABC, abstractmethod


@attr.s
class Validator(ABC):
    name: str = attr.ib(
        metadata={"help_text": "Name of your validator for easy search and reuse"}
    )

    @abstractmethod
    def validate_next(training_stats, validation_scores) -> bool:
        """
        Whether or not to validate now.
        Takes in the existing training_stats.
        """
        pass
