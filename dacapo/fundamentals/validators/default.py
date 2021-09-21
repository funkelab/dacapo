from .helpers import Validator

from dacapo.fundamentals.training_stats import TrainingStats
from dacapo.fundamentals.validation_scores import ValidationScores

import attr


@attr.s
class DefaultValidator(Validator):

    validation_interval: int = attr.ib()

    def validate(self, training_stats: TrainingStats, validation_scores: ValidationScores):
        return (training_stats.trained_until + 1) % self.validation_interval == 0
