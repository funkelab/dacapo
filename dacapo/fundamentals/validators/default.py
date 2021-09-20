from .helpers import Validator

from dacapo.fundamentals.training_stats import TrainingStats

import attr


@attr.s
class DefaultValidator(Validator):

    validation_interval: int = attr.ib()

    def validate(self, training_stats: TrainingStats):
        return (training_stats.trained_until + 1) % self.validation_interval == 0
