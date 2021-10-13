from .helpers import Validator

import attr


@attr.s
class DefaultValidator(Validator):

    validation_interval: int = attr.ib()

    def validate_next(self, training_stats, validation_scores):
        return (training_stats.trained_until + 1) % self.validation_interval == 0
