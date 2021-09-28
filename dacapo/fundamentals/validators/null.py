from .helpers import Validator

import attr


@attr.s
class Null(Validator):
    name: str = attr.ib("null")

    def validate_next(self, training_stats, validation_scores):
        return False
