from .helpers import Validator

import attr


@attr.s
class Null(Validator):
    name: str = attr.ib("null")

    def validate(self, training_stats, validation_stats):
        return False
