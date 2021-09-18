from .helpers import Validator

import attr


@attr.s
class Null(Validator):
    name: str = attr.ib("null")
    validation_interval: int = attr.ib(2 ** 32)
