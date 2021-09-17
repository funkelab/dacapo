from .helpers import Pipeline
from dacapo.fundamentals.augments import Augment

import attr

from typing import List


@attr.s
class Default(Pipeline):
    augments: List[Augment] = attr.ib(
        factory=lambda: list(),
        metadata={"help_text": "The augments you want to apply during training"},
    )
