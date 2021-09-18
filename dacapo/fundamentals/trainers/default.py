from .helpers import Trainer

import attr


@attr.s
class Default(Trainer):
    num_iterations: int = attr.ib()
    batch_size: int = attr.ib()