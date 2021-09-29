from .helpers import Optimizer

import attr


@attr.s
class NullOptimizer(Optimizer):
    """
    This optimizer returns None. Used for debugging and in cases
    where you just want to post process or validate data rather
    than train.
    """

    name: str = attr.ib("null_optimizer")

    def optim(self, params):
        return None
