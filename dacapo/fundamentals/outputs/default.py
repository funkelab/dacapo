from .helpers import Output
from dacapo.fundamentals.predictors import Predictor
from dacapo.fundamentals.losses import Loss
from dacapo.fundamentals.post_processors import PostProcessor
from dacapo.fundamentals.evaluators import Evaluator

import attr

from typing import List, Tuple


@attr.s
class Default(Output):

    outputs: List[Tuple[Predictor, Loss, PostProcessor, Evaluator]] = attr.ib(
        metadata={
            "help_text": "Predictors, Losses, PostProcessors, and Evaluators together define what "
            "your model outputs and how it is evaluated"
        }
    )

    @property
    def num_outputs(self):
        return len(self.outputs)

    @property
    def predictors(self):
        return (o[0] for o in self.outputs)