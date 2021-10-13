from .helpers import Output
from dacapo.stateless.predictors import Predictor
from dacapo.stateless.losses import Loss
from dacapo.stateless.post_processors import PostProcessor
from dacapo.stateless.evaluators import Evaluator

import attr

from typing import List, Tuple


@attr.s
class DefaultOutput(Output):

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

    @property
    def losses(self):
        return (o[1] for o in self.outputs)

    @property
    def post_processors(self):
        return (o[2] for o in self.outputs)

    @property
    def evaluators(self):
        return (o[3] for o in self.outputs)

    def is_better(self, new_validation_scores, current_best):
        return True
