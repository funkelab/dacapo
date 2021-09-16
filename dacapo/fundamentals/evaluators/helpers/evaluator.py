from abc import ABC, abstractmethod

import attr


@attr.s
class Evaluator(ABC):
    """
    The Evaluator takes the postprocessed output of your model and evaluates
    its performace.
    """

    name = attr.ib(
        metadata={"help_text": "Name of your evaluator for easy search and reuse"}
    )

