from dacapo.fundamentals.outputs import DefaultOutput
from ..predictors import simple_one_hot
from ..losses import simple_mse
from ..postprocessors import simple_argmax
from ..evaluators import simple_label_evaluator

simple_default = DefaultOutput(
    name="simple_default",
    outputs=[
        (
            simple_one_hot,
            simple_mse,
            simple_argmax,
            simple_label_evaluator,
        )
    ],
)
