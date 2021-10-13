from .affinity_evaluators import simple_affinity_evaluator
from .detection_evaluators import simple_detection_evaluator
from .label_evaluators import simple_label_evaluator

EVALUATORS = [
    simple_affinity_evaluator,
    simple_detection_evaluator,
    simple_label_evaluator,
]
