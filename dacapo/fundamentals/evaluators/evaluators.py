from .affinity_evaluator import AffinityEvaluator
from .detection_evaluator import DetectionEvaluator
from .label_evaluator import LabelEvaluator

from typing import Union

Evaluators = Union[AffinityEvaluator, DetectionEvaluator, LabelEvaluator]
