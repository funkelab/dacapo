from .helpers import PostProcessor
from dacapo.fundamentals.processing_steps import ArgMaxStep
from dacapo.fundamentals.arraysources import ArraySource
from dacapo.basics import Parameters

import daisy

import attr

from typing import Tuple, List


@attr.s
class ArgMax(PostProcessor):
    outputs: str = attr.ib(
        default="labels", metadata={"help_text": "The name of the provided outputs."}
    )

    def tasks(
        self,
        pred_id: str,
        in_source: ArraySource,
        prediction_task: daisy.Task,
    ) -> Tuple[daisy.Task, Parameters, ArraySource]:
        parameters = Parameters(0)
        tasks, parameter_sets, output_sources = ArgMaxStep("labels", None, 2).tasks(
            pred_id, upstream_tasks=[[prediction_task], [parameters], [in_source]]
        )

        return tasks, parameter_sets, output_sources
