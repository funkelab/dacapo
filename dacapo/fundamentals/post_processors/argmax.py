from .helpers import PostProcessor

import attr


@attr.s
class ArgMax(PostProcessor):
    outputs: str = attr.ib(
        default="labels", metadata={"help_text": "The name of the provided outputs."}
    )
    # steps: Tuple[ArgMaxStep] = attr.ib()

    def tasks(
        self,
        pred_id: str,
        container,
        input_dataset,
        output_dataset,
    ):
        tasks, parameters = ArgMaxStep().tasks(
            pred_id,
            container,
            input_dataset,
            output_dataset,
        )

        return tasks, parameters
