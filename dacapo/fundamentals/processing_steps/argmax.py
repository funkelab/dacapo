# from dacapo.store import MongoDbStore
from .helpers import ProcessingStep
from dacapo.fundamentals.arraysources import ArraySource, ZarrSource
from dacapo.basics.arraytypes import Annotations, OneHotArray
from dacapo.basics import Parameters

import attr
import numpy as np
import daisy

import time
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


@attr.s
class ArgMaxStep(ProcessingStep):
    step_id: str = attr.ib(default="argmax")

    # blockwise_processing_parameters
    write_shape: Optional[List[int]] = attr.ib(default=None)
    num_workers: int = attr.ib(default=2)

    def tasks(
        self,
        pred_id: str,
        upstream_tasks: Tuple[
            List[daisy.Task], List[Parameters], List[ArraySource]
        ] = None,
    ):
        """
        pred_id: The id of the prediction that you are postprocessing
        upstream_tasks: Contains a list of tasks, parameters and array sources.
            For each triplet (`task`, `parameters`, `arraysource`), the `parameters`
            should define all the parameters used so far in `task` and all of
            `task`'s upstream tasks. task should be the `task` that populates
            the `arraysource` which should be empty until `task` is complete.
        """

        tasks, task_parameters, out_sources = [], [], []

        if upstream_tasks is None:
            upstream_tasks = [None, {}]

        for upstream_task, upstream_parameters, in_source in zip(*upstream_tasks):
            # no parameters to explore for argmaxing so simply use upstream parameters
            parameters = upstream_parameters

            # input dataset in the form of a daisy array for convenient roi indexing
            assert isinstance(
                in_source.array_type, OneHotArray
            ), f"got array of symantic type: {in_source.array_type}, but expected one hot array"
            num_classes = in_source.array_type.num_channels
            probs = in_source.daisy_array

            # input_roi defined by provided dataset
            # TODO: allow for subrois?
            input_roi = probs.roi

            # get write_shape
            if self.write_shape is None:
                # default to input array chunk size
                write_shape = (
                    daisy.Coordinate(probs.chunk_shape[-probs.voxel_size.dims :])
                    * probs.voxel_size
                )
            else:
                write_shape = self.write_shape
            # define write_roi based on write_shape
            write_roi = daisy.Roi((0,) * write_shape.dims, write_shape)

            # create output dataset
            daisy.prepare_ds(
                in_source.filename,
                f"{in_source.ds_name}_{self.step_id}",
                total_roi=probs.roi,
                voxel_size=probs.voxel_size,
                dtype=np.uint32,
                write_size=write_shape,
                num_channels=1,
            )
            out_source = ZarrSource(
                name=f"{in_source.name}_{self.step_id}",
                filename=in_source.filename,
                ds_name=f"{in_source.ds_name}_{self.step_id}",
                array_type=Annotations(num_classes=num_classes),
            )
            labels = out_source.daisy_array

            t = daisy.Task(
                task_id=f"{pred_id}_{self.step_id}",
                total_roi=input_roi,
                read_roi=write_roi,
                write_roi=write_roi,
                process_function=self.get_process_function(pred_id, probs, labels),
                # check_function=self.get_check_function(pred_id),
                num_workers=self.num_workers,
                fit="shrink",
                upstream_tasks=[upstream_task],
            )
            tasks.append(t)
            task_parameters.append(parameters)
            out_sources.append(out_source)

        return tasks, task_parameters, out_sources

    def get_process_function(self, pred_id, probs, labels):
        def argmax_block(b):
            probabilities = probs.to_ndarray(b.write_roi)

            predicted_labels = np.argmax(probabilities, axis=0)

            labels[b.write_roi] = predicted_labels

        return argmax_block
