from .helpers import Validation
from dacapo.store.validation_iteration_scores import ValidationIterationScores
from dacapo.store import (
    ConfigStore,
    StatsStore,
    stats_store,
    LocalWeightsStore,
    DebugScoresStore,
)
from dacapo.fundamentals.arraysources import ZarrSource

import daisy

import attr
import torch
import numpy as np

from pathlib import Path
import logging
import time

logger = logging.getLogger(__file__)


@attr.s
class DefaultValidation(Validation):
    pass

    @property
    def validation_scores(self):
        if self._validation_scores is None:
            self.init_stats()
            return self._validation_scores
        else:
            return self._validation_scores

    def init_stats(self):
        # Read training stats from db
        self._validation_scores = self.retrieve_validation_scores()

    def retrieve_validation_scores(self):
        return self.stats_store.retrieve_validation_scores(
            self.experiment_name, self.repitition
        )

    def setup(self):
        pass

    def teardown(self):
        pass

    def run_blockwise(self):

        # info for uniquely identifying results
        experiment_name = self.experiment_name
        repitition = self.repitition
        iteration = self.iteration
        val_id = f"{experiment_name}_{repitition}_{iteration}"

        # PREDICTIONS
        # get model input/output shapes
        input_shape = self.architecture.input_shape
        output_shape = self.architecture.output_shape
        context = (input_shape - output_shape) / 2

        # create read/write rois
        write_roi = daisy.Roi(context, output_shape)
        read_roi = daisy.Roi((0,) * context.dims, input_shape)
        # get total write roi
        total_write_roi = self.dataset.gt.roi
        total_read_roi = total_write_roi.grow(context, context)

        # initialize prediction datasets
        predictions = {}
        prediction_sources = {}
        for predictor in self.output.predictors:
            predictions[predictor.name] = daisy.prepare_ds(
                f'{self.root_dir / "validations" / f"{self.iteration}.zarr"}',
                f"{predictor.name}",
                total_roi=total_write_roi,
                voxel_size=self.dataset.gt.voxel_size,
                dtype=np.float32,
                write_size=output_shape,
                num_channels=predictor.fmaps_out,
            )
            prediction_sources[predictor.name] = ZarrSource(
                name=f"{experiment_name}_{repitition}_{iteration}_valpred_{predictor.name}",
                filename=f'{self.root_dir / "validations" / f"{self.iteration}.zarr"}',
                ds_name=f"{predictor.name}",
                array_type=predictor.output_arraytype,
            )

        # function for predicting in block
        def predict_in_block(
            block, predictions, dataset, output, architecture, checkpoint
        ):
            raw = dataset.raw.daisy_array

            backbone = architecture.module()
            heads = [
                predictor.head(architecture, dataset) for predictor in output.predictors
            ]

            backbone = backbone.eval()
            heads = [head.eval() for head in heads]

            assert checkpoint.exists(), f"{checkpoint}"
            state_dict = torch.load(checkpoint)
            backbone.load_state_dict(state_dict["backbone"])
            for head, predictor in zip(heads, output.predictors):
                head.load_state_dict(state_dict[predictor.name])

            raw_input = raw.to_ndarray(roi=block.read_roi, fill_value=0)
            # create batch dim
            raw_input = np.expand_dims(raw_input, 0)

            latent_prediction = backbone.forward(torch.from_numpy(raw_input).float())
            for predictor, head in zip(output.predictors, heads):
                prediction_array = daisy.Array(
                    head(latent_prediction).detach().cpu().numpy()[0],
                    block.write_roi,
                    raw.voxel_size,
                )
                write_roi = block.write_roi.intersect(predictions[predictor.name].roi)
                predictions[predictor.name][write_roi] = prediction_array.to_ndarray(
                    roi=write_roi
                )

            print(f"Block {block} done!")

        # Define the prediction task
        predict_task = daisy.Task(
            f"{val_id}_pred",
            total_roi=total_read_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            process_function=lambda b: predict_in_block(
                b,
                predictions,
                self.dataset,
                self.output,
                self.architecture,
                self.checkpoint,
            ),
            fit="overhang",
            num_workers=1,
        )

        # Get the post_processing and evaluation tasks
        evaluation_tasks = []
        scores = DebugScoresStore(self.root_dir / "validations/eval_scores")
        for predictor, post_processor, evaluator in zip(
            self.output.predictors, self.output.post_processors, self.output.evaluators
        ):

            if evaluator is not None:
                pred_id = f"{val_id}_{predictor.name}"
                (
                    post_processor_tasks,
                    parameter_sets,
                    post_processed_sources,
                ) = post_processor.tasks(
                    pred_id=pred_id,
                    in_source=prediction_sources[predictor.name],
                    prediction_task=predict_task,
                )
                for post_processor_task, parameters, post_processed_source in zip(
                    post_processor_tasks, parameter_sets, post_processed_sources
                ):
                    evaluator_task = evaluator.task(
                        pred_id=pred_id,
                        post_processing_parameters=parameters,
                        pred_source=post_processed_source,
                        gt_source=self.dataset.gt,
                        scores_store=scores,
                        upstream_task=post_processor_task,
                    )

                    evaluation_tasks.append(
                        (evaluator, predictor, parameters, evaluator_task)
                    )

        daisy.run_blockwise([eval_task for _, _, _, eval_task in evaluation_tasks])

        evaluator_scores = []
        for evaluator, predictor, parameters, evaluation_task in evaluation_tasks:
            validation_scores = evaluator.aggregate_block_scores(
                scores.retrieve_scores(evaluation_task.task_id)
            )
            evaluator_scores.append((predictor, parameters, validation_scores))

        return ValidationIterationScores(
            iteration=iteration, parameter_scores=evaluator_scores
        )
