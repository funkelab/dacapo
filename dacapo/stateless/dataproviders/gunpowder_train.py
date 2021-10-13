from .helpers import DataProvider
from dacapo.stateless.augments import Augment
from dacapo.categoricals.datakeys import ArrayKey
from dacapo.gp import AddChannelDim

import gunpowder as gp

import attr

from typing import List


@attr.s
class GunpowderTrain(DataProvider):
    augments: List[Augment] = attr.ib(
        factory=lambda: list(),
        metadata={"help_text": "The augments you want to apply during training"},
    )

    # internal variables
    _gp_pipeline = None
    _gp_request = None
    _has_next = True

    @property
    def gp_pipeline(self):
        if self._gp_pipeline is None:
            raise ValueError("Gunpowder pipeline has not been initialized yet!")
        else:
            return self._gp_pipeline

    @property
    def gp_request(self):
        if self._gp_request is None:
            raise ValueError("Gunpowder request has not been initialized yet!")
        else:
            return self._gp_request

    def init_provider(self, dataset, architecture, output, trainer):

        raw_channels = max(1, dataset.raw.num_channels)
        input_shape = architecture.input_shape
        output_shape = architecture.output_shape
        voxel_size = dataset.raw.voxel_size

        # switch to world units
        input_size = voxel_size * input_shape
        output_size = voxel_size * output_shape

        # keys for provided datasets
        raw = gp.ArrayKey("RAW")
        gt = gp.ArrayKey("GT")
        if dataset.provides(ArrayKey.MASK):
            mask = gp.ArrayKey("MASK")
        else:
            mask = None

        snapshot_dataset_names = {
            raw: "raw",
        }

        predictor_keys = {}
        for predictor in output.predictors:
            name = predictor.name
            predictor_keys[name] = (
                gp.ArrayKey(f"{name.upper()}_TARGET"),
                gp.ArrayKey(f"{name.upper()}_WEIGHT"),
            )

            predictor_target, _ = predictor_keys[name]

            snapshot_dataset_names[predictor_target] = f"{name}_target"

        channel_dims = 0 if raw_channels == 1 else 1

        # Get source nodes
        gp_pipeline = dataset.random_location_provider(raw, gt)

        for augmentation in self.augments:
            gp_pipeline += augmentation.node(raw)

        # Add predictor nodes to gp_pipeline
        for predictor in output.predictors:
            name = predictor.name
            predictor_target, predictor_weights = predictor_keys[name]
            predictor_target_node, predictor_weights_node = predictor.add_target(
                gt, predictor_target, predictor_weights, mask
            )
            gp_pipeline += predictor_target_node
            if predictor_weights_node is not None:
                if predictor_weights_node is not True:
                    gp_pipeline += predictor_weights_node
                else:
                    # if weights are provided, but not by a new node.
                    # should maybe just return a no-op node
                    pass
                snapshot_dataset_names[predictor_weights] = f"{name}_weights"
            else:
                predictor_keys[name] = (predictor_target, None)

        # if there is no channel dimension, add one
        if channel_dims == 0:
            gp_pipeline += AddChannelDim(raw)

        gp_pipeline += gp.PreCache(num_workers=5)

        # stack to create a batch dimension
        gp_pipeline += gp.Stack(trainer.batch_size)

        # generate gp_request for all necessary inputs to training
        gp_request = gp.BatchRequest()
        gp_request.add(raw, input_size)
        gp_request.add(gt, output_size)
        for predictor in output.predictors:
            name = predictor.name
            predictor_target, predictor_weight = predictor_keys[name]
            gp_request.add(predictor_target, output_size)
            if predictor_weight is not None:
                gp_request.add(predictor_weight, output_size)

        self._gp_pipeline = gp_pipeline
        self._gp_request = gp_request
        self._predictor_keys = predictor_keys
        self.generator = self.generate()
        next(self.generator)

    def generate(self):
        with gp.build(self.gp_pipeline):
            self._has_next = True
            while self.has_next():
                batch = self.gp_pipeline.request_batch(self.gp_request)
                raw = batch[gp.ArrayKeys.RAW].data
                targets = {
                    name: batch[target_key].data
                    for name, (target_key, _) in self._predictor_keys.items()
                }
                weights = {
                    name: batch[weight_key].data
                    for name, (_, weight_key) in self._predictor_keys.items()
                    if weight_key is not None
                }
                self._has_next = yield {
                    "raw": raw,
                    "targets": targets,
                    "weights": weights,
                }
        yield None

    def next(self, done=False):
        return self.generator.send(not done)

    def has_next(self):
        return self._has_next
