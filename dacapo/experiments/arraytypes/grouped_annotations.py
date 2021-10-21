from arraytype import ArrayType

import gunpowder as gp

import attr
from typing import Optional, List

import gunpowder as gp
import numpy as np


class GroupLabels(gp.BatchFilter):
    def __init__(self, in_array, out_array, groupings):
        self.in_array = in_array
        self.out_array = out_array
        self.groupings = groupings

    def setup(self):
        spec = self.spec[self.in_array].copy()
        self.provides(self.out_array, spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        spec = gp.ArraySpec(roi=request[self.out_array].roi)
        deps[self.in_array] = spec
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()

        if self.in_array not in batch:
            return

        data = batch[self.in_array].data
        spec = batch[self.in_array].spec.copy()

        assert (
            len(data.shape) == batch[self.in_array].spec.roi.dims
        ), f"{data.shape} vs {batch[self.in_array].spec.roi.dims}"
        grouped = np.zeros((len(self.groupings), *data.shape), dtype=data.dtype)
        for i, ids in enumerate(self.groupings):
            for id in ids:
                grouped[i] += data == id

        outputs[self.out_array] = gp.Array(grouped, spec)
        return outputs

@attr.s
class GroupedAnnotations(ArrayType):

    groupings: List[List[int]] = attr.ib()

    @property
    def num_classes(self) -> int:
        return len(self.groupings)

    @property
    def interpolatable(self):
        return False

    def node(self, in_key, out_key):
        return GroupLabels(in_key, out_key, self.groupings)
