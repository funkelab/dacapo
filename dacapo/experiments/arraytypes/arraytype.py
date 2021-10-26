import gunpowder as gp

import attr

from abc import ABC, abstractmethod
from typing import List, Optional


class NoOp(gp.BatchFilter):
    def __init__(self, in_array, out_array):
        self.in_array = in_array
        self.out_array = out_array

    def setup(self):
        spec = self.spec[self.in_array].copy()
        self.provides(self.out_array, spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.in_array] = request[self.out_array].copy()
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()

        outputs[self.out_array] = batch[self.in_array]
        return outputs



# TODO: Should be read only
class ArrayType(ABC):
    """
    The type of data provided by your array. For example, your ground truth data might
    provide labels in uint64. In this case we would also need to know the number of
    classes.
    """

    @property
    @abstractmethod
    def interpolatable(self) -> bool:
        pass

    def node(self, in_key, out_key) -> gp.BatchFilter:
        return NoOp(in_key, out_key)