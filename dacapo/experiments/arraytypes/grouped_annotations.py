from arraytype import ArrayType

import numpy as np
import attr

from typing import List



@attr.s
class GroupedAnnotationArray(ArrayType):
    """
    The GroupedAnnotationArray converts a single channel annotated volume
    into a `k` channeled binary volume where each a voxel in channel `l` is
    1 iff that voxel in the original volume is contained in `groupings[l]`.
    """

    groupings: List[List[int]] = attr.ib()

    @property
    def num_classes(self) -> int:
        return len(self.groupings)

    @property
    def interpolatable(self) -> bool:
        return False
