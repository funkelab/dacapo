from .helpers import Dataset
from dacapo.fundamentals.arraysources import ArraySource
from dacapo.categoricals.datakeys import ArrayKey

from gunpowder.array import ArrayKey as GPArrayKey
from gunpowder.nodes import MergeProvider, RandomLocation

import attr

from typing import Optional


@attr.s
class DefaultDataset(Dataset):
    raw: ArraySource = attr.ib(
        metadata={"help_text": "The source from which to fetch the raw array"}
    )

    gt: ArraySource = attr.ib(
        metadata={"help_text": "The source from which to fetch the ground truth array"}
    )

    mask: Optional[ArraySource] = attr.ib(
        default=None,
        metadata={"help_text": "The source from which to fetch the mask array"},
    )

    def provides(self, key):
        if key in [ArrayKey.RAW, ArrayKey.GT]:
            return True
        elif key == ArrayKey.MASK:
            return self.mask is not None
        else:
            return False

    def random_location_provider(self, raw: GPArrayKey, gt: GPArrayKey, mask: GPArrayKey = None):
        raw_source = self.raw.node(raw)
        gt_source = self.gt.node(gt)
        if self.mask is not None:
            mask_source = self.mask.node(mask)
            pipeline = (raw_source, gt_source, mask_source) + MergeProvider()
        else:
            pipeline = (raw_source, gt_source) + MergeProvider()

        pipeline += RandomLocation()
        return pipeline
