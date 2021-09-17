from .helpers import Dataset
from dacapo.fundamentals.arraysources import ArraySource
from dacapo.categoricals.datakeys import ArrayKey

import attr

from typing import Optional


@attr.s
class Default(Dataset):
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
