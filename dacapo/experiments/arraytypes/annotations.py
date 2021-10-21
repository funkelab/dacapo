from arraytype import ArrayType

import attr
from typing import Optional, List


@attr.s
class Annotations(ArrayType):

    num_classes: int = attr.ib()

    @property
    def interpolatable(self):
        return False
