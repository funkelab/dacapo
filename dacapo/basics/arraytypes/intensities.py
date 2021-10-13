from .helpers import ArrayType

import attr


@attr.s
class Intensities(ArrayType):

    shift: float = attr.ib()
    scale: float = attr.ib()
