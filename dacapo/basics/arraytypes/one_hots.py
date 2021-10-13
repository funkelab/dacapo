from .helpers import ArrayType

import attr


@attr.s
class OneHotArray(ArrayType):
    num_channels: int = attr.ib()
