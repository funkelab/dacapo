from .helpers import ArrayType

import attr


@attr.s
class Annotations(ArrayType):

    num_classes: int = attr.ib()
