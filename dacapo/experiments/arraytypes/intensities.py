from arraytype import ArrayType

from dacapo.gp import IntensityScaleShift, NoOp

import attr


@attr.s
class Intensities(ArrayType):

    min: float = attr.ib()
    max: float = attr.ib()

    @property
    def scale(self):
        return 2 / (self.max - self.min)

    @property
    def shift(self):
        return -1 - self.min * self.scale

    @property
    def interpolatable(self) -> bool:
        return True

    def node(self, in_key, out_key):
        return IntensityScaleShift(in_key, self.scale, self.shift) + NoOp(
            in_key, out_key
        )
