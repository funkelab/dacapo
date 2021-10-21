from arraytype import ArrayType

import attr


@attr.s
class DistanceArray(ArrayType):
    num_channels: int = attr.ib()

    @property
    def interpolatable(self) -> bool:
        return True