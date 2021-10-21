from arraytype import ArrayType

import attr


@attr.s
class AffinityArray(ArrayType):

    @property
    def interpolatable(self) -> bool:
        return False