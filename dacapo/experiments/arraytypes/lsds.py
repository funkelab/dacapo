from arraytype import ArrayType

import attr


@attr.s
class LSDArray(ArrayType):

    @property
    def interpolatable(self) -> bool:
        return True