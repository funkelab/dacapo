from arraytype import ArrayType

import numpy as np

import attr


@attr.s
class IntensitiesArray(ArrayType):
    """
    An IntensitiesArray normalizes the Intensities to a standard
    range that DaCapo can rely on. i.e. if your intensities are
    stored as uint8, you can set the min and max to 0 and 255
    respectively and the intensities will be normalized between
    0 and 1.
    """

    min: float = attr.ib(
        metadata={"help_text": "The minimum possible value of your intensities."}
    )
    max: float = attr.ib(
        metadata={"help_text": "The maximum possible value of your intensities."}
    )

    @property
    def scale(self):
        return self.target_max / (self.max - self.min)

    @property
    def shift(self):
        return self.target_min - self.min * self.scale

    @property
    def interpolatable(self) -> bool:
        return True
