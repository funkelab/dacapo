from .intensity_augment import IntensityAugment
from .simple_augment import SimpleAugment

from typing import Union

Augments = Union[SimpleAugment, IntensityAugment]