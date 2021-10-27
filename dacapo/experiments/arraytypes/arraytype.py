import gunpowder as gp

import numpy as np

import attr

from abc import ABC, abstractmethod
from typing import List, Optional



# TODO: Should be read only
class ArrayType(ABC):
    """
    The type of data provided by an array. This class is in charge of not only
    making available the semantic meaning of the raw data (i.e. intensities,
    annotations, distance to boundary, etc.), but also handles transforming
    the provided data into a standard format for dacapo.
    """

    @property
    @abstractmethod
    def interpolatable(self) -> bool:
        pass