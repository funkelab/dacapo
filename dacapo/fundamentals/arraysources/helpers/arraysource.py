from dacapo.fundamentals.arraytypes import ArrayType

from funlib.geometry import Coordinate, Roi

import attr

from abc import ABC, abstractmethod
from typing import List, Optional


@attr.s
class ArraySource(ABC):
    """
    An ArraySource provides an interface for working with some set of data
    in the form of an array. It must be able to return a gunpowder node
    that provides the array under a given array key.

    Upon query ArraySources must be able to provide meta data about the array
    including the number of dimensions, semantic meaning of each axis, voxel_size
    and roi.

    Finally because many data sources such as Zarr, BossDB, N5 etc. have methods
    for storing meta data associated with them, you must also define functions
    for reading that metadata and verifying any user configurations match stored
    metadata.
    """

    name: str = attr.ib(
        metadata={"help_text": "Name of your ArraySource for easy search and reuse"}
    )
    array_type: ArrayType = attr.ib()

    @property
    @abstractmethod
    def axes(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def voxel_size(self) -> Coordinate:
        pass

    @property
    @abstractmethod
    def offset(self) -> Coordinate:
        pass

    @property
    @abstractmethod
    def shape(self) -> Coordinate:
        pass

    @property
    @abstractmethod
    def num_channels(self) -> int:
        pass

    @property
    @abstractmethod
    def interpolatable(self) -> bool:
        pass

    @property
    @abstractmethod
    def h5py_like(self):
        pass

    @property
    def dims(self):
        """Returns the number of spatial dimensions."""
        return self.voxel_size.dims

    @property
    def roi(self):
        """The total ROI of this array, in world units."""
        return Roi(self.offset, self.shape)

    @abstractmethod
    def node(self, key):
        """
        Return a gunpowder node that provides
        data from the data source into provided key
        """
        pass
