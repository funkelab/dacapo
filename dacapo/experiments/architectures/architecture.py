from funlib.geometry import Coordinate

from typing import Tuple
from abc import ABC, abstractmethod


class Architecture(ABC):
    @property
    @abstractmethod
    def input_shape(self):
        """The spatial input shape (i.e., not accounting for channels and batch
        dimensions) of this architecture."""
        pass

    @property
    @abstractmethod
    def num_in_channels(self):
        """Return the number of input channels this architecture expects."""
        pass

    @property
    @abstractmethod
    def num_out_channels(self):
        """Return the number of output channels of this architecture."""
        pass

    @abstractmethod
    def forward(self, x):
        """Process an input tensor."""
        pass

    def scale_factor(self) -> Tuple[bool, Coordinate]:
        """
        The factor by which the input voxel size will differ from the output
        voxel size.

        Returns: `Tuple[up: bool, factor: Coordinate]`:
            `up`: is a bool determining whether the output has a higher
            resolution that the input.
            `factor`: Determines the change in voxel size.

        For example, if you input is 8x8x8 nm, and your output is 4x4x4 nm,
        then you should return (true, Coordinate(2,2,2))
        If your input is 4x4x4 nm, and your output is 12x12x12 nm,
        then you should return (false, Coordinate(3,3,3))

        Default behavior is to assume no change in scale.
        """
        return (True, Coordinate((1,)*self.input_shape.dims))
