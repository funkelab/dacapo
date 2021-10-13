from dacapo.gp.bossdb import BossDB
from .helpers.arraysource import ArraySource

from gunpowder.array_spec import ArraySpec
from funlib.geometry import Coordinate, Roi
import attr
from intern import array

from typing import List, Optional


@attr.s
class BossDBSource(ArraySource):
    """
    The class representing a single zarr dataset
    """

    db_name: str = attr.ib()

    @property
    def axes(self) -> List[str]:
        return self.h5py_like.axes

    @property
    def voxel_size(self) -> Coordinate:
        return Coordinate(self.h5py_like.resolution)

    @property
    def offset(self) -> Coordinate:
        return Coordinate(self.h5py_like.offset)

    @property
    def shape(self) -> Coordinate:
        return self.h5py_like.shape

    @property
    def size(self) -> Coordinate:
        return self.h5py_like.size

    @property
    def num_channels(self) -> Coordinate:
        if "c" not in self.axes:
            return 0
        else:
            channel_axis = self.axes.index("c")
            return self.shape[channel_axis]

    @property
    def interpolatable(self) -> Coordinate:
        self.h5py_like.interpolatable

    @property
    def h5py_like(self):
        return array(self.db_name)

    def node(self, array):
        spec = ArraySpec(
            roi=self.roi,
            voxel_size=self.voxel_size,
            interpolatable=self.interpolatable,
        )
        return BossDB(
            array,
            self.db_name,
            spec,
        )
