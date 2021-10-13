from .helpers import ArraySource

import gunpowder as gp
from funlib.geometry import Coordinate
import daisy

import zarr

import logging
from typing import List, Optional
from pathlib import Path
import attr


@attr.s
class ZarrSource(ArraySource):
    """
    The class representing a single zarr dataset
    """

    filename: Path = attr.ib()
    ds_name: str = attr.ib()

    @property
    def axes(self) -> List[str]:
        return self.attributes["axes"]

    @property
    def voxel_size(self) -> Coordinate:
        return Coordinate(self.attributes["resolution"])

    @property
    def offset(self) -> Coordinate:
        return Coordinate(self.attributes["offset"])

    @property
    def shape(self) -> Coordinate:
        if "c" not in self.axes:
            return Coordinate(self.h5py_like.shape) * self.voxel_size
        else:
            phys_shape = [
                s for s, name in zip(self.h5py_like.shape, self.axes) if name != "c"
            ]
            return Coordinate(phys_shape) * self.voxel_size

    @property
    def num_channels(self) -> Coordinate:
        if "c" not in self.axes:
            return 0
        else:
            channel_axis = self.axes.index("c")
            return self.shape[channel_axis]

    @property
    def interpolatable(self) -> bool:
        self.attributes["interpolatable"]

    @property
    def h5py_like(self):
        zarr_container = zarr.open(str(self.filename))
        ds = zarr_container[self.ds_name]
        return ds

    @property
    def daisy_array(self):
        return daisy.open_ds(f"{self.filename}", self.ds_name, mode="r+")

    @property
    def attributes(self):
        return self.h5py_like.attrs

    def node(self, array):
        spec = gp.ArraySpec(
            roi=self.roi, voxel_size=self.voxel_size, interpolatable=self.interpolatable
        )
        return gp.ZarrSource(
            str(self.filename), {array: self.ds_name}, array_specs={array: spec}
        )
