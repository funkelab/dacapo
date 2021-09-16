from .helpers import ArraySource

import gunpowder as gp
from funlib.geometry import Coordinate
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

    axes: Optional[List[str]] = attr.ib(default=None)
    voxel_size: Optional[Coordinate] = attr.ib(default=None)
    offset: Optional[Coordinate] = attr.ib(default=None)
    shape: Optional[Coordinate] = attr.ib(default=None)
    num_channels: Optional[int] = attr.ib(default=None)
    interpolatable: Optional[bool] = attr.ib(default=None)

    def update_metadata(self):
        zarr_container = zarr.open(str(self.filename))
        ds = zarr_container[self.ds_name]

        # axes
        try:
            provided_axes = ds.attrs["axes"]
        except KeyError:
            provided_axes = None
        if self.axes is None and provided_axes is None:
            raise ValueError("Axes must be defined in config or in bossdb metadata")
        elif provided_axes is not None:
            self.axes = provided_axes

        # voxel_size
        try:
            provided_voxel_size = Coordinate(ds.attrs["voxel_size"])
        except KeyError:
            provided_voxel_size = None
        if self.voxel_size is None and provided_voxel_size is None:
            raise ValueError(
                "Voxel size must be defined in config or in bossdb metadata"
            )
        elif provided_voxel_size is not None:
            self.voxel_size = provided_voxel_size

        # shape
        try:
            provided_shape = Coordinate(ds.attrs["shape"])
        except KeyError:
            provided_shape = None
        if self.shape is None and provided_shape is None:
            raise ValueError("Shape must be defined in config or in bossdb metadata")
        elif provided_shape is not None:
            self.shape = provided_shape

        # offset
        try:
            provided_offset = Coordinate(ds.attrs["offset"])
        except KeyError:
            provided_offset = None
        if self.offset is None and provided_offset is None:
            raise ValueError("Offset must be defined in config or in bossdb metadata")
        elif provided_offset is not None:
            self.offset = provided_offset

        # interpolatable
        try:
            provided_interpolatable = Coordinate(ds.attrs["interpolatable"])
        except KeyError:
            provided_interpolatable = None
        if self.interpolatable is None and provided_interpolatable is None:
            raise ValueError("Interpolatable must be defined in config or in bossdb metadata")
        elif provided_interpolatable is not None:
            self.interpolatable = provided_interpolatable

        # num_channels (1 if shape.dims == voxel_size.dims else shape[0])
        provided_num_channels = (
            1 if self.shape.dims == self.voxel_size.dims else self.shape[0]
        )
        if self.num_channels is None:
            self.num_channels = provided_num_channels
        else:
            assert (
                self.num_channels <= provided_num_channels
            ), f"Cannot use more channels than are provided"

    def node(self, array):
        spec = gp.ArraySpec(roi=self.roi, voxel_size=self.voxel_size)
        return gp.ZarrSource(
            str(self.filename), {array: self.ds_name}, array_specs={array: spec}
        )
