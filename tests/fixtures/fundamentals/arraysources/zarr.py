from dacapo.fundamentals.arraysources import ZarrSource

from pathlib import Path

zarr_source = ZarrSource(
    name="test_zarr",
    filename=Path("test.zarr"),
    ds_name="volumes/raw",
    axes=["x", "y"],
    voxel_size=[1, 1],
    offset=[0, 0],
)
