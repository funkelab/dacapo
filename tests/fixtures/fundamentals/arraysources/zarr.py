from dacapo.fundamentals.arraysources import ZarrSource

import zarr
import numpy as np

from pathlib import Path

zarr_source = ZarrSource(
    name="test_zarr",
    filename=Path("test.zarr"),
    ds_name="volumes/raw",
)


def mk_train_raw(path):
    noise = np.random.randn(100, 100, 100)
    label_0 = np.zeros_like(noise)
    noise_0 = np.random.randn(100, 100, 100)
    label_1 = np.zeros_like(noise)
    noise_1 = np.random.randn(100, 100, 100)
    label_0[:50, :, :] = 1
    label_1[50:, :, :] = 1
    data = noise + label_0 * noise_0 * 0.5 + label_1 * noise_1 * 2

    container_path = path / "data.zarr"
    if not container_path.exists():
        zarr_container = zarr.open(str(container_path.resolve()), mode="w")
    else:
        zarr_container = zarr.open(str(container_path.resolve()), mode="r+")
    try:
        dataset = zarr_container.create_dataset(
            "volumes/train/raw", data=data, shape=[100, 100, 100], dtype=np.float64
        )
        dataset.attrs["axes"] = ["z", "y", "x"]
        dataset.attrs["resolution"] = (1, 1, 1)
        dataset.attrs["offset"] = (0, 0, 0)
        dataset.attrs["interpolatable"] = True
    except Exception as e:
        raise e
        dataset = zarr_container["volumes/train/raw"]
        dataset[:, :, :] = data

    return ZarrSource(
        name="train_raw_zarr",
        filename=path / "data.zarr",
        ds_name="volumes/train/raw",
    )


def mk_train_gt(path):
    noise = np.zeros((100, 100, 100), dtype=np.uint64)
    label_0 = np.zeros_like(noise)
    label_1 = np.zeros_like(noise)
    label_0[:50, :, :] = 1
    label_1[50:, :, :] = 1
    data = noise + label_1

    container_path = path / "data.zarr"
    if not container_path.exists():
        zarr_container = zarr.open(str((path / "data.zarr").resolve()), mode="w")
    else:
        zarr_container = zarr.open(str((path / "data.zarr").resolve()), mode="r+")
    try:
        dataset = zarr_container.create_dataset(
            "volumes/train/gt", data=data, shape=[100, 100, 100], dtype=np.uint64
        )
        dataset.attrs["axes"] = ["z", "y", "x"]
        dataset.attrs["resolution"] = (1, 1, 1)
        dataset.attrs["offset"] = (0, 0, 0)
        dataset.attrs["interpolatable"] = False
    except Exception as e:
        raise e
        dataset = zarr_container["volumes/train/gt"]
        dataset[:, :, :] = data

    return ZarrSource(
        name="train_gt_zarr",
        filename=path / "data.zarr",
        ds_name="volumes/train/gt",
    )
