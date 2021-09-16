from .bossdb_source import BossDBSource
from .zarr_source import ZarrSource
from .rasterize_source import RasterizeSource

from typing import Union

ArraySources = Union[ZarrSource, BossDBSource, RasterizeSource]
