from .zarr_source import ZarrSource
from .bossdb import BossDBSource
from .rasterize_source import RasterizeSource
from .csv_source import CSVSource
from .nxgraph import NXGraphSource
from dacapo.converter import converter

from typing import Union

DataSources = Union[ZarrSource, BossDBSource, RasterizeSource, CSVSource, NXGraphSource]

converter.register_unstructure_hook(
    DataSources,
    lambda o: {"__type__": type(o).__name__, **converter.unstructure(o)},
)
converter.register_structure_hook(
    DataSources,
    lambda o, _: converter.structure(o, eval(o["__type__"])),
)
