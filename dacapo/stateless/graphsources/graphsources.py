from .csv_source import CSVSource
from .nxgraph_source import NXGraphSource

from typing import Union

GraphSources = Union[CSVSource, NXGraphSource]
