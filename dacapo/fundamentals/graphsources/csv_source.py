from .helpers import GraphSource

import gunpowder as gp

import attr

from pathlib import Path
from typing import Optional, List


@attr.s
class CSVSource(GraphSource):
    """
    The class representing a csv data source
    """

    filename: Path = attr.ib()
    ndims: Optional[int] = attr.ib(default=None)
    id_dim: Optional[int] = attr.ib(default=None)
    axes: Optional[List[str]] = attr.ib(default=None)

    def node(self, graph):
        return gp.CsvPointsSource(
            self.filename, graph, id_dim=self.id_dim, ndims=self.ndims
        )
