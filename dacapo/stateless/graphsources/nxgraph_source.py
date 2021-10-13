from .helpers import GraphSource
from dacapo.gp import NXSource

import attr

from typing import Optional, List
from pathlib import Path


@attr.s
class NXGraphSource(GraphSource):
    """
    The class representing a csv data source
    """

    filename: Path = attr.ib()
    axes: Optional[List[str]] = attr.ib(default=None)

    def node(self, graph):
        return NXSource(graph, self.filename)
