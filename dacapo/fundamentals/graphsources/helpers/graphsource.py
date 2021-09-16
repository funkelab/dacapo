from gunpowder.graph import GraphKey

import attr

from abc import ABC, abstractmethod
from typing import Optional, List

@attr.s
class GraphSource(ABC):
    """
    A class representing a graph data source.
    """

    name: str = attr.ib()

    axes: Optional[List[str]] = attr.ib()

    @abstractmethod
    def node(self, graph: GraphKey):
        """
        Provide a gunpowder node that provides the desired data into
        the provided graph key.
        """
        pass
