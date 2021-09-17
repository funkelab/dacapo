from dacapo.fundamentals.arraysources import ArraySource
from dacapo.fundamentals.graphsources import GraphSource
from dacapo.categoricals.datakeys import DataKey

import attr

from abc import ABC, abstractmethod
from typing import Union


@attr.s
class Dataset(ABC):
    name: str = attr.ib(
        metadata={"help_text": "Name of your dataset for easy search and reuse"}
    )

    @abstractmethod
    def provides(self, key: DataKey) -> bool:
        """
        Does this dataset provide a source for the given `key`?
        """
        pass
