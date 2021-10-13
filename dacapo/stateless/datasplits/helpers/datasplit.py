from dacapo.stateless.datasets import Dataset


import attr

from abc import ABC, abstractmethod
from typing import Union


@attr.s
class DataSplit(ABC):
    name: str = attr.ib(
        metadata={"help_text": "Name of your dataset for easy search and reuse"}
    )
