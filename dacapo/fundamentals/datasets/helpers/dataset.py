import attr

from abc import ABC, abstractmethod


@attr.s
class Dataset(ABC):
    name: str = attr.ib()
