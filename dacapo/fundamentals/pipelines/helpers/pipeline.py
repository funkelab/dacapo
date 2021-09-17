import attr

from abc import ABC, abstractmethod


@attr.s
class Pipeline(ABC):
    name: str = attr.ib()
