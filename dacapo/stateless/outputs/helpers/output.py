import attr

from abc import ABC, abstractmethod


@attr.s
class Output(ABC):
    name: str = attr.ib()
