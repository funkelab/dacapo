from abc import ABC, abstractmethod

import attr


@attr.s
class Loss(ABC):

    name: str = attr.ib()

    @abstractmethod
    def module(self):
        pass
