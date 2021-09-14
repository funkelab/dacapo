from abc import ABC, abstractmethod


class LossABC(ABC):
    @abstractmethod
    def module(self):
        pass