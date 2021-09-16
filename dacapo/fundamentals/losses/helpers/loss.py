from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def module(self):
        pass