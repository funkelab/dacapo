import attr

from abc import ABC, abstractmethod


@attr.s
class DataProvider(ABC):
    name: str = attr.ib()

    @abstractmethod
    def next(self):
        """
        Should retrieve a single sample that is ready for processing.
        """
        pass

    @abstractmethod
    def has_next(self):
        """
        Should let the user know if there is more data to process
        """
