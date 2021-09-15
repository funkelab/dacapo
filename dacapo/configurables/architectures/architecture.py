import attr

from dacapo.converter import converter

from abc import ABC, abstractmethod


@attr.s
class Architecture(ABC):
    name: str = attr.ib(
        metadata={"help_text": "Name of your model for easy search and reuse"}
    )

    def verify(self):
        unstructured = converter.unstructure(self)
        structured = converter.structure(unstructured, self.__class__)
        assert self == structured
        return True

    @abstractmethod
    def module(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass
