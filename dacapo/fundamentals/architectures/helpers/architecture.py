import attr

from abc import ABC, abstractmethod


@attr.s
class Architecture(ABC):
    """
    The Architecture defines the backbone of the model that will be trained.
    The output of the Architecture will probably be some intermediate soft value
    that must be post processed to generate final results.

    An Architecture in dacapo must have a name, and be able to produce a module
    on demand.
    """
    name: str = attr.ib(
        metadata={"help_text": "Name of your model for easy search and reuse"}
    )

    @abstractmethod
    def module(self):
        pass
