from abc import ABC, abstractmethod

import attr


@attr.s
class Augment(ABC):
    """
    The Augment is a component that represents a parameterization of an augmentation
    you want to apply to your data during training. Some examples might include
    adding noise, mirroring, transposing, etc.

    The Augment must always be able to provide a gunpowder node that will apply
    the desired operation.
    """

    name = attr.ib(
        metadata={"help_text": "Name of your augment for easy search and reuse"}
    )

    @abstractmethod
    def node(self, array):
        # get the gunpowder node that performs this augment
        # takes an array. Currently only supports RAW
        pass
