from dacapo.basics.arraytypes import ArrayType

import torch
import attr

from abc import ABC, abstractmethod


@attr.s
class Predictor(ABC):
    """
    The `Predictor` is essentially a "head" on top of the "backbone"
    defined by the `Architecture`. This will ensure that the output
    of the network is transformed into the appropriate shape/range/dtype
    for your intended purposes.

    Since this is where we define what the output of the network is,
    this is also where we need to define how we transform the ground
    truth data so that we can apply a loss and backprop that through
    the network.
    """

    name: str = attr.ib(
        metadata={"help_text": "This name is used to differentiate between predictors."}
    )
    
    @property
    @abstractmethod
    def output_arraytype(self) -> ArrayType:
        pass

    @abstractmethod
    def head(self) -> torch.nn.Module:
        """
        Provide a torch module that can be plugged in on top of the
        architecture module. Thus we will have:
        output = head.forward(backbone.forward(input))
        """
        pass

    @abstractmethod
    def add_target(self):
        """
        Provide a gunpowder Batch filter that can transform the ground
        truth data into the appropriate target for the head.
        """
        pass
