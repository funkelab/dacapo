import attr

from torch.optim import Optimizer

from abc import ABC, abstractmethod


@attr.s
class Optimizer(ABC):
    name: str = attr.ib(
        metadata={"help_text": "Name of optimizer for easy search and reuse"}
    )
    batch_size: int = attr.ib(default=2)

    @abstractmethod
    def optim(self, params) -> Optimizer:
        """
        Returns the torch optimizer ready to optimize the given parameters
        """
