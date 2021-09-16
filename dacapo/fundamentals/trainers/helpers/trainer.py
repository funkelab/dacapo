import attr

from abc import ABC, abstractmethod


@attr.s
class Trainer(ABC):
    name: str = attr.ib(
        metadata={"help_text": "Name of your trainer for easy search and reuse"}
    )
    num_iterations: int = attr.ib()
