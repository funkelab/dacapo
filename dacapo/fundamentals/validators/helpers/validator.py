import attr

from abc import ABC, abstractmethod


@attr.s
class Validator(ABC):
    name: str = attr.ib(
        metadata={"help_text": "Name of your validator for easy search and reuse"}
    )
