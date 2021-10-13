import attr

from typing import Optional
from abc import ABC, abstractmethod


@attr.s
class Application(ABC):
    name: Optional[str] = attr.ib(
        metadata={"help_text": "Name of your validator for easy search and reuse"},
    )
