import attr

from typing import Optional
from abc import ABC, abstractmethod


@attr.s
class Run(ABC):
    name: Optional[str] = attr.ib(
        metadata={"help_text": "Name of your validator for easy search and reuse"},
    )
