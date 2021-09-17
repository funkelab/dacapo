import attr

from typing import Optional
from abc import ABC, abstractmethod


@attr.s
class Experiment(ABC):
    name: Optional[str] = attr.ib(
        metadata={"help_text": "Name of your experiment for easy search and reuse"},
    )
