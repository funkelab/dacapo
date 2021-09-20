import attr

from abc import ABC, abstractmethod
from typing import List, Optional


@attr.s
class ArrayType(ABC):
    """
    The type of data provided by your array. For example, your ground truth data might
    provide labels in uint64. In this case we would also need to know the number of
    classes.
    """

    name: str = attr.ib(
        metadata={"help_text": "Name of your ArrayType for easy search and reuse"}
    )