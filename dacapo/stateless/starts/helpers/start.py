from abc import ABC, abstractmethod

import attr


@attr.s
class Start(ABC):
    """
    """

    name = attr.ib(
        metadata={"help_text": "Name of your start for easy search and reuse"}
    )
