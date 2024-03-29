import attr

from typing import Tuple


@attr.s
class DatasetConfig:
    """Configuration class for datasets, to be used to create a ``Dataset``
    instance.
    """

    name: str = attr.ib(
        metadata={
            "help_text": "A unique name for this dataset. This will be saved so you "
            "and others can find and reuse this dataset. Keep it short "
            "and avoid special characters."
        }
    )
    weight: int = attr.ib(
        metadata={
            "help_text": "A weight to indicate this dataset should be sampled from more "
            "heavily"
        },
        default=1,
    )

    def verify(self) -> Tuple[bool, str]:
        """
        Check whether this is a valid DataSet
        """
        return True, "No validation for this DataSet"
