import attr

from .array_config import ArrayConfig
from .cellmap_array import CellMapArray
from .array import Array

from typing import List


@attr.s
class CellMapArrayConfig(ArrayConfig):
    """This config class provides the necessary configuration for turning an Annotated dataset into a
    multi class binary classification problem"""

    array_type = CellMapArray

    source_array_config: Array = attr.ib(
        metadata={
            "help_text": "The Array from which to pull annotated data. Is expected to contain a volume with uint64 voxels and no channel dimension"
        }
    )

    groupings: List[List[int]] = attr.ib(
        metadata={
            "help_text": "List of id groups, where each id group is a List of ids. "
            "Group i found in groupings[i] will be binarized and placed in channel i."
        }
    )
