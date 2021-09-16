from .affinities import Affinities  # noqa
from .one_hot_labels import OneHotLabels  # noqa
from .lsd import LSD  # noqa

from typing import Union

Predictors = Union[Affinities, OneHotLabels, LSD]
