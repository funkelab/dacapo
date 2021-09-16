from .adam import Adam
from .radam import RAdam

from typing import Union, get_args

Optimizers = Union[Adam, RAdam]
