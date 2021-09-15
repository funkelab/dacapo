from .adam import Adam
from .radam import RAdam
from dacapo.converter import converter

from typing import Union, get_args

Optimizers = Union[Adam, RAdam]

converter.register_unstructure_hook(
    Optimizers,
    lambda o: {"__type__": type(o).__name__, **converter.unstructure(o)},
)
converter.register_structure_hook(
    Optimizers,
    lambda o, _: converter.structure(o, eval(o["__type__"])),
)
