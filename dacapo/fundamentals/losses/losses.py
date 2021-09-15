from .cross_entropy_loss import CrossEntropyLoss
from .mse_loss import MSELoss
from .weighted_mse_loss import WeightedMSELoss
from dacapo.converter import converter

from typing import Union

Losses = Union[MSELoss, WeightedMSELoss, CrossEntropyLoss]

converter.register_unstructure_hook(
    Losses,
    lambda o: {"__type__": type(o).__name__, **converter.unstructure(o)},
)
converter.register_structure_hook(
    Losses,
    lambda o, _: converter.structure(o, eval(o.pop("__type__"))),
)
