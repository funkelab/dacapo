from .cross_entropy_loss import CrossEntropyLoss
from .mse_loss import MSELoss
from .weighted_mse_loss import WeightedMSELoss

from typing import Union

Losses = Union[MSELoss, WeightedMSELoss, CrossEntropyLoss]
