from .cross_entropy_losses import simple_cross_entropy
from .mse_losses import simple_mse
from .weighted_mse_losses import simple_weighted_mse

LOSSES = [simple_cross_entropy, simple_mse, simple_weighted_mse]