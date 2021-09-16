from .unet import UNet
from .vggnet import VGGNet

from typing import Union

Architectures = Union[UNet, VGGNet]
