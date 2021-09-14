from .unet import UNet
from .vggnet import VGGNet
from dacapo.converter import converter

from typing import Union, get_args

Architectures = Union[UNet, VGGNet]

converter.register_unstructure_hook(
    Architectures,
    lambda o: {"__type__": type(o).__name__, **converter.unstructure(o)},
)
converter.register_structure_hook(
    Architectures,
    lambda o, _: converter.structure(o, eval(o["__type__"])),
)
