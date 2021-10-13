from .unet import unet  # noqa
from .vggnet import vggnet  # noqa

UNETS = [unet]
ARCHITECTURES = [unet, vggnet]