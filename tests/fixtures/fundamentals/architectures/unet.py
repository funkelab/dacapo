from dacapo.fundamentals.architectures import UNet

from funlib.geometry import Coordinate

unet = UNet(
    name="test_unet",
    fmaps_in=2,
    input_shape=Coordinate(24, 24, 24),
    output_shape=Coordinate(8, 8, 8),
    fmaps_out=5,
    fmap_inc_factor=2,
    downsample_factors=[Coordinate(2, 2, 2)],
)
