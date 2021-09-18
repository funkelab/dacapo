from dacapo.fundamentals.architectures import VGGNet

from funlib.geometry import Coordinate

vggnet = VGGNet(
    name="test_vggnet",
    fmaps_in=2,
    input_shape=Coordinate(20, 20, 20),
    fmaps_out=10,
    downsample_factors=[Coordinate(2, 2, 2), Coordinate(2, 2, 2)],
)
