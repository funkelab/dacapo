from dacapo.fundamentals.architectures import VGGNet

vggnet = VGGNet(
    name="test_vggnet",
    fmaps_in=2,
    input_shape=[20, 20, 20],
    fmaps_out=10,
    downsample_factors=[[2, 2, 2], [2, 2, 2]],
)
