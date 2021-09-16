from dacapo.fundamentals.architectures import UNet

unet = UNet(
    name="test_unet",
    fmaps_in=2,
    input_shape=[20, 20],
    fmaps_out=5,
    fmap_inc_factor=2,
    downsample_factors=[[2, 2], [2, 2]],
)
