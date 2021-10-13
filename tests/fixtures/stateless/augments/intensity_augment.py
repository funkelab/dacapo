from dacapo.stateless.augments import IntensityAugment

intensity_augment = IntensityAugment(
    name="test_intensity_augment",
    scale_min=0.9,
    scale_max=1.1,
    shift_min=-0.5,
    shift_max=0.5,
)
