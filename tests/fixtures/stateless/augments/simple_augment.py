from dacapo.stateless.augments import SimpleAugment

simple_augment = SimpleAugment(
    name="test_simple_augment", mirror_only=[0, 1], transpose_only=[0, 1]
)
