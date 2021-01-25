from .dataset import Dataset


class RawData:

    def __init__(self, filename, train_ds, validate_ds):

        self.train = Dataset(filename, train_ds)
        self.validate = Dataset(filename, validate_ds)

        assert self.train.num_channels == self.validate.num_channels
        assert self.train.voxel_size == self.validate.voxel_size
        assert self.train.spatial_dims == self.validate.spatial_dims

        self.num_channels = self.train.num_channels
        self.voxel_size = self.train.voxel_size
        self.spatial_dims = self.train.spatial_dims


class GtData:

    def __init__(self, filename, train_ds, validate_ds):

        self.train = Dataset(filename, train_ds)
        self.validate = Dataset(filename, validate_ds)

        assert self.train.num_channels == self.validate.num_channels
        assert self.train.voxel_size == self.validate.voxel_size
        assert self.train.num_classes == self.validate.num_classes
        assert self.train.background_label == self.validate.background_label
        assert self.train.spatial_dims == self.validate.spatial_dims

        self.num_channels = self.train.num_channels
        self.voxel_size = self.train.voxel_size
        self.num_classes = self.train.num_classes
        self.background_label = self.train.background_label
        self.spatial_dims = self.train.spatial_dims


class MaskData:

    def __init__(self, filename, train_ds):

        self.train = Dataset(filename, train_ds)

        # mask should not have a channel dimension
        assert self.train.num_channels == 0

        self.num_channels = self.train.num_channels
        self.voxel_size = self.train.voxel_size


class Data:

    def __init__(self, data_config):

        self.filename = str(data_config.filename)
        self.raw = RawData(
            self.filename,
            data_config.train_raw,
            data_config.validate_raw)
        self.gt = GtData(
            self.filename,
            data_config.train_gt,
            data_config.validate_gt)

        if hasattr(data_config, 'train_mask'):
            self.mask = MaskData(
                self.filename,
                data_config.train_mask)
            self.has_mask = True
        else:
            self.has_mask = False

        if hasattr(data_config, 'ignore_label'):
            self.gt.ignore_label = data_config.ignore_label
        if hasattr(data_config, 'neg_label'):
            self.gt.neg_label = data_config.neg_label
            self.gt.neg_target = data_config.neg_target

        assert self.raw.spatial_dims == self.gt.spatial_dims
