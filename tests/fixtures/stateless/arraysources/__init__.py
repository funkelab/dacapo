from .bossdb import bossdb_intensities, bossdb_labels
from .zarr import zarr_source, mk_train_raw, mk_train_gt, mk_val_raw, mk_val_gt

ARRAYSOURCES = [bossdb_labels, bossdb_intensities, zarr_source]
MK_FUNCTIONS = [mk_train_raw, mk_train_gt, mk_val_raw, mk_val_gt]