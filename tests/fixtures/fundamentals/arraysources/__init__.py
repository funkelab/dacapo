import zarr
from .bossdb import bossdb_source
from .zarr import zarr_source, mk_train_raw, mk_train_gt

ARRAYSOURCES = [bossdb_source, zarr_source]
MK_FUNCTIONS = [mk_train_raw, mk_train_gt]