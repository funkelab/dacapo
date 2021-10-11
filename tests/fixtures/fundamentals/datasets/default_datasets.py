from dacapo.fundamentals.datasets import DefaultDataset
from ..arraysources import mk_train_raw, mk_train_gt, mk_val_raw, mk_val_gt


def mk_simple_train(path):
    """
    create the data necessary to use the
    "simple_train" dataset at the given `path`
    """
    train_raw = mk_train_raw(path)
    train_gt = mk_train_gt(path)
    return DefaultDataset(name="simple_train", raw=train_raw, gt=train_gt, mask=None)


def mk_simple_validate(path):
    """
    create the data necessary to use the
    "simple_validate" dataset at the given `path`
    """
    val_raw = mk_val_raw(path)
    val_gt = mk_val_gt(path)
    return DefaultDataset(name="simple_validate", raw=val_raw, gt=val_gt, mask=None)
