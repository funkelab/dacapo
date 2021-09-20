from ..datasets import mk_simple_train, mk_simple_validate
from dacapo.fundamentals.datasplits import DefaultDataSplit


def mk_simple_train_validate(path):
    simple_train = mk_simple_train(path)
    # simple_validate = mk_simple_validate(path)
    simple_validate = None
    return DefaultDataSplit(
        name="simple_train_validate", train=simple_train, validate=simple_validate
    )
