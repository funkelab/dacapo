from ..datasets import simple_train, simple_validate
from dacapo.fundamentals.datasplits import Default

simple_train_validate = Default(
    name="simple_train_validate", train=simple_train, validate=simple_validate
)
