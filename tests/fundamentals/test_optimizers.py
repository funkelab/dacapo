from ..fixtures.fundamentals.optimizers import OPTIMIZERS

from dacapo.store.converter import converter
from dacapo.fundamentals.optimizers import Optimizer

import pytest


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_augments(optimizer):

    # Test that the optimizer provides all the necessary information
    assert optimizer.name is not None and isinstance(optimizer.name, str)

    # Test that the optimizer is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(optimizer)
    assert serialized == converter.unstructure(
        converter.structure(serialized, Optimizer)
    )
