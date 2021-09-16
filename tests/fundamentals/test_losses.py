from ..fixtures.fundamentals.losses import LOSSES

from dacapo.store.converter import converter
from dacapo.fundamentals.losses import Loss

import pytest


@pytest.mark.parametrize("loss", LOSSES)
def test_augments(loss):

    # Test that the loss provides all the necessary information
    assert loss.name is not None and isinstance(loss.name, str)

    # Test that the loss is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(loss)
    assert serialized == converter.unstructure(
        converter.structure(serialized, Loss)
    )
