from ..fixtures.fundamentals.augments import AUGMENTS

from dacapo.store.converter import converter
from dacapo.fundamentals.augments import Augment

import gunpowder as gp

import pytest


@pytest.mark.parametrize("augment", AUGMENTS)
def test_augments(augment):

    # Test that the augment provides all the necessary information
    assert augment.name is not None and isinstance(augment.name, str)

    test_key = gp.ArrayKey("Test")
    assert isinstance(augment.node(test_key), gp.BatchFilter)

    # Test that the augment is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(augment)
    assert serialized == converter.unstructure(
        converter.structure(serialized, Augment)
    )
