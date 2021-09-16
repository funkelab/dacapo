from ..fixtures.fundamentals.starts import STARTS

from dacapo.store.converter import converter
from dacapo.fundamentals.starts import Start

import pytest


@pytest.mark.parametrize("start", STARTS)
def test_augments(start):

    # Test that the start provides all the necessary information
    assert start.name is not None and isinstance(start.name, str)

    # Test that the start is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(start)
    assert serialized == converter.unstructure(converter.structure(serialized, Start))
