from gunpowder.batch import Batch
from ..fixtures.fundamentals.arraysources import ARRAYSOURCES

from dacapo.store.converter import converter
from dacapo.fundamentals.arraysources import ArraySources

from gunpowder.array import ArrayKey
from gunpowder.nodes import BatchProvider

import pytest


@pytest.mark.parametrize("arraysource", ARRAYSOURCES)
def test_augments(arraysource):

    # Test that the arraysource provides all the necessary information
    assert arraysource.name is not None and isinstance(arraysource.name, str)

    test_key = ArrayKey("Test")
    assert isinstance(arraysource.node(test_key), BatchProvider)

    # Test that the arraysource is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(arraysource)
    assert serialized == converter.unstructure(
        converter.structure(serialized, ArraySources)
    )
