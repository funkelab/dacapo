from ..fixtures.stateless.arraysources import ARRAYSOURCES
from ..fixtures.db import DB_AVAILABLE, mongo_config_store

from dacapo.attrs.converter import converter
from dacapo.stateless.arraysources import ArraySources

from gunpowder.array import ArrayKey
from gunpowder.nodes import BatchProvider

import pytest


@pytest.mark.parametrize("arraysource", ARRAYSOURCES)
def test_augments(arraysource):

    # Test that the arraysource provides all the necessary information
    assert arraysource.name is not None and isinstance(arraysource.name, str)

    test_key = ArrayKey("Test")
    
    # TODO: Test the arraysource api. This is expected to only work
    # If there is actually data available.
    # assert isinstance(arraysource.node(test_key), BatchProvider)

    # Test that the arraysource is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(arraysource)
    assert serialized == converter.unstructure(
        converter.structure(serialized, ArraySources)
    )


@pytest.mark.skipif(not DB_AVAILABLE, reason="Mongodb not available")
@pytest.mark.parametrize("arraysource", ARRAYSOURCES)
def test_db(arraysource, mongo_config_store):
    mongo_config_store.store_arraysource(arraysource)
    retrieved_arraysource = mongo_config_store.retrieve_arraysource(arraysource.name)
    assert converter.unstructure(arraysource) == converter.unstructure(
        retrieved_arraysource
    )
