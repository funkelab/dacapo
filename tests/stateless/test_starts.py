from ..fixtures.stateless.starts import STARTS
from ..fixtures.db import DB_AVAILABLE, mongo_config_store

from dacapo.attrs.converter import converter
from dacapo.stateless.starts import Start

import pytest


@pytest.mark.parametrize("start", STARTS)
def test_starts(start):

    # Test that the start provides all the necessary information
    assert start.name is not None and isinstance(start.name, str)

    # Test that the start is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(start)
    assert serialized == converter.unstructure(converter.structure(serialized, Start))


@pytest.mark.skipif(not DB_AVAILABLE, reason="Mongodb not available")
@pytest.mark.parametrize("start", STARTS)
def test_db(start, mongo_config_store):
    mongo_config_store.store_start(start)
    retrieved_start = mongo_config_store.retrieve_start(start.name)
    assert converter.unstructure(start) == converter.unstructure(retrieved_start)
