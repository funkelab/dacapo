from ..fixtures.stateless.graphsources import GRAPHSOURCES
from ..fixtures.db import DB_AVAILABLE, mongo_config_store

from dacapo.attrs.converter import converter
from dacapo.stateless.graphsources import GraphSource

from gunpowder.graph import GraphKey
from gunpowder.nodes import BatchProvider

import pytest


@pytest.mark.parametrize("datasource", GRAPHSOURCES)
def test_graphsources(datasource):

    # Test that the datasource provides all the necessary information
    assert datasource.name is not None and isinstance(datasource.name, str)

    test_key = GraphKey("Test")
    assert isinstance(datasource.node(test_key), BatchProvider)

    # Test that the datasource is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(datasource)
    assert serialized == converter.unstructure(
        converter.structure(serialized, GraphSource)
    )


@pytest.mark.skipif(not DB_AVAILABLE, reason="Mongodb not available")
@pytest.mark.parametrize("graphsource", GRAPHSOURCES)
def test_db(graphsource, mongo_config_store):
    mongo_config_store.store_graphsource(graphsource)
    retrieved_graphsource = mongo_config_store.retrieve_graphsource(graphsource.name)
    assert converter.unstructure(graphsource) == converter.unstructure(
        retrieved_graphsource
    )
