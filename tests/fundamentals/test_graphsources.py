from ..fixtures.fundamentals.graphsources import GRAPHSOURCES

from dacapo.store.converter import converter
from dacapo.fundamentals.graphsources import GraphSources

from gunpowder.graph import GraphKey
from gunpowder.nodes import BatchProvider

import pytest


@pytest.mark.parametrize("datasource", GRAPHSOURCES)
def test_augments(datasource):

    # Test that the datasource provides all the necessary information
    assert datasource.name is not None and isinstance(datasource.name, str)

    test_key = GraphKey("Test")
    assert isinstance(datasource.node(test_key), BatchProvider)

    # Test that the datasource is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(datasource)
    assert serialized == converter.unstructure(
        converter.structure(serialized, GraphSources)
    )
