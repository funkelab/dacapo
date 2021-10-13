from ..fixtures.stateless.optimizers import OPTIMIZERS
from ..fixtures.db import DB_AVAILABLE, mongo_config_store

from dacapo.attrs.converter import converter
from dacapo.stateless.optimizers import Optimizer

import pytest


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_optimizers(optimizer):

    # Test that the optimizer provides all the necessary information
    assert optimizer.name is not None and isinstance(optimizer.name, str)

    # Test that the optimizer is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(optimizer)
    assert serialized == converter.unstructure(
        converter.structure(serialized, Optimizer)
    )


@pytest.mark.skipif(not DB_AVAILABLE, reason="Mongodb not available")
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_db(optimizer, mongo_config_store):
    mongo_config_store.store_optimizer(optimizer)
    retrieved_optimizer = mongo_config_store.retrieve_optimizer(optimizer.name)
    assert converter.unstructure(optimizer) == converter.unstructure(
        retrieved_optimizer
    )
