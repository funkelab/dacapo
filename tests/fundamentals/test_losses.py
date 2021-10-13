from ..fixtures.fundamentals.losses import LOSSES
from ..fixtures.db import DB_AVAILABLE, mongo_config_store

from dacapo.attrs.converter import converter
from dacapo.fundamentals.losses import Loss

import pytest


@pytest.mark.parametrize("loss", LOSSES)
def test_losses(loss):

    # Test that the loss provides all the necessary information
    assert loss.name is not None and isinstance(loss.name, str)

    # Test that the loss is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(loss)
    assert serialized == converter.unstructure(converter.structure(serialized, Loss))


@pytest.mark.skipif(not DB_AVAILABLE, reason="Mongodb not available")
@pytest.mark.parametrize("loss", LOSSES)
def test_db(loss, mongo_config_store):
    mongo_config_store.store_loss(loss)
    retrieved_loss = mongo_config_store.retrieve_loss(loss.name)
    assert converter.unstructure(loss) == converter.unstructure(retrieved_loss)
