from ..fixtures.fundamentals.trainers import TRAINERS
from ..fixtures.db import DB_AVAILABLE, mongo_config_store

from dacapo.attrs.converter import converter
from dacapo.fundamentals.trainers import Trainer

import pytest


@pytest.mark.parametrize("trainer", TRAINERS)
def test_trainers(trainer):

    # Test that the trainer provides all the necessary information
    assert trainer.name is not None and isinstance(trainer.name, str)

    # Test that the trainer is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(trainer)
    assert serialized == converter.unstructure(converter.structure(serialized, Trainer))


@pytest.mark.skipif(not DB_AVAILABLE, reason="Mongodb not available")
@pytest.mark.parametrize("trainer", TRAINERS)
def test_db(trainer, mongo_config_store):
    mongo_config_store.store_trainer(trainer)
    retrieved_trainer = mongo_config_store.retrieve_trainer(trainer.name)
    assert converter.unstructure(trainer) == converter.unstructure(retrieved_trainer)
