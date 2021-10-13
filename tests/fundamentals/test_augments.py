from ..fixtures.fundamentals.augments import AUGMENTS
from ..fixtures.db import DB_AVAILABLE, mongo_config_store

from dacapo.attrs.converter import converter
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
    assert serialized == converter.unstructure(converter.structure(serialized, Augment))


@pytest.mark.skipif(not DB_AVAILABLE, reason="Mongodb not available")
@pytest.mark.parametrize("augment", AUGMENTS)
def test_db(augment, mongo_config_store):
    mongo_config_store.store_augment(augment)
    retrieved_augment = mongo_config_store.retrieve_augment(augment.name)
    assert converter.unstructure(augment) == converter.unstructure(retrieved_augment)
