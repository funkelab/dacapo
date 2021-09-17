from ..fixtures.fundamentals.validators import VALIDATORS
from ..fixtures.db import DB_AVAILABLE, mongo_config_store

from dacapo.store.converter import converter
from dacapo.fundamentals.validators import Validator

import pytest


@pytest.mark.parametrize("validator", VALIDATORS)
def test_validators(validator):

    # Test that the validator provides all the necessary information
    assert validator.name is not None and isinstance(validator.name, str)

    # Test that the validator is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(validator)
    assert serialized == converter.unstructure(
        converter.structure(serialized, Validator)
    )


@pytest.mark.skipif(not DB_AVAILABLE, reason="Mongodb not available")
@pytest.mark.parametrize("validator", VALIDATORS)
def test_db(validator, mongo_config_store):
    mongo_config_store.store_validator(validator)
    retrieved_validator = mongo_config_store.retrieve_validator(validator.name)
    assert converter.unstructure(validator) == converter.unstructure(
        retrieved_validator
    )
