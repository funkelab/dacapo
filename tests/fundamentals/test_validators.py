from ..fixtures.fundamentals.validators import VALIDATORS

from dacapo.store.converter import converter
from dacapo.fundamentals.validators import Validator

import pytest


@pytest.mark.parametrize("validator", VALIDATORS)
def test_augments(validator):

    # Test that the validator provides all the necessary information
    assert validator.name is not None and isinstance(validator.name, str)

    # Test that the validator is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(validator)
    assert serialized == converter.unstructure(converter.structure(serialized, Validator))
