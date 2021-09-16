from ..fixtures.fundamentals.executers import EXECUTERS

from dacapo.store.converter import converter
from dacapo.fundamentals.executers import Executer

import pytest


@pytest.mark.parametrize("executer", EXECUTERS)
def test_augments(executer):

    # Test that the executer provides all the necessary information
    assert executer.name is not None and isinstance(executer.name, str)

    # Test that the executer is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(executer)
    assert serialized == converter.unstructure(
        converter.structure(serialized, Executer)
    )
