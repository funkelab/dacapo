from ..fixtures.fundamentals.executers import EXECUTERS
from ..fixtures.db import DB_AVAILABLE, mongo_config_store

from dacapo.store.converter import converter
from dacapo.fundamentals.executers import Executer

import pytest


@pytest.mark.parametrize("executer", EXECUTERS)
def test_executers(executer):

    # Test that the executer provides all the necessary information
    assert executer.name is not None and isinstance(executer.name, str)

    # Test that the executer is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(executer)
    assert serialized == converter.unstructure(
        converter.structure(serialized, Executer)
    )


@pytest.mark.skipif(not DB_AVAILABLE, reason="Mongodb not available")
@pytest.mark.parametrize("executer", EXECUTERS)
def test_db(executer, mongo_config_store):
    mongo_config_store.store_executer(executer)
    retrieved_executer = mongo_config_store.retrieve_executer(executer.name)
    assert converter.unstructure(executer) == converter.unstructure(retrieved_executer)
