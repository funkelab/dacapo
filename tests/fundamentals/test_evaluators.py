from ..fixtures.fundamentals.evaluators import EVALUATORS
from ..fixtures.db import DB_AVAILABLE, mongo_config_store

from dacapo.store.converter import converter
from dacapo.fundamentals.evaluators import Evaluator

import pytest


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_evaluators(evaluator):

    # Test that the evaluator provides all the necessary information
    assert evaluator.name is not None and isinstance(evaluator.name, str)

    # Test that the evaluator is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(evaluator)
    assert serialized == converter.unstructure(
        converter.structure(serialized, Evaluator)
    )


@pytest.mark.skipif(not DB_AVAILABLE, reason="Mongodb not available")
@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_db(evaluator, mongo_config_store):
    mongo_config_store.store_evaluator(evaluator)
    retrieved_evaluator = mongo_config_store.retrieve_evaluator(evaluator.name)
    assert converter.unstructure(evaluator) == converter.unstructure(
        retrieved_evaluator
    )
