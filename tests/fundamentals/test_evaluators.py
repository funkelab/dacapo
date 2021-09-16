from ..fixtures.fundamentals.evaluators import EVALUATORS

from dacapo.store.converter import converter
from dacapo.fundamentals.evaluators import Evaluator

import pytest


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_augments(evaluator):

    # Test that the evaluator provides all the necessary information
    assert evaluator.name is not None and isinstance(evaluator.name, str)

    # Test that the evaluator is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(evaluator)
    assert serialized == converter.unstructure(
        converter.structure(serialized, Evaluator)
    )
