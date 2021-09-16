from ..fixtures.fundamentals.predictors import PREDICTORS

from dacapo.store.converter import converter
from dacapo.fundamentals.predictors import Predictor

import pytest


@pytest.mark.parametrize("predictor", PREDICTORS)
def test_augments(predictor):

    # Test that the predictor provides all the necessary information
    assert predictor.name is not None and isinstance(predictor.name, str)

    # Test that the predictor is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(predictor)
    assert serialized == converter.unstructure(
        converter.structure(serialized, Predictor)
    )
