from ..fixtures.stateless.predictors import PREDICTORS
from ..fixtures.db import DB_AVAILABLE, mongo_config_store

from dacapo.attrs.converter import converter
from dacapo.stateless.predictors import Predictor

import pytest


@pytest.mark.parametrize("predictor", PREDICTORS)
def test_predictors(predictor):

    # Test that the predictor provides all the necessary information
    assert predictor.name is not None and isinstance(predictor.name, str)

    # Test that the predictor is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(predictor)
    assert serialized == converter.unstructure(
        converter.structure(serialized, Predictor)
    )


@pytest.mark.skipif(not DB_AVAILABLE, reason="Mongodb not available")
@pytest.mark.parametrize("predictor", PREDICTORS)
def test_db(predictor, mongo_config_store):
    mongo_config_store.store_predictor(predictor)
    retrieved_predictor = mongo_config_store.retrieve_predictor(predictor.name)
    assert converter.unstructure(predictor) == converter.unstructure(
        retrieved_predictor
    )
