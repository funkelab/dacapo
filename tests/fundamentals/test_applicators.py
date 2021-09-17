from ..fixtures.fundamentals.applicators import APPLICATORS
from ..fixtures.db import mongo_config_store, DB_AVAILABLE

from dacapo.store.converter import converter
from dacapo.fundamentals.applicators import Applicators

import pytest


from pathlib import Path


@pytest.mark.parametrize("applicator", APPLICATORS)
def test_applicators(applicator):

    # Test that the applicator provides all the necessary information
    assert applicator.name is not None and isinstance(applicator.name, str)
    assert applicator.keep_model_predictions is not None and isinstance(
        applicator.keep_model_predictions, bool
    )
    assert applicator.keep_post_processing_intermediates is not None and isinstance(
        applicator.keep_post_processing_intermediates, bool
    )

    assert isinstance(applicator.out_dir(Path("test"), 2, "test_data"), Path)
    assert isinstance(applicator.out_container(Path("test"), 2, "test_data"), str)

    # Test that the applicator is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(applicator)
    assert serialized == converter.unstructure(
        converter.structure(serialized, Applicators)
    )


@pytest.mark.skipif(not DB_AVAILABLE, reason="Mongodb not available")
@pytest.mark.parametrize("applicator", APPLICATORS)
def test_db(applicator, mongo_config_store):
    mongo_config_store.store_applicator(applicator)
    retrieved_applicator = mongo_config_store.retrieve_applicator(applicator.name)
    assert converter.unstructure(applicator) == converter.unstructure(
        retrieved_applicator
    )
