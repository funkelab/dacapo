from ..fixtures.fundamentals.applicators import APPLICATORS

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