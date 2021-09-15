from ..fixtures.fundamentals.applicators import APPLICATORS

import pytest

from pathlib import Path


@pytest.mark.parametrize("applicator", APPLICATORS)
def test_applicators(applicator):

    assert applicator.name is not None and isinstance(applicator.name, str)
    assert applicator.keep_model_predictions is not None and isinstance(
        applicator.keep_model_predictions, bool
    )
    assert applicator.keep_post_processing_intermediates is not None and isinstance(
        applicator.keep_post_processing_intermediates.name, str
    )

    assert isinstance(applicator.out_dir(Path("test"), 2, "test_data"), Path)
    assert isinstance(applicator.out_container(Path("test"), 2, "test_data"), str)
