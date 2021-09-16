from ..fixtures.fundamentals.trainers import TRAINERS

from dacapo.store.converter import converter
from dacapo.fundamentals.trainers import Trainer

import pytest


@pytest.mark.parametrize("trainer", TRAINERS)
def test_augments(trainer):

    # Test that the trainer provides all the necessary information
    assert trainer.name is not None and isinstance(trainer.name, str)

    # Test that the trainer is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(trainer)
    assert serialized == converter.unstructure(converter.structure(serialized, Trainer))
