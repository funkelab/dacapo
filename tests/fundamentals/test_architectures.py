from ..fixtures.fundamentals.architectures import ARCHITECTURES
from ..fixtures.db import DB_AVAILABLE, mongo_config_store

from dacapo.attrs.converter import converter
from dacapo.fundamentals.architectures import Architectures

import pytest
import torch


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_architectures(architecture):

    # Test that the architecture provides all the necessary information
    assert architecture.name is not None and isinstance(architecture.name, str)

    assert isinstance(architecture.module(), torch.nn.Module)

    # Test that the architecture is serializable/deserializable
    # so that it works properly with the database
    serialized = converter.unstructure(architecture)
    assert serialized == converter.unstructure(
        converter.structure(serialized, Architectures)
    )


@pytest.mark.skipif(not DB_AVAILABLE, reason="Mongodb not available")
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_db(architecture, mongo_config_store):
    mongo_config_store.store_architecture(architecture)
    retrieved_architecture = mongo_config_store.retrieve_architecture(architecture.name)
    assert converter.unstructure(architecture) == converter.unstructure(
        retrieved_architecture
    )
