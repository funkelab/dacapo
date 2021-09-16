from ..fixtures.db import db_available, db

import pytest

pytest.mark.skipif(not db_available)


def test_connection(db):
    pass
