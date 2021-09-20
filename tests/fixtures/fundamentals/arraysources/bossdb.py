from ..arraytypes import ARRAYTYPES

from dacapo.fundamentals.arraysources import BossDBSource

from funlib.geometry import Coordinate

bossdb_intensities = BossDBSource(
    name="bossdb_intensities",
    db_name="test",
    array_type=ARRAYTYPES[0]
)
bossdb_labels = BossDBSource(
    name="bossdb_labels",
    db_name="test",
    array_type=ARRAYTYPES[1]
)
