from dacapo.fundamentals.arraysources import BossDBSource

from funlib.geometry import Coordinate

bossdb_source = BossDBSource(
    name="test_bossdb",
    db_name="test",
    axes=["x", "y"],
    voxel_size=Coordinate(1, 1),
    offset=Coordinate(2, 3),
    shape=Coordinate(20, 20),
    num_channels=1,
)
