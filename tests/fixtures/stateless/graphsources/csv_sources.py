from dacapo.stateless.graphsources import CSVSource

from pathlib import Path

simple_csv_source = CSVSource(name="test_csv_source", filename=Path("test_csv"), ndims=2, id_dim=3)