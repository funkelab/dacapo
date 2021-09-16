from dacapo.fundamentals.predictors import Affinities

simple_affinities = Affinities(
    name="simple_affinities",
    neighborhood=[(0, 1), (1, 0)],
    weighting_type="balanced_labels",
)
