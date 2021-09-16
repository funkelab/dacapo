from .affinity_predictors import simple_affinities
from .one_hot_predictors import simple_one_hot
from .lsd_predictors import simple_lsd

PREDICTORS = [simple_lsd, simple_affinities, simple_one_hot]
