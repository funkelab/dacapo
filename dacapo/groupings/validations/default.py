from .helpers import Validation
from dacapo.fundamentals.validation_iteration_scores import ValidationIterationScores
from dacapo.store import ConfigStore, StatsStore, stats_store, LocalWeightsStore

import attr
import torch

from pathlib import Path
import logging
import time

logger = logging.getLogger(__file__)


@attr.s
class DefaultValidation(Validation):
    pass
