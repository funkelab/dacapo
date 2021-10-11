from .stats_store import StatsStore

from dacapo.store.training_stats import TrainingStats
from dacapo.store.training_iteration_stats import TrainingIterationStats
from dacapo.store.validation_scores import ValidationScores
from dacapo.store.validation_iteration_scores import ValidationIterationScores


class DebugStatsStore(StatsStore):

    def __init__(self):
        self._training_stats = {}
        self._validation_scores = {}

    def retrieve_training_stats(self, experiment_name, run_repitition):
        key = (experiment_name, run_repitition)
        if key not in self._training_stats:
            self._training_stats[key] = TrainingStats()
        return self._training_stats[key]

    def retrieve_validation_scores(self, experiment_name, run_repitition):
        key = (experiment_name, run_repitition)
        if key not in self._validation_scores:
            self._validation_scores[key] = ValidationScores()
        return self._validation_scores[key]

    def store_training_stats(self):
        raise NotImplementedError()

    def store_validation_scores(self):
        raise NotImplementedError()
