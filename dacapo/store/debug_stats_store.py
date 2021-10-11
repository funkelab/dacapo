from .stats_store import StatsStore

from dacapo.store.training_stats import TrainingStats
from dacapo.store.training_iteration_stats import TrainingIterationStats
from dacapo.store.validation_scores import ValidationScores
from dacapo.store.validation_iteration_scores import ValidationIterationScores


class DebugStatsStore(StatsStore):
    def retrieve_training_stats(self, experiment_name, run_repitition):
        stats = TrainingStats()
        return stats

    def retrieve_validation_scores(self, experiment_name, run_repitition):
        scores = ValidationScores()
        return scores

    def store_training_stats(self):
        raise NotImplementedError()

    def store_validation_scores(self):
        raise NotImplementedError()
