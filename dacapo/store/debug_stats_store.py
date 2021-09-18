from .stats_store import StatsStore

from dacapo.groupings.runs import TrainingStats, TrainingIterationStats
# from dacapo.groupings.run import ValidationScores, ValidationIterationScores


class DebugStatsStore(StatsStore):
    def retrieve_training_stats(self, experiment_name, run_repitition):
        stats = TrainingStats()
        return stats

    def retrieve_validation_scores(self, experiment_name, run_repitition):
        raise NotImplementedError()

    def store_training_stats(self):
        raise NotImplementedError()

    def store_validation_scores(self):
        raise NotImplementedError()
