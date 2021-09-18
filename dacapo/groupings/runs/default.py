from .helpers import Run
from .training_iteration_stats import TrainingIterationStats
from dacapo.store import ConfigStore, StatsStore, stats_store, LocalWeightsStore

import attr

from pathlib import Path
import logging

logger = logging.getLogger(__file__)


@attr.s
class Default(Run):

    _training_stats = None

    @property
    def weights_store(self):
        return LocalWeightsStore(self.root_dir / "checkpoints")

    @property
    def training_stats(self):
        if self._training_stats is None:
            raise ValueError("Training stats have not been initialized!")
        else:
            return self._training_stats

    @property
    def name(self):
        return f"{self.experiment_name}:{self.repitition}"

    @property
    def trained_iterations(self):
        return self.training_stats.trained_until

    @property
    def complete(self):
        return self.trained_iterations >= self.trainer.num_iterations

    def retrieve_training_stats(self):
        return self.stats_store.retrieve_training_stats(
            self.experiment_name, self.repitition
        )

    def retrieve_validation_scores(self):
        return self.stats_store.retrieve_validation_scores(
            self.experiment_name, self.repitition
        )

    def __attrs_post_init__(self):

        # read in previous training/validation stats
        self._training_stats = self.retrieve_training_stats()
        # self._validation_scores = self.retrieve_validation_scores()

        self.pipeline.init_train_pipeline(
            self.datasplit.train, self.architecture, self.output, self.trainer
        )
        # self.pipeline.init_validation_pipeline()

        """
        train_until = self.trainer.num_iterations
        trained_until = self.training_stats.trained_until
        validation_interval = self.validator.validation_interval

        logger.info("Current state: trained until %d/%d", trained_until, train_until)

        # read weights of the latest iteration

        latest_weights_iteration = self.weights_store.latest_iteration()

        if trained_until > 0:

            if latest_weights_iteration is None:

                logger.warning(
                    "Run %s was previously trained until %d, but no weights are "
                    "stored. Will restart training from scratch.",
                    self.name,
                    trained_until,
                )

                trained_until = 0
                self.training_stats.delete_after(0)
                self.validation_scores.delete_after(0)

            elif latest_weights_iteration < trained_until:

                logger.warning(
                    "Run %s was previously trained until %d, but the latest "
                    "weights are stored for iteration %d. Will resume training "
                    "from %d.",
                    self.name,
                    trained_until,
                    latest_weights_iteration,
                    latest_weights_iteration,
                )

                trained_until = latest_weights_iteration
                self.training_stats.delete_after(trained_until)
                self.validation_scores.delete_after(trained_until)
                self.weights_store.retrieve_weights(iteration=trained_until)

            elif latest_weights_iteration == trained_until:

                logger.info("Resuming training from iteration %d", trained_until)

                self.weights_store.retrieve_weights(iteration=trained_until)

            elif latest_weights_iteration > trained_until:

                raise RuntimeError(
                    f"Found weights for iteration {latest_weights_iteration}, but "
                    f"run {self.name} was only trained until {trained_until}."
                )
        """

    def step(self):

        iteration_stats = self.training_step()

        self.training_stats.add_iteration_stats(iteration_stats)

        if (iteration_stats.iteration + 1) % self.validator.validation_interval == 0:

            self.validation_step()

        logger.info("Trained until %d, finished.", self.trained_iterations)

    def training_step(self):
        result = self.pipeline.training_step()
        stats = TrainingIterationStats(
            iteration=self.trained_iterations + 1,
            loss=result.loss,
            time=None,
        )
        return stats

    def validation_step(self):
        self.model.eval()

        self.weights_store.store_weights(self.iteration_stats.iteration + 1)
        validate_run(self.iteration_stats.iteration + 1)
        stats_store.store_validation_scores(run_name, self.validation_scores)

        self.model.train()

        stats_store.store_training_stats(run_name, self.training_stats)
        trained_until = self.training_stats.trained_until()
