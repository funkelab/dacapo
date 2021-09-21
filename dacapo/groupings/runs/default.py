from .helpers import Run
from dacapo.fundamentals.training_iteration_stats import TrainingIterationStats
from dacapo.store import ConfigStore, StatsStore, stats_store, LocalWeightsStore

import attr
import torch

from pathlib import Path
import logging
import time

logger = logging.getLogger(__file__)


@attr.s
class DefaultRun(Run):

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

    def setup(self):
        self.setup_training()
        if self.validation_executer is None:
            self.validation_setup()

    def setup_training(self):
        # Read training stats from db
        self._training_stats = self.retrieve_training_stats()
        # initialize the data provider
        self.train_provider.init_provider(
            self.datasplit.train, self.architecture, self.output, self.trainer
        )

        # TODO: This should be initialized elsewhere, not on every training step
        self._backbone = self.architecture.module()
        self._heads = [
            predictor.head(self.architecture, self.datasplit.train)
            for predictor in self.output.predictors
        ]
        self._losses = [loss.module() for loss in self.output.losses]
        self._optimizer = self.optimizer.optim(
            [
                {"params": self._backbone.parameters()},
                *[{"params": head.parameters() for head in self._heads}],
            ]
        )
        self._device = None

    def setup_validation(self):
        # self._validation_scores = self.retrieve_validation_scores()

        # self.validation_provider.init_provider(
        #     self.datasplit.validate, self.architecture, self.output, self.validator
        # )
        pass

    def teardown(self):
        self.training_teardown()
        if self.validation_executer is None:
            self.validation_teardown()

    def training_teardown(self):
        self.train_provider.next(done=True)

    def validation_teardown(self):
        pass

    def step(self):

        iteration_stats = self.training_step()

        self.training_stats.add_iteration_stats(iteration_stats)

        if self.validator.validation_step(self.training_stats):

            self.validation_step()

        logger.info("Trained until %d, finished.", self.trained_iterations)

    def training_step(self):
        training_data = self.train_provider.next()

        t1 = time.time()
        self._optimizer.zero_grad()
        backbone_output = self._backbone.forward(
            torch.as_tensor(training_data["raw"], device=self._device).float()
        )
        losses = []
        for predictor, loss, head in zip(
            self.output.predictors, self._losses, self._heads
        ):
            # gather loss inputs
            predicted = head.forward(backbone_output)
            target = torch.as_tensor(
                training_data["targets"][predictor.name], device=self._device
            ).float()
            weights_data = training_data["weights"].get(predictor.name)
            if weights_data is not None:
                weights = torch.as_tensor(weights_data, device=self._device).float()
            else:
                weights = None

            if weights is not None:
                losses.append(loss.forward(predicted, target, weights))
            else:
                losses.append(loss.forward(predicted, target))

        total_loss = torch.prod(torch.stack(losses))
        total_loss.backward()
        self._optimizer.step()
        t2 = time.time()

        stats = TrainingIterationStats(
            iteration=self.trained_iterations + 1,
            loss=total_loss.detach().cpu(),
            time=t2 - t1,
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
