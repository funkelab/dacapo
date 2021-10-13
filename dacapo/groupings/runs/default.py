from .helpers import Run
from dacapo.store.training_iteration_stats import TrainingIterationStats
from dacapo.store import ConfigStore, StatsStore, stats_store, LocalWeightsStore

import attr
import torch

from typing import Optional, Dict, Any
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
            self.init_stats()
            return self._training_stats
        else:
            return self._training_stats

    @property
    def validation_scores(self):
        if self._validation_scores is None:
            self.init_stats()
            return self._validation_scores
        else:
            return self._validation_scores

    @property
    def name(self):
        return f"{self.experiment_name}:{self.repitition}"

    @property
    def trained_iterations(self):
        return self.training_stats.trained_until

    @property
    def complete(self):
        return self.trained_iterations >= self.trainer.num_iterations

    @property
    def weights_dir(self) -> Path:
        return self.root_dir / "weights"

    def best_weights(self) -> Optional[Path]:
        checkpoint = self.weights_dir / "best.checkpoint"
        if checkpoint.exists():
            return checkpoint
        else:
            return None

    def latest_weights(self) -> Optional[Path]:
        checkpoints = sorted(
            [
                int(checkpoint.name.split(".")[0])
                for checkpoint in self.weights_dir.iterdir()
                if not checkpoint.name.startswith("best")
            ]
        )
        if len(checkpoints) > 0:
            return self.weights_dir / f"{checkpoints[-1]}.checkpoint"
        else:
            return None

    def checkpoint(self, iteration: int) -> Path:
        return self.weights_dir / f"{iteration}.checkpoint"

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

    def init_stats(self):
        # Read training stats from db
        self._training_stats = self.retrieve_training_stats()
        self._validation_scores = self.retrieve_validation_scores()

    def setup_training(self):
        # initialize the data provider
        self.dataprovider.init_provider(
            self.dataset, self.architecture, self.output, self.trainer
        )

        # TODO: This should be initialized elsewhere, not on every training step
        self._backbone = self.architecture.module()
        self._heads = [
            predictor.head(self.architecture, self.dataset)
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

    def teardown(self):
        self.training_teardown()

    def training_teardown(self):
        self.dataprovider.next(done=True)

    def step(self):
        training_data = self.dataprovider.next()

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
            iteration=self.trained_iterations,
            loss=total_loss.detach().cpu(),
            time=t2 - t1,
        )

        self.training_stats.add_iteration_stats(stats)

        if self.validator.validate_next(self.training_stats, self.validation_scores):
            self.save_weights()
            return True
        else:
            return False

    def add_iteration_scores(self, iteration_scores):
        new_best = self.output.is_better(
            iteration_scores, self.validation_scores.best_iteration_score
        )
        self.validation_scores.add_iteration_scores(iteration_scores, new_best)
        if new_best:
            self.update_best_model(iteration_scores.iteration)

    def save_weights(self):
        if not self.weights_dir.exists():
            self.weights_dir.mkdir(parents=True)

        # the last trained iteration
        iteration = self.trained_iterations - 1

        state_dict = {
            "backbone": self._backbone.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }
        for predictor, head in zip(self.output.predictors, self._heads):
            state_dict[predictor.name] = head.state_dict()

        torch.save(state_dict, self.weights_dir / f"{iteration}.checkpoint")

    def update_best_model(self, iteration: int):
        iteration_weights = self.weights_dir / f"{iteration}.checkpoint"
        best_weights = self.weights_dir / "best.checkpoint"
        best_weights.write_bytes(iteration_weights.read_bytes())
