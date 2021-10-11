from .helpers import Run
from dacapo.store.training_iteration_stats import TrainingIterationStats
from dacapo.store import ConfigStore, StatsStore, stats_store, LocalWeightsStore

import attr
import torch

from typing import Optional
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
        self.validation_provider.init_provider(
            self.datasplit.validate, self.architecture, self.output, self.validator
        )

        self._backbone.eval()
        for head in self._heads:
            head.eval()

    def teardown(self):
        self.training_teardown()

    def training_teardown(self):
        self.train_provider.next(done=True)

    def validation_teardown(self):
        # prepare for continued training

        self._backbone.train()
        for head in self._heads:
            head.train()

    def step(self):

        iteration_stats = self.training_step()

        self.training_stats.add_iteration_stats(iteration_stats)

        if self.validator.validate_next(self.training_stats, self.validation_scores):

            self.validation_step()

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
        # Do any setup necessary for validation
        self.setup_validation()

        # Run validations
        # self.weights_store.store_weights(self.iteration_stats.iteration + 1)
        # validate_run(self.iteration_stats.iteration + 1)
        # stats_store.store_validation_scores(run_name, self.validation_scores)
        # stats_store.store_training_stats(run_name, self.training_stats)
        # trained_until = self.training_stats.trained_until()

        self.save_weights(overwrite_best=True)

        self.validation_teardown()

    def save_weights(self, overwrite_best: bool = False):
        if not self.weights_dir.exists():
            self.weights_dir.mkdir(parents=True)

        iteration = self.trained_iterations
        state_dict = {
            "backbone": self._backbone.state_dict,
            "optimizer": self._optimizer.state_dict,
        }
        for predictor, head in zip(self.output.predictors, self._heads):
            state_dict[predictor.name] = head.state_dict

        torch.save(state_dict, self.weights_dir / f"{iteration}.checkpoint")

        if overwrite_best:
            torch.save(state_dict, self.weights_dir / f"best.checkpoint")
            assert (self.weights_dir / f"best.checkpoint").exists()
            assert self.best_weights().exists()
