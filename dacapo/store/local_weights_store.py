from .weights_store import WeightsStore
from pathlib import Path
import logging
import torch

logger = logging.getLogger(__name__)


class LocalWeightsStore(WeightsStore):
    """A local store for network weights."""

    def __init__(self, basedir):

        logger.info("Creating local weights store in directory %s", basedir)

        self.basedir = basedir

    def latest_iteration(self):
        """Return the latest iteration for which weights are available."""

        iterations = sorted([int(path.parts[-1]) for path in self.basedir.glob("*")])

        if not iterations:
            return None

        return iterations[-1]

    def store_weights(self, iteration, model, optimizer):
        """Store the network weights of the given run."""

        logger.info("Storing weights for iteration %d", iteration)

        weights_name = Path(self.basedir, str(iteration))

        if not self.basedir.exists():
            self.basedir.mkdir(parents=True, exist_ok=True)

        weights = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        torch.save(weights, weights_name)

    def retrieve_weights(self, iteration, model, optimizer):
        """Retrieve the network weights of the given run."""

        logger.info("Retrieving weights for iteration %d", iteration)

        weights_name = Path(self.basedir, str(iteration))

        weights = torch.load(weights_name)

        model.load_state_dict(weights["model"])
        optimizer.load_state_dict(weights["optimizer"])
