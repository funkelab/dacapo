from abc import ABC, abstractmethod

from typing import List, Tuple, Dict


class ScoresStore(ABC):
    """Base class for score stores.

    Stores evaluation scores blockwise."""

    @abstractmethod
    def store_scores(self, task_id, block_id, scores):
        """Get the array identifier for a particular validation prediction."""
        pass

    @abstractmethod
    def retrieve_scores(self, task_id) -> List[Tuple[int, Dict]]:
        """Get the array identifier for a particular validation output."""
        pass
