from .scores_store import ScoresStore

import daisy

from typing import List, Tuple, Dict, Any
from pathlib import Path
import pickle


class DebugScoresStore(ScoresStore):
    """Base class for score stores.

    Stores evaluation scores blockwise."""

    def __init__(self, scores_dir: Path):
        self.scores_dir = scores_dir

    def store_scores(self, block: daisy.Block, scores: Dict[str, Any]):
        task_dir = self.scores_dir / block.task_id
        if not task_dir.exists():
            task_dir.mkdir(parents=True)
        block_file = task_dir / f"{block.block_id[1]}.obj"
        pickle.dump((block, scores), block_file.open("wb"))

    def retrieve_scores(self, task_id: str) -> List[Tuple[int, Dict[str, Any]]]:
        task_dir = self.scores_dir / task_id
        if not task_dir.exists():
            task_dir.mkdir(parents=True)
        block_scores = []
        for score_file in task_dir.iterdir():
            block_id = int(score_file.name[:-4])
            block_scores.append(pickle.load(score_file.open("rb")))

        return block_scores
