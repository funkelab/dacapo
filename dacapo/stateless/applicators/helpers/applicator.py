import attr

from abc import ABC, abstractmethod
from pathlib import Path


@attr.s
class Applicator(ABC):
    """
    An Applicator is used when you want to run the results of some training runs
    on new data.
    The Applicator is unique for a given experiment, run, and dataset.
    The Applicator will answer the questions of:
    - Where are the results stored?
    - Do we store the network output or just the post processed result? Where does it go?
    - If post_processing creates intermediate results do we keep those and where?
    """

    name: str = attr.ib()

    @abstractmethod
    def out_dir(
        self, experiment_path: Path, run_repitition: int, dataset_name: str
    ) -> Path:
        raise NotImplementedError("Please implement in subclass!")

    @abstractmethod
    def out_container(
        self, experiment_name: str, run_repitition: int, dataset_name: str
    ) -> str:
        raise NotImplementedError("Please implement in subclass!")

    @property
    @abstractmethod
    def keep_model_predictions(self) -> bool:
        raise NotImplementedError("Please implement in subclass!")

    @property
    @abstractmethod
    def keep_post_processing_intermediates(self) -> bool:
        raise NotImplementedError("Please implement in subclass!")
