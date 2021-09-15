import attr

from dacapo.converter import converter

from abc import ABC, abstractmethod
from pathlib import Path


@attr.s
class Applicator(ABC):
    """
    An applicator is used when you want to run the results of some training runs
    on new data. This will answer the questions of:
    - Where are the results stored?
    - Do we store the network output or just the post processed result? Where does it go?
    - If post_processing creates intermediate results do we keep those and where?
    """

    name: str = attr.ib(metadata={"help_text": "Unique identifier for this Applicator"})

    @abstractmethod
    def out_dir(self, experiment_path: Path) -> Path:
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
