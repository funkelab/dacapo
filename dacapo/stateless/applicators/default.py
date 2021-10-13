from .helpers import Applicator

import attr

from pathlib import Path


@attr.s
class DefaultApplicator(Applicator):
    name: str = attr.ib(default="default")

    keep_model_predictions: bool = True
    keep_post_processing_intermediates: bool = True

    def out_dir(self, experiment_path: Path, run_repitition: int, dataset_name: str) -> Path:
        return experiment_path / f"{run_repitition}" / "applications"

    def out_container(
        self, experiment_name: str, run_repitition: int, dataset_name: str
    ) -> str:
        return f"{dataset_name}"
