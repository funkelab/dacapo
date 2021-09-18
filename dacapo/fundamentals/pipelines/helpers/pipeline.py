import attr

from abc import ABC, abstractmethod


@attr.s
class Pipeline(ABC):
    name: str = attr.ib()

    @abstractmethod
    def init_train_pipeline(self, dataset):
        pass

    @abstractmethod
    def init_validation_pipeline(self):
        pass

    @abstractmethod
    def training_step(self):
        pass

    @abstractmethod
    def validation_step(self):
        pass
