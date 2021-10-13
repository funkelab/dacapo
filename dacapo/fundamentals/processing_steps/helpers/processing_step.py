from abc import ABC, abstractmethod

import attr

# from dacapo.store import MongoDbStore


@attr.s
class ProcessingStep(ABC):

    step_id: str = attr.ib()

    @abstractmethod
    def tasks(self, **kwargs):
        # Must return a list of Tasks, and a list of their respective parameters
        pass

    @abstractmethod
    def get_process_function(self):
        # All PostProcessingSteps must define a process function
        # see daisy process function documentation.
        pass

    def get_check_function(self, pred_id):
        # default check function is provided.
        # store = MongoDbStore()
        # return lambda b: store.check_block(pred_id, self.step_id, b.block_id)
        return False
