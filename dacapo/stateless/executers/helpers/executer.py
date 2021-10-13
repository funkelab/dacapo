from abc import ABC, abstractmethod

import attr

@attr.s
class Executer(ABC):
    """
    The Executer class controls where and how dacapos
    main functions are run.

    Once we have gathered all the components required to train,
    we will call executer.train(...). The executer now has everything
    required to run the training and can choose when and where to
    execute the train call.
    The executer is also in charge of all logging and error printing.
    If the executer decides to open a bsub call, it should also define
    a place for err and log messages to be stored.
    """
    
    name: str = attr.ib()

    @abstractmethod
    def train(self):
        """
        How should I execute the train call?
        """
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def apply(self):
        pass

    @abstractmethod
    def post_process(self):
        pass