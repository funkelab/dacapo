from abc import ABC, abstractmethod


class DuplicateNameError(Exception):
    pass


class ConfigStore(ABC):
    """Base class for configuration stores."""

    #    @abstractmethod
    #    def store_user(self, user):
    #        """
    #        Store a user config. This should throw an exception if attempting to store
    #        a user that already exists in the database.
    #        """
    #        pass
    #
    #    @abstractmethod
    #    def retrieve_user(self, user_name):
    #        """
    #        Retrieve a user config from a user name.
    #        """
    #        pass

    @abstractmethod
    def store_applicator(self, applicator):
        """
        Store a applicator config. This should throw an exception if attempting to store
        a applicator that already exists in the database.
        """
        pass

    @abstractmethod
    def retrieve_applicator(self, applicator_name):
        """
        Retrieve a applicator config from a applicator name.
        """
        pass

    @abstractmethod
    def store_arraysource(self, arraysource):
        """
        Store a arraysource config. This should throw an exception if attempting to store
        a arraysource that already exists in the database.
        """
        pass

    @abstractmethod
    def retrieve_arraysource(self, arraysource_name):
        """
        Retrieve a arraysource config from a arraysource name.
        """
        pass

    @abstractmethod
    def store_dataset(self, dataset):
        """
        Store a dataset config. This should throw an exception if attempting to store
        a dataset that already exists in the database.
        """
        pass

    @abstractmethod
    def retrieve_dataset(self, dataset_name):
        """
        Retrieve a dataset config from a dataset name.
        """
        pass

    @abstractmethod
    def store_executer(self, executer):
        """
        Store a executer config. This should throw an exception if attempting to store
        a executer that already exists in the database.
        """
        pass

    @abstractmethod
    def retrieve_executer(self, executer_name):
        """
        Retrieve a executer config from a executer name.
        """
        pass

    @abstractmethod
    def store_loss(self, loss):
        """
        Store a loss config. This should throw an exception if attempting to store
        a loss that already exists in the database.
        """
        pass

    @abstractmethod
    def retrieve_loss(self, loss_name):
        """
        Retrieve a loss config from a loss name.
        """
        pass

    @abstractmethod
    def store_output(self, output):
        """
        Store a output config. This should throw an exception if attempting to store
        a output that already exists in the database.
        """
        pass

    @abstractmethod
    def retrieve_output(self, output_name):
        """
        Retrieve a output config from a output name.
        """
        pass

    @abstractmethod
    def store_predictor(self, predictor):
        """
        Store a predictor config. This should throw an exception if attempting to store
        a predictor that already exists in the database.
        """
        pass

    @abstractmethod
    def retrieve_predictor(self, predictor_name):
        """
        Retrieve a predictor config from a predictor name.
        """
        pass

    #    @abstractmethod
    #    def store_task(self, task):
    #        """
    #        Store a task config. This should throw an exception if attempting to store
    #        a task that already exists in the database.
    #        """
    #        pass
    #
    #    @abstractmethod
    #    def retrieve_task(self, task_name):
    #        """
    #        Retrieve a task config from a task name.
    #        """
    #        pass

    @abstractmethod
    def store_validator(self, validator):
        """
        Store a validator config. This should throw an exception if attempting to store
        a validator that already exists in the database.
        """
        pass

    @abstractmethod
    def retrieve_validator(self, validator_name):
        """
        Retrieve a validator config from a validator name.
        """
        pass

    @abstractmethod
    def store_architecture(self, architecture):
        """
        Store a architecture config. This should throw an exception if attempting to store
        a architecture that already exists in the database.
        """
        pass

    @abstractmethod
    def retrieve_architecture(self, architecture_name):
        """
        Retrieve a architecture config from a architecture name.
        """
        pass

    @abstractmethod
    def store_augment(self, augment):
        """
        Store a augment config. This should throw an exception if attempting to store
        a augment that already exists in the database.
        """
        pass

    @abstractmethod
    def retrieve_augment(self, augment_name):
        """
        Retrieve a augment config from a augment name.
        """
        pass

    @abstractmethod
    def store_evaluator(self, evaluator):
        """
        Store a evaluator config. This should throw an exception if attempting to store
        a evaluator that already exists in the database.
        """
        pass

    @abstractmethod
    def retrieve_evaluator(self, evaluator_name):
        """
        Retrieve a evaluator config from a evaluator name.
        """
        pass

    @abstractmethod
    def store_graphsource(self, graphsource):
        """
        Store a graphsource config. This should throw an exception if attempting to store
        a graphsource that already exists in the database.
        """
        pass

    @abstractmethod
    def retrieve_graphsource(self, graphsource_name):
        """
        Retrieve a graphsource config from a graphsource name.
        """
        pass

    @abstractmethod
    def store_optimizer(self, optimizer):
        """
        Store a optimizer config. This should throw an exception if attempting to store
        a optimizer that already exists in the database.
        """
        pass

    @abstractmethod
    def retrieve_optimizer(self, optimizer_name):
        """
        Retrieve a optimizer config from a optimizer name.
        """
        pass

    @abstractmethod
    def store_post_processor(self, post_processor):
        """
        Store a post_processor config. This should throw an exception if attempting to store
        a post_processor that already exists in the database.
        """
        pass

    @abstractmethod
    def retrieve_post_processor(self, post_processor_name):
        """
        Retrieve a post_processor config from a post_processor name.
        """
        pass

    @abstractmethod
    def store_processing_step(self, processing_step):
        """
        Store a processing_step config. This should throw an exception if attempting to store
        a processing_step that already exists in the database.
        """
        pass

    @abstractmethod
    def retrieve_processing_step(self, processing_step_name):
        """
        Retrieve a processing_step config from a processing_step name.
        """
        pass

    @abstractmethod
    def store_start(self, start):
        """
        Store a start config. This should throw an exception if attempting to store
        a start that already exists in the database.
        """
        pass

    @abstractmethod
    def retrieve_start(self, start_name):
        """
        Retrieve a start config from a start name.
        """
        pass

    @abstractmethod
    def store_trainer(self, trainer):
        """
        Store a trainer config. This should throw an exception if attempting to store
        a trainer that already exists in the database.
        """
        pass

    @abstractmethod
    def retrieve_trainer(self, trainer_name):
        """
        Retrieve a trainer config from a trainer name.
        """
        pass
