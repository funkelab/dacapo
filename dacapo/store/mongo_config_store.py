from .config_store import ConfigStore, DuplicateNameError
from .converter import converter
from dacapo.fundamentals.applicators import Applicator
from dacapo.fundamentals.arraysources import ArraySource
from dacapo.fundamentals.datasets import Dataset
from dacapo.fundamentals.executers import Executer
from dacapo.fundamentals.losses import Loss
from dacapo.fundamentals.outputs import Output
from dacapo.fundamentals.predictors import Predictor
# from dacapo.fundamentals.tasks import Task
from dacapo.fundamentals.validators import Validator
from dacapo.fundamentals.architectures import Architecture
from dacapo.fundamentals.augments import Augment
from dacapo.fundamentals.evaluators import Evaluator
from dacapo.fundamentals.graphsources import GraphSource
from dacapo.fundamentals.optimizers import Optimizer
from dacapo.fundamentals.post_processors import PostProcessor
from dacapo.fundamentals.processing_steps import ProcessingStep
from dacapo.fundamentals.starts import Start
from dacapo.fundamentals.trainers import Trainer

from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError

import logging

logger = logging.getLogger(__name__)


class MongoConfigStore(ConfigStore):
    """A MongoDB store for configurations. Used to store and retrieve
    configurations for runs, tasks, architectures, trainers, and datasets.
    """

    def __init__(self, db_host, db_name):

        logger.info(
            "Creating MongoConfigStore:\n\thost    : %s\n\tdatabase: %s",
            db_host,
            db_name,
        )

        self.db_host = db_host
        self.db_name = db_name

        self.client = MongoClient(self.db_host)
        self.database = self.client[self.db_name]
        self.__open_collections()
        self.__init_db()

    def store_applicator(self, applicator):

        applicator_doc = converter.unstructure(applicator)
        self.__save_insert(self.applicators, applicator_doc)

    def retrieve_applicator(self, applicator_name):

        applicator_doc = self.applicators.find_one(
            {"name": applicator_name}, projection={"_id": False}
        )
        return converter.structure(applicator_doc, Applicator)

    def store_arraysource(self, arraysource):

        arraysource_doc = converter.unstructure(arraysource)
        self.__save_insert(self.arraysources, arraysource_doc)

    def retrieve_arraysource(self, arraysource_name):

        arraysource_doc = self.arraysources.find_one(
            {"name": arraysource_name}, projection={"_id": False}
        )
        return converter.structure(arraysource_doc, ArraySource)

    def store_dataset(self, dataset):

        dataset_doc = converter.unstructure(dataset)
        self.__save_insert(self.datasets, dataset_doc)

    def retrieve_dataset(self, dataset_name):

        dataset_doc = self.datasets.find_one(
            {"name": dataset_name}, projection={"_id": False}
        )
        return converter.structure(dataset_doc, Dataset)

    def store_executer(self, executer):

        executer_doc = converter.unstructure(executer)
        self.__save_insert(self.executers, executer_doc)

    def retrieve_executer(self, executer_name):

        executer_doc = self.executers.find_one(
            {"name": executer_name}, projection={"_id": False}
        )
        return converter.structure(executer_doc, Executer)

    def store_loss(self, loss):

        loss_doc = converter.unstructure(loss)
        self.__save_insert(self.losses, loss_doc)

    def retrieve_loss(self, loss_name):

        loss_doc = self.losses.find_one(
            {"name": loss_name}, projection={"_id": False}
        )
        return converter.structure(loss_doc, Loss)

    def store_output(self, output):

        output_doc = converter.unstructure(output)
        self.__save_insert(self.outputs, output_doc)

    def retrieve_output(self, output_name):

        output_doc = self.outputs.find_one(
            {"name": output_name}, projection={"_id": False}
        )
        return converter.structure(output_doc, Output)

    def store_predictor(self, predictor):

        predictor_doc = converter.unstructure(predictor)
        self.__save_insert(self.predictors, predictor_doc)

    def retrieve_predictor(self, predictor_name):

        predictor_doc = self.predictors.find_one(
            {"name": predictor_name}, projection={"_id": False}
        )
        return converter.structure(predictor_doc, Predictor)

    #    def store_task(self, task):
    #
    #        task_doc = converter.unstructure(task)
    #        self.__save_insert(self.tasks, task_doc)
    #
    #    def retrieve_task(self, task_name):
    #
    #        task_doc = self.tasks.find_one(
    #            {"name": task_name}, projection={"_id": False}
    #        )
    #        return converter.structure(task_doc, Task)

    def store_validator(self, validator):

        validator_doc = converter.unstructure(validator)
        self.__save_insert(self.validators, validator_doc)

    def retrieve_validator(self, validator_name):

        validator_doc = self.validators.find_one(
            {"name": validator_name}, projection={"_id": False}
        )
        return converter.structure(validator_doc, Validator)

    def store_architecture(self, architecture):

        architecture_doc = converter.unstructure(architecture)
        self.__save_insert(self.architectures, architecture_doc)

    def retrieve_architecture(self, architecture_name):

        architecture_doc = self.architectures.find_one(
            {"name": architecture_name}, projection={"_id": False}
        )
        return converter.structure(architecture_doc, Architecture)

    def store_augment(self, augment):

        augment_doc = converter.unstructure(augment)
        self.__save_insert(self.augments, augment_doc)

    def retrieve_augment(self, augment_name):

        augment_doc = self.augments.find_one(
            {"name": augment_name}, projection={"_id": False}
        )
        return converter.structure(augment_doc, Augment)

    def store_evaluator(self, evaluator):

        evaluator_doc = converter.unstructure(evaluator)
        self.__save_insert(self.evaluators, evaluator_doc)

    def retrieve_evaluator(self, evaluator_name):

        evaluator_doc = self.evaluators.find_one(
            {"name": evaluator_name}, projection={"_id": False}
        )
        return converter.structure(evaluator_doc, Evaluator)

    def store_graphsource(self, graphsource):

        graphsource_doc = converter.unstructure(graphsource)
        self.__save_insert(self.graphsources, graphsource_doc)

    def retrieve_graphsource(self, graphsource_name):

        graphsource_doc = self.graphsources.find_one(
            {"name": graphsource_name}, projection={"_id": False}
        )
        return converter.structure(graphsource_doc, GraphSource)

    def store_optimizer(self, optimizer):

        optimizer_doc = converter.unstructure(optimizer)
        self.__save_insert(self.optimizers, optimizer_doc)

    def retrieve_optimizer(self, optimizer_name):

        optimizer_doc = self.optimizers.find_one(
            {"name": optimizer_name}, projection={"_id": False}
        )
        return converter.structure(optimizer_doc, Optimizer)

    def store_post_processor(self, post_processor):

        post_processor_doc = converter.unstructure(post_processor)
        self.__save_insert(self.post_processors, post_processor_doc)

    def retrieve_post_processor(self, post_processor_name):

        post_processor_doc = self.post_processors.find_one(
            {"name": post_processor_name}, projection={"_id": False}
        )
        return converter.structure(post_processor_doc, PostProcessor)

    def store_processing_step(self, processing_step):

        processing_step_doc = converter.unstructure(processing_step)
        self.__save_insert(self.processing_steps, processing_step_doc)

    def retrieve_processing_step(self, processing_step_name):

        processing_step_doc = self.processing_steps.find_one(
            {"name": processing_step_name}, projection={"_id": False}
        )
        return converter.structure(processing_step_doc, ProcessingStep)

    def store_start(self, start):

        start_doc = converter.unstructure(start)
        self.__save_insert(self.starts, start_doc)

    def retrieve_start(self, start_name):

        start_doc = self.starts.find_one(
            {"name": start_name}, projection={"_id": False}
        )
        return converter.structure(start_doc, Start)

    def store_trainer(self, trainer):

        trainer_doc = converter.unstructure(trainer)
        self.__save_insert(self.trainers, trainer_doc)

    def retrieve_trainer(self, trainer_name):

        trainer_doc = self.trainers.find_one(
            {"name": trainer_name}, projection={"_id": False}
        )
        return converter.structure(trainer_doc, Trainer)

    def __save_insert(self, collection, data, ignore=None):

        name = data["name"]

        try:

            collection.insert_one(dict(data))

        except DuplicateKeyError:

            existing = collection.find({"name": name}, projection={"_id": False})[0]

            if not self.__same_doc(existing, data, ignore):

                raise DuplicateNameError(
                    f"Data for {name} does not match already stored "
                    f"entry. Found\n\n{existing}\n\nin DB, but was "
                    f"given\n\n{data}"
                )

    def __same_doc(self, a, b, ignore=None):

        if ignore:
            a = dict(a)
            b = dict(b)
            for key in ignore:
                if key in a:
                    del a[key]
                if key in b:
                    del b[key]

        return a == b

    def __init_db(self):

        self.users.create_index([("username", ASCENDING)], name="username", unique=True)

        self.applicators.create_index([("name", ASCENDING)], name="name", unique=True)
        self.arraysources.create_index([("name", ASCENDING)], name="name", unique=True)
        self.datasets.create_index([("name", ASCENDING)], name="name", unique=True)
        self.executers.create_index([("name", ASCENDING)], name="name", unique=True)
        self.losses.create_index([("name", ASCENDING)], name="name", unique=True)
        self.outputs.create_index([("name", ASCENDING)], name="name", unique=True)
        self.predictors.create_index([("name", ASCENDING)], name="name", unique=True)
        self.tasks.create_index([("name", ASCENDING)], name="name", unique=True)
        self.validators.create_index([("name", ASCENDING)], name="name", unique=True)
        self.architectures.create_index([("name", ASCENDING)], name="name", unique=True)
        self.augments.create_index([("name", ASCENDING)], name="name", unique=True)
        self.evaluators.create_index([("name", ASCENDING)], name="name", unique=True)
        self.graphsources.create_index([("name", ASCENDING)], name="name", unique=True)
        self.optimizers.create_index([("name", ASCENDING)], name="name", unique=True)
        self.post_processors.create_index(
            [("name", ASCENDING)], name="name", unique=True
        )
        self.processing_steps.create_index(
            [("name", ASCENDING)], name="name", unique=True
        )
        self.starts.create_index([("name", ASCENDING)], name="name", unique=True)
        self.trainers.create_index([("name", ASCENDING)], name="name", unique=True)

    def __open_collections(self):

        self.users = self.database["users"]
        self.applicators = self.database["applicators"]
        self.arraysources = self.database["arraysources"]
        self.datasets = self.database["datasets"]
        self.executers = self.database["executers"]
        self.losses = self.database["losses"]
        self.outputs = self.database["outputs"]
        self.predictors = self.database["predictors"]
        self.tasks = self.database["tasks"]
        self.validators = self.database["validators"]
        self.architectures = self.database["architectures"]
        self.augments = self.database["augments"]
        self.evaluators = self.database["evaluators"]
        self.graphsources = self.database["graphsources"]
        self.optimizers = self.database["optimizers"]
        self.post_processors = self.database["post_processors"]
        self.processing_steps = self.database["processing_steps"]
        self.starts = self.database["starts"]
        self.trainers = self.database["trainers"]
