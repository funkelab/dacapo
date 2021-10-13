# star imports ensure visibility of concrete classes, so here they are accepted
# flake8: noqa: F405
from dacapo.basics.arraytypes import *
from dacapo.fundamentals.applicators import *
from dacapo.fundamentals.architectures import *
from dacapo.fundamentals.arraysources import *
from dacapo.fundamentals.augments import *
from dacapo.fundamentals.evaluators import *
from dacapo.fundamentals.executers import *
from dacapo.fundamentals.graphsources import *
from dacapo.fundamentals.losses import *
from dacapo.fundamentals.optimizers import *
from dacapo.fundamentals.predictors import *
# from dacapo.fundamentals.processing_steps import *
from dacapo.fundamentals.starts import *
from dacapo.fundamentals.trainers import *
from dacapo.fundamentals.validators import *

from funlib.geometry import Coordinate, Roi

from pathlib import Path


def register_hierarchy_hooks(converter):
    """Central place to register type hierarchies for conversion."""

    converter.register_hierarchy(Applicator, cls_fun)
    converter.register_hierarchy(Architecture, cls_fun)
    converter.register_hierarchy(ArraySource, cls_fun)
    converter.register_hierarchy(ArrayType, cls_fun)
    converter.register_hierarchy(Augment, cls_fun)
    converter.register_hierarchy(Evaluator, cls_fun)
    converter.register_hierarchy(Executer, cls_fun)
    converter.register_hierarchy(GraphSource, cls_fun)
    converter.register_hierarchy(Loss, cls_fun)
    converter.register_hierarchy(Optimizer, cls_fun)
    converter.register_hierarchy(Predictor, cls_fun)
    # converter.register_hierarchy(Processing_step, cls_fun)
    converter.register_hierarchy(Start, cls_fun)
    converter.register_hierarchy(Trainer, cls_fun)
    converter.register_hierarchy(Validator, cls_fun)


def register_hooks(converter):
    """Central place to register all conversion hooks with the given
    converter."""

    #########################
    # DaCapo specific hooks #
    #########################

    # class hierarchies:
    register_hierarchy_hooks(converter)

    """
    # data source dictionaries:
    converter.register_structure_hook(
        DataSourceConfigs,
        lambda obj, cls: {
            converter.structure(key, DataKey): converter.structure(
                value,
                ArraySourceConfig if isinstance(key, ArrayKey) else GraphSourceConfig,
            )
            for key, value in obj.items()
        },
    )

    # data key enums:
    converter.register_unstructure_hook(
        DataKey,
        lambda obj: type(obj).__name__ + "::" + obj.value,
    )
    converter.register_structure_hook(
        DataKey,
        lambda obj, _: eval(obj.split("::")[0])(obj.split("::")[1]),
    )
    """

    #################
    # general hooks #
    #################

    # coordinate to tuple and back
    converter.register_unstructure_hook(Coordinate, lambda o: tuple(o))
    converter.register_structure_hook(Coordinate, lambda o, _: Coordinate(o))

    # Roi to tuple of tuples and back
    converter.register_unstructure_hook(
        Roi, lambda o: (tuple(o.offset), tuple(o.shape))
    )
    converter.register_structure_hook(Roi, lambda o, _: Roi(o[0], o[1]))

    # path to string and back
    converter.register_unstructure_hook(
        Path,
        lambda o: str(o),
    )
    converter.register_structure_hook(
        Path,
        lambda o, _: Path(o),
    )


def cls_fun(typ):
    """Convert a type string into the corresponding class. The class must be
    visible to this module (hence the star imports at the top)."""
    return eval(typ)
