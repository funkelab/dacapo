from ..fixtures.fundamentals.datasplits import MK_FUNCTIONS
from ..fixtures.fundamentals.outputs import OUTPUTS
from ..fixtures.fundamentals.architectures import ARCHITECTURES
from ..fixtures.fundamentals.trainers import TRAINERS
from ..fixtures.fundamentals.dataproviders import DATAPROVIDERS
from ..fixtures.fundamentals.validators import VALIDATORS
from ..fixtures.fundamentals.evaluators import EVALUATORS
from ..fixtures.fundamentals.postprocessors import POSTPROCESSORS

from dacapo import Experiment, validate
from dacapo.fundamentals.optimizers import NullOptimizer
from dacapo.store.debug_config_store import DebugConfigStore
from dacapo.store.debug_stats_store import DebugStatsStore

import pytest


@pytest.mark.parametrize("mkfunction", MK_FUNCTIONS)
@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("output", OUTPUTS)
@pytest.mark.parametrize("trainer", TRAINERS)
@pytest.mark.parametrize("validator", VALIDATORS)
@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("post_processor", POSTPROCESSORS)
@pytest.mark.parametrize("dataprovider", DATAPROVIDERS)
def test_validate(
    tmp_path,
    mkfunction,
    architecture,
    output,
    trainer,
    validator,
    evaluator,
    post_processor,
    dataprovider,
):
    # make the temporary datasets to use:
    datasplit = mkfunction(tmp_path)

    name = "test_validate"

    config_store = DebugConfigStore()
    stats_store = DebugStatsStore()

    null_optimizer = NullOptimizer()

    experiment = Experiment(
        name=name,
        datasplit=datasplit,
        architecture=architecture,
        output=output,
        optimizer=null_optimizer,
        dataprovider=dataprovider,
        trainer=trainer,
        validator=validator,
        config_store=config_store,
        stats_store=stats_store,
    )
    can_validate = experiment.can_validate()

    if not can_validate:
        with pytest.raises(ValueError):
            validate(experiment, 0, 0)
    else:
        validate(experiment, 0, 0)
