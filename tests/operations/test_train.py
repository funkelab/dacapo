from ..fixtures.fundamentals.datasplits import MK_FUNCTIONS
from ..fixtures.fundamentals.outputs import OUTPUTS
from ..fixtures.fundamentals.architectures import ARCHITECTURES
from ..fixtures.fundamentals.starts import STARTS
from ..fixtures.fundamentals.trainers import TRAINERS
from ..fixtures.fundamentals.dataproviders import DATAPROVIDERS
from ..fixtures.fundamentals.optimizers import OPTIMIZERS
from ..fixtures.fundamentals.executers import simple_local as executer

from dacapo import Experiment, train
from dacapo.fundamentals.validators import Null as NullValidator
from dacapo.store.debug_config_store import DebugConfigStore
from dacapo.store.debug_stats_store import DebugStatsStore

import pytest


@pytest.mark.parametrize("mkfunction", MK_FUNCTIONS)
@pytest.mark.parametrize("start", STARTS)
@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("output", OUTPUTS)
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("trainer", TRAINERS)
@pytest.mark.parametrize("dataprovider", DATAPROVIDERS)
def test_train(
    tmp_path,
    mkfunction,
    start,
    architecture,
    output,
    optimizer,
    trainer,
    dataprovider,
):
    # make the temporary datasets to use:
    datasplit = mkfunction(tmp_path)

    name = (
        f"{datasplit.name}-{start.name}-{architecture.name}-"
        f"{output.name}-{trainer.name}-{dataprovider.name}"
    )

    validator = NullValidator()
    config_store = DebugConfigStore()
    stats_store = DebugStatsStore()

    experiment = Experiment(
        name=name,
        datasplit=datasplit,
        architecture=architecture,
        output=output,
        optimizer=optimizer,
        dataprovider=dataprovider,
        trainer=trainer,
        validator=validator,
        config_store=config_store,
        stats_store=stats_store,
    )
    valid = experiment.is_valid()

    if not valid:
        with pytest.raises(ValueError):
            train(experiment)
    else:
        train(experiment)
