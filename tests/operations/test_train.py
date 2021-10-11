from dacapo.groupings import experiments
from ..fixtures.fundamentals.datasplits import MK_FUNCTIONS
from ..fixtures.fundamentals.outputs import OUTPUTS
from ..fixtures.fundamentals.architectures import ARCHITECTURES
from ..fixtures.fundamentals.starts import STARTS
from ..fixtures.fundamentals.dataproviders import DATAPROVIDERS
from ..fixtures.fundamentals.optimizers import OPTIMIZERS


from dacapo import Experiment, train
from dacapo.fundamentals.trainers import DefaultTrainer
from dacapo.fundamentals.validators import Null as NullValidator
from dacapo.store.debug_config_store import DebugConfigStore
from dacapo.store.debug_stats_store import DebugStatsStore

import pytest


@pytest.mark.parametrize("mkfunction", MK_FUNCTIONS)
@pytest.mark.parametrize("start", STARTS)
@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("output", OUTPUTS)
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("dataprovider", DATAPROVIDERS)
def test_train(
    tmp_path,
    mkfunction,
    start,
    architecture,
    output,
    optimizer,
    dataprovider,
):
    # make the temporary datasets to use:
    datasplit = mkfunction(tmp_path)

    name = "test_train"
    trainer = DefaultTrainer(name="test_train", num_iterations=100, batch_size=1)
    validator = NullValidator()

    config_store = DebugConfigStore()
    stats_store = DebugStatsStore()

    experiment = Experiment(
        name=name,
        datasplit=datasplit,
        architecture=architecture,
        output=output,
        optimizer=optimizer,
        train_provider=dataprovider,
        val_provider=None,
        trainer=trainer,
        validator=validator,
        config_store=config_store,
        stats_store=stats_store,
    )
    can_train = experiment.can_train()

    if not can_train:
        with pytest.raises(ValueError):
            repitition = train(experiment)
    else:
        repitition = train(experiment)

    run = experiment.run(repitition)
    assert run.training_stats.trained_until == 100
