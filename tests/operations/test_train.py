from dacapo.groupings import experiments
from ..fixtures.stateless.datasplits import MK_FUNCTIONS
from ..fixtures.stateless.outputs import OUTPUTS
from ..fixtures.stateless.architectures import ARCHITECTURES
from ..fixtures.stateless.starts import STARTS
from ..fixtures.stateless.dataproviders import DATAPROVIDERS
from ..fixtures.stateless.optimizers import OPTIMIZERS


from dacapo import Experiment, train
from dacapo.stateless.trainers import DefaultTrainer
from dacapo.stateless.validators import Null as NullValidator
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
