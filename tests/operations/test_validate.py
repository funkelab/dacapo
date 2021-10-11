from ..fixtures.fundamentals.datasplits import MK_FUNCTIONS
from ..fixtures.fundamentals.outputs import OUTPUTS
from ..fixtures.fundamentals.architectures import ARCHITECTURES
from ..fixtures.fundamentals.dataproviders import DATAPROVIDERS

from dacapo import Experiment, train
from dacapo.fundamentals.optimizers import Adam
from dacapo.fundamentals.trainers import DefaultTrainer
from dacapo.fundamentals.validators import DefaultValidator
from dacapo.fundamentals.dataproviders import GunpowderTrain, GunpowderValidate
from dacapo.store.debug_config_store import DebugConfigStore
from dacapo.store.debug_stats_store import DebugStatsStore

import pytest


@pytest.mark.parametrize("mkfunction", MK_FUNCTIONS)
@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("output", OUTPUTS)
@pytest.mark.parametrize("dataprovider", DATAPROVIDERS)
def test_validate(
    tmp_path,
    mkfunction,
    architecture,
    output,
    dataprovider,
):
    # make the temporary datasets to use:
    datasplit = mkfunction(tmp_path)

    name = "test_validate"
    trainer = DefaultTrainer(name="test_validate", num_iterations=100, batch_size=1)
    validator = DefaultValidator(name="test_validate", validation_interval=1)

    config_store = DebugConfigStore()
    stats_store = DebugStatsStore()
    assert len(stats_store._training_stats) == 0

    optimizer = Adam("validation_test")

    train_provider = GunpowderTrain(name="simple_gp_train")
    val_provider = GunpowderValidate(name="simple_gp_validate")

    experiment = Experiment(
        name=name,
        datasplit=datasplit,
        architecture=architecture,
        output=output,
        optimizer=optimizer,
        train_provider=train_provider,
        val_provider=val_provider,
        trainer=trainer,
        validator=validator,
        config_store=config_store,
        stats_store=stats_store,
        experiments_dir=tmp_path,
    )
    can_validate = experiment.can_validate()

    if not can_validate:
        with pytest.raises(ValueError):
            repitition = train(experiment)
    else:
        repitition = train(experiment)

    run = experiment.run(repitition)
    assert run.training_stats.trained_until == 100
    assert run.validation_scores.validated_until == 100

    path_to_best_model = run.best_weights()
    assert path_to_best_model.exists()
    
    path_to_latest_model = run.latest_weights()
    assert path_to_latest_model.exists()
