from ..fixtures.fundamentals.datasplits import MK_FUNCTIONS
from ..fixtures.fundamentals.outputs import OUTPUTS
from ..fixtures.fundamentals.architectures import UNETS

from dacapo import Experiment, train
from dacapo.fundamentals.optimizers import Adam
from dacapo.fundamentals.trainers import DefaultTrainer
from dacapo.fundamentals.validators import DefaultValidator
from dacapo.fundamentals.dataproviders import GunpowderTrain
from dacapo.store.debug_config_store import DebugConfigStore
from dacapo.store.debug_stats_store import DebugStatsStore

import daisy

import pytest

NUM_ITERATIONS = 1

@pytest.mark.parametrize("mkfunction", MK_FUNCTIONS)
@pytest.mark.parametrize("architecture", UNETS)
@pytest.mark.parametrize("output", OUTPUTS)
def test_validate(
    tmp_path,
    mkfunction,
    architecture,
    output,
):
    # set daisy to log worker outputs to the temp directory
    daisy.logging.set_log_basedir(f"{tmp_path}/daisy_logs")

    # make the temporary datasets to use:
    datasplit = mkfunction(tmp_path)

    name = "test_validate"
    trainer = DefaultTrainer(name="test_validate", num_iterations=NUM_ITERATIONS, batch_size=1)
    validator = DefaultValidator(name="test_validate", validation_interval=1)

    config_store = DebugConfigStore()
    stats_store = DebugStatsStore()
    assert len(stats_store._training_stats) == 0

    optimizer = Adam("validation_test")

    train_provider = GunpowderTrain(name="simple_gp_train")

    experiment = Experiment(
        name=name,
        datasplit=datasplit,
        architecture=architecture,
        output=output,
        optimizer=optimizer,
        train_provider=train_provider,
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
    assert run.training_stats.trained_until == NUM_ITERATIONS
    assert run.validation_scores.validated_until == NUM_ITERATIONS

    path_to_best_model = run.best_weights()
    assert path_to_best_model.exists()

    path_to_latest_model = run.latest_weights()
    assert path_to_latest_model.exists()
