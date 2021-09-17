from ..fixtures.fundamentals.datasplits import DATASPLITS
from ..fixtures.fundamentals.outputs import OUTPUTS
from ..fixtures.fundamentals.architectures import ARCHITECTURES
from ..fixtures.fundamentals.starts import STARTS
from ..fixtures.fundamentals.trainers import TRAINERS
from ..fixtures.fundamentals.pipelines import PIPELINES
from ..fixtures.fundamentals.executers import simple_local as executer

from dacapo import Experiment

import pytest


@pytest.mark.parametrize("datasplit", DATASPLITS)
@pytest.mark.parametrize("start", STARTS)
@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("output", OUTPUTS)
@pytest.mark.parametrize("trainer", TRAINERS)
@pytest.mark.parametrize("pipeline", PIPELINES)
def test_train(
    datasplit,
    start,
    architecture,
    output,
    trainer,
    pipeline,
):
    name = (
        f"{datasplit.name}-{start.name}-{architecture.name}-"
        f"{output.name}-{trainer.name}-{pipeline.name}"
    )

    run = Experiment(
        name=name,
        datasplit=datasplit,
        architecture=architecture,
        output=output,
        pipeline=pipeline,
        trainer=trainer,
        validator=None,
        executer=executer,
        config_store=None,
        stats_store=None,
    )
    valid = run.is_valid()

    if not valid:
        with pytest.raises(ValueError):
            run.train()
    else:
        run.train()
