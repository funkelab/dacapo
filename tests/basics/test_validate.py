import pytest

from dacapo.run import Run
from dacapo.executers import Local

ARCHITECTURES = []
DATASETS = []
TRAINERS = []
LOSSES = []
EVALUATORS = []
POST_PROCESSORS = []
PRE_PROCESSORS = []
PREDICTORS = []
VALIDATORS = []


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("predictor", PREDICTORS)
@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("post_processor", POST_PROCESSORS)
@pytest.mark.parametrize("trainer", TRAINERS)
@pytest.mark.parametrize("validator", VALIDATORS)
def test_validate(
    dataset,
    architecture,
    predictor,
    loss,
    evaluator,
    post_processor,
    trainer,
    validator,
):
    executer = Local()

    raise NotImplementedError()