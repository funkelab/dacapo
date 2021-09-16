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
@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("post_processor", POST_PROCESSORS)
def test_evaluate(
    dataset,
    architecture,
    predictor,
    evaluator,
    post_processor,
):
    executer = Local()
    
    raise NotImplementedError()
