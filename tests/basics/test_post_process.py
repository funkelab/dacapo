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
@pytest.mark.parametrize("post_processor", POST_PROCESSORS)
def test_post_process(
    dataset,
    architecture,
    predictor,
    post_processor,
):
    executer = Local()

    raise NotImplementedError()