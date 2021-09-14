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
@pytest.mark.parametrize("trainer", TRAINERS)
def test_train(
    dataset,
    architecture,
    predictor,
    loss,
    trainer,
):
    name = (
        f"{dataset.name}-{architecture.name}-{predictor.name}-"
        f"{loss.name}-none-none-{trainer.name}-none"
    )
    executer = Local()
    
    run = Run(
        name=name,
        dataset=dataset,
        architecture=architecture,
        predictor=predictor,
        loss=loss,
        evaluator=None,
        post_processor=None,
        trainer=trainer,
        validator=None,
        executer=executer,
        store=None,
    )
    valid = run.is_valid()
    
    if not valid:
        with pytest.raises(ValueError):
            run.train()
    else:
        run.train()
