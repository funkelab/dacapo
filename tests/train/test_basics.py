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
def test_train(
    dataset,
    architecture,
    predictor,
    loss,
    evaluator,
    post_processor,
    trainer,
    validator,
):
    name = (
        f"{dataset.name}-{architecture.name}-{predictor.name}-"
        f"{loss.name}-{evaluator.name}-{post_processor.name}-"
        f"{trainer.name}-{validator.name}"
    )
    executer = Local()
    
    run = Run(
        name=name,
        dataset=dataset,
        architecture=architecture,
        predictor=predictor,
        loss=loss,
        evaluator=evaluator,
        post_processor=post_processor,
        trainer=trainer,
        validator=validator,
        executer=executer,
        store=None,
    )
    valid = run.is_valid()
    # Create the Run:
    if not valid:
        with pytest.raises(ValueError):
            run.train()
    else:
        run.train()
