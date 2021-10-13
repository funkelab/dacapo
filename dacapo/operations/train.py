from dacapo.groupings.experiments import Experiment
from dacapo.stateless.executers import Executer
from dacapo.operations.validate import validate

from typing import Optional


def train(
    experiment: Experiment,
    repitition: Optional[int] = None,
    executer: Optional[Executer] = None,
):
    """
    Train an experiment with a given executer. This creates a new `Run` with
    repitition number `i+1` where `i` is the number of runs this
    experiment has already started.

    If repitition is provided, it must be less than or equal to `i+1`.
    If less than `i+1` we will attempt to continue training.
    If equal to `i+1` behavior will be the same as if repitition was
    not provided
    """

    # If an executer is chosen, use it. Else, train.
    if executer is not None:
        executer.train(experiment, repitition)
        return

    run = experiment.run(repitition)

    # TODO: Use a context manager?
    run.setup()  # load model weights, initialize data pipeline etc.

    while not run.complete:
        validate_next = run.step()
        if validate_next:
            iteration_scores = validate(
                experiment,
                run.repitition,
                iteration=run.trained_iterations - 1,
                executer=experiment.validation_executer,
            )
            run.add_iteration_scores(iteration_scores)

    run.teardown()  # free resources, stop workers, etc.

    return run.repitition
