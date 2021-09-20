from dacapo.groupings.experiments import Experiment
from dacapo.fundamentals.executers import Executer

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
    
    while not run.complete:
        run.step()

    run.teardown()
