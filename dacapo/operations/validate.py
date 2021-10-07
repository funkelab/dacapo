def validate(experiment, repitition, iteration, executer=None):

    # If an executer is chosen, use it. Else, train.
    if executer is not None:
        executer.validate(experiment, repitition, iteration, executer)

    validation = experiment.validate(repitition, iteration)

    # TODO: Use a context manager?
    validation.setup()  # load model weights, initialize data pipeline etc.

    validation.run_blockwise()

    validation.teardown()  # free resources, stop workers, etc.
