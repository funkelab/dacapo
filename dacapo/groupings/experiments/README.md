# `Experiment`

An Experiment is a combination of various `fundamental` components.

## Uniqueness
An Experiment is unique by `name` which can be automatically generated from its components or provided.

## State
The Experiment has a significant amount of state. Each experiment has a root directory within which
it will store things such as weights during training, snapshots, validation results, and applications
to new datasets.
Within the database the `Experiment` will store a running count of the number of repititions of its
training have been run, and what `Applications` have been run.
It will keep track of the best `repitition`/`iteration` pair accross all of its runs.