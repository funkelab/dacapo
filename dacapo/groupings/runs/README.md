# `Run`

A `Run` is a specific instance of pushing data through the model with backprop for a predefined number of iterations.

## Uniqueness
A `Run` is unique by its `repitition` number and has no name

## State
A `Run` will store its weights, snapshots, etc. in sub directories of its parent `Experiment`.
In the database, a `Run` will store training stats, and its best iteration.