# `Application`

When you want to apply the results of training one of your experiments to a new dataset, you must create
an `Application`.

## State
If the `Application` is run blockwise which it usually must due to the fact that you usually want to apply
to large datasets, the `Application` will store things such as block statuses.