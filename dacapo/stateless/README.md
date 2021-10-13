### `fundamental`s

DaCapo `fundamental`s are semantically meaningful and reusable pieces that have no state. This is important for a couple reasons:
1) Reusability
    - Using them does not change their values in the database. Thus they can be safely used in many scripts with confidence that fetching a specific `fundamental` by name from the database will always return the same data.
2) Testing
    - Once created a `fundamental` does not need access to the database, so instances can be easily created and tested in isolation.
3) Dynamic fields
    - Since you will never push updated fields to the database, this makes it easy to leave fields as `None`. Thus if you use a `fundamental` that lists its `num_channels` as `None`, you can be confident that it will work with any number of channels, and fill in your specific value from your data sources.