# Initial development history

These documents are retained for the reasoning and coordination history of
the simulator's **initial implementation**. They are **not** maintained as
current implementation contracts, task lists, or proof that a proposed feature
has shipped. Their original content is preserved apart from repairing a
relative link after this move.

- [Initial architecture design](xzd-rust-simulator-design-revised.md) -
  initial choices, motivations, and unimplemented directions.
- [Initial implementation plan](implementation-plan.md) -
  initial delivery slices and historical completion criteria.
- [Initial Game/Runner integration contract](integration-contract.md) -
  the parallel-implementation coordination agreement, not present-day file
  ownership or edit restrictions.

For current information, use the [architecture overview](../architecture/overview.md)
and the actual [Rust source](../../src/lib.rs); for intended gameplay, use the
[fixed baseline rules](../baseline-rules.md). See
[project governance](../governance.md) for the distinction between implemented
evidence, proposed work, and historical decisions.
