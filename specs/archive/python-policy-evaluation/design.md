# Online policy evaluation: design rationale

Rust owns Game, canonical legal candidates and encoding. Python owns
scheduling, frozen neural inference and experiment reports. A PyO3 Env
returns an owned snapshot and accepts a candidate index; binding identity
to the reset generation rejects submissions for an outdated decision.

Batch across independent environments on the Python side. A Python
callback inside a Rust Policy would put batching and scheduling under
the synchronous Runner loop. The online package uses the offline model
and label-free collation primitives while keeping its extension build
independent of the root Cargo workspace.

Teacher projection acts on the original Rust Decision, preserving
native tie-breaks and action semantics. Six balanced policy placements
per seed separate seating advantage from policy comparison. Persisted
game records are the authority for the summary.

Track checkpoint/training identity separately from the executing
Python source and compiled Rust binding. This distinguishes a changed
model from a changed environment implementation.
