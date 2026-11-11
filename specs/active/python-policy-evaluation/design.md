# Python-driven neural policy evaluation design

Structural choices for the [specification](spec.md).

## Ownership

```text
Rust Game -> Decision -> encoder -> owned numeric snapshot
Python scheduling -> batched policy inference -> local candidate index
Rust cached Action -> Game.submit -> next decision or terminal result
```

Python owns scheduling and policy computation; Rust owns rules, candidates, and
encoding. Direct environment calls enable cross-game batching; a Python callback
inside `Policy::choose` would put scheduling on the Rust side. PyTorch remains the
only neural runtime.

Use a separate extension crate/package. Online scheduling and reporting live
outside the offline package, sharing its policy primitives while preserving
independent builds and installation.

## Environment handle

```python
pending = env.reset(seed)
# decision_id, actor_seat, stage, state, candidate_features
outcome = env.step(pending.decision_id, candidate_index)
# next pending decision OR terminal result
```

Each handle owns a Game and cached canonical actions. Decision identity includes
reset generation; indices resolve through the cache. Owned snapshots isolate
Python array lifetime/mutation from authoritative state. Game retains progression,
response arbitration, and settlement.

The teacher uses the original pending Decision and the encoder's semantic target
lookup, preserving native tie-breaking. Teacher seats may share an explicitly
owned heuristic cache, as in existing tools.

## Shared inference

Reuse `load_policy_checkpoint`, `CandidatePolicy`, and `greedy_indices` from the
[offline package](../../../python/README.md#training-metrics-and-checkpoints).
Python groups pending decisions by policy, batches inference, and routes indices.

The offline `collate_decisions` requires labels. Extract shared label-free
state/candidate padding and mask construction; offline collation adds supervision
and online inference consumes only model inputs.

## Comparison and evidence

Reuse [policy_ab](../../../examples/policy_ab.rs)'s balanced placement and
seed-block approach and the Rust collection API's seed split. Offline imitation
measures teacher agreement; online scores expose accumulated errors and
distribution shift, so they complement rather than replace offline selection.

Follow the existing imported-package source-record approach for Python and
compiled provenance for Rust. These identify the executing implementations
separately from checkpoint content and training history.

Extend the Rust-to-Python handoff fixture to include binding snapshots. Native/
Python teacher replay separately checks action mapping and progression; neither
requires a strong learned policy.

## Build choice

T1 verified a minimal binding build before committing to versions. The tested
environment is CPython 3.12.10 x86-64 with NumPy 2.5.4, PyO3 0.23.5, rust-numpy
0.23.0, maturin 1.16.0, and Rust 1.75.0. The wheel uses the PyO3 `abi3-py312`
feature, so its verified minimum is Python 3.12; later Python versions are not
claimed without evidence. The online package declares `numpy>=2.0` for the
tested NumPy generation.

The PyO3 crate is `crates/python`, its own Cargo workspace with an independent
lockfile and target directory. The root workspace excludes it, so core, neural,
collector, and offline training builds neither resolve nor link Python. The
online `pyproject.toml` uses maturin with `manifest-path =
"../crates/python/Cargo.toml"`, `python-source = "."`, and `module-name =
"mahjong_eval._native"`. A committed lockfile pins `rustc-hash` to 2.1.1
because later releases declare an MSRV above 1.75.

The first-decision fixture is generated from the public core and encoder APIs
and compared by both a Rust test and the installed Python package. It matches
the existing seed-0 collection corpus bit for bit.

Runtime provenance remains T4 work. `collection::source` identifies the neural
crate's own build; T4 must add extension-scoped evidence that covers this crate,
the core, and the neural crate rather than treating that identity as complete.
