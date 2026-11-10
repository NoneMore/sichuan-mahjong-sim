# Python-driven neural policy evaluation

## Scope and boundary

Run complete games with Python-controlled PyTorch policies and compare them with
`HeuristicPolicy`. This Change owns the PyO3 extension, environment lifecycle,
synchronous multi-game inference, and online evaluation. It reuses the policy
package from [offline BC](../bc-offline-training/design.md); binding development
can begin with simple test policies before a trained checkpoint exists.

```text
Rust Game -> Decision -> encoder -> owned numeric snapshot
Python scheduling -> batched policy inference -> local candidate index
Rust cached Action -> Game.submit -> next decision or terminal result
```

Python owns scheduling and policy computation; Rust owns rules, legal candidates,
and encoding. Use a separate extension crate/package so ordinary core/collector
builds and offline training remain independent of Python bindings. Preserve the
existing Rust 1.75 baseline when choosing the extension's build arrangement.

Direct environment calls let Python batch across games. Calling Python through
Rust's single-decision `Policy::choose` would put scheduling on the wrong side
of this boundary. No model export or second neural runtime is needed.

## Decision lifecycle

One handle owns one Game and its pending canonical actions. Expose whichever seat
must act next, including response decisions:

```python
pending = env.reset(seed)
# decision_id, actor_seat, stage, state, candidate_features
outcome = env.step(pending.decision_id, candidate_index)
# next pending decision OR terminal result
```

- Snapshots contain owned float32 arrays from the existing encoder. Python array
  mutation cannot change Game or the handle's stored actions.
- Python submits a local index; Rust resolves the semantic action and submits it
  to Game. A decision identity includes reset generation, so stale or out-of-range
  requests are rejected before mutation.
- Each step accepts one decision and advances Game's automatic transitions. Other
  seats and forced decisions remain explicit decisions.
- Terminal output includes four seat-ordered scores, winners, end reason, and
  decision count. Reset is explicit; a seat winning does not end the whole game.
- Rule errors and the per-game decision limit stop execution with seed/decision
  context. They are not normal terminal results or occasions for substitute actions.

Only encoded player-visible features enter the model. Seat and decision identity
route batches; the binding does not expose concealed opponents' hands or wall
order.

## Batching and teacher access

Python manages independent handles, groups pending decisions by assigned policy,
pads neural inputs, runs a batch, and routes indices back to their handles. Start
with synchronous stepping in one process; parallel workers are a later throughput
optimization. For a single candidate, submit index zero without a network call,
while retaining the decision in counts and traces.

Evaluate without gradients using masked greedy argmax and canonical-first ties,
with the shared model and batching semantics from offline BC.

Expose a native teacher operation that calls `HeuristicPolicy` on the original
Rust Decision and maps its choice to the current canonical candidate index. It
does not advance the game; the normal step commits the choice. This preserves
teacher tie-breaking without reimplementing strategy in Python. Teacher seats
may share an explicitly owned heuristic cache, as in existing native tools.

## Comparison and records

Follow the existing [policy_ab](../../../examples/policy_ab.rs) balanced 2A2B
placement design: for each seed, run all six neural-seat pairs against heuristic
seats, starting fresh games for each placement. This is a free-for-all comparison,
not a cooperative team game.

Use mean terminal net score per neural seat as the primary strength observation;
report score spread, finish reasons, and decision counts. Group placements by seed
when estimating uncertainty. Use validation-bucket seeds for repeated diagnostics
and test-bucket seeds for the frozen final comparison, following the corpus split
algorithm. Checkpoint selection remains offline validation NLL.

Persist the evaluation plan and raw completed-game records, then derive summaries
from them. Record checkpoint identity, runtime source, encoding/model versions,
seed/placement list, selection mode, device/batching settings, decision limit,
teacher cache configuration, and dependency versions. Stop on game/output failure;
preserve partial records without presenting them as a complete comparison.

Check the boundary with native/Python same-seed replay and a shared offline/online
encoding fixture. These establish action mapping and observation consistency
independently of playing strength. Measure environment, teacher, and inference
costs before choosing CPU/GPU batching settings.

## Deferred RL semantics

The environment returns decisions and game results, not algorithm-specific
single-agent transitions: seat B's next decision is not seat A's next observation.
Per-seat trajectories, rewards, discounting, bootstrap rules, and critics belong
to a future RL Change. Async actors, live weight updates, and environment recovery
are likewise outside this evaluation baseline.
