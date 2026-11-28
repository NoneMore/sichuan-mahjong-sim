# Online evaluation refactor — Design

[spec.md](spec.md) · [tasks.md](tasks.md). Baseline inspected: `f47801ccc4e03bd667a5ec1693e0b5524546f8fc`. See [governance](../../../docs/governance.md) for ownership.

## Motivation and boundaries

The existing [scheduler](../../../online/mahjong_eval/scheduler.py) owns a useful synchronous refill/batch/step loop, but `GameSpec.neural_seats` fixes two neural seats, its results contain neural/teacher-specific counts, and terminal validation calls the 2N2T `TerminalRecord`. Copying this loop into RL would duplicate legality routing, Env ownership and failure semantics. Extract a neutral kernel with four-seat policy assignments and keep the old entry point as a compatibility adapter. The Rust game remains the only rule/settlement authority.

```text
compare CLI -> 2N2T adapter -> public execution kernel -> Env -> Game
                    |                  |
            records / summary      selectors / teacher adapter
                                       ^
                          future RL collector (external caller)
```

Keep `records.py` and the compare plan/summary format specialized to minimize migration cost. Extract shared terminal-fact validation only if necessary to avoid duplicate invariants. The general kernel is independent of evaluation plans, statistical measures and BC training selection.

## Interface sketch

Suggested module: `mahjong_eval.execution`, separate from the existing comparison adapter `scheduler.py`. API names are provisional; [spec.md](spec.md) defines the observable contract. Game identity is independent of the seed so repeated deals remain distinct within a plan.

```python
@dataclass(frozen=True)
class GameSpec:
    game_id: str
    seed: int
    seat_policies: tuple[str, str, str, str]

@dataclass(frozen=True)
class DecisionRequest:
    game_id: str
    seed: int
    policy_id: str
    decision: Decision  # actor_seat, stage, decision_id, owned arrays

@dataclass(frozen=True)
class Choice:
    index: int
    metadata: object = None  # caller-owned, not persisted by kernel

# Ordinary selectors: select(requests) -> Sequence[Choice]
# NativeTeacher is a trusted registration, not an ordinary selector.
# The run shares one explicitly supplied cache (disabled if omitted).
result = execute_games(
    specs,
    policies={"A": actor_a, "B": actor_b, "T": NativeTeacher()},
    teacher_cache=shared_cache,
    batch_size=64, concurrent_games=6,
    on_decision=accepted_decision_observer,
    on_terminal=terminal_observer,
)
```

Caller-supplied selectors allow a future learner to compute its sampled index, old log-probability and value together, then finalize trajectories in the terminal callback. Opaque choice metadata transports these values without introducing a training protocol. Observers receive accepted actions, not permission to replace them.

Expose a dedicated native teacher registration to public callers. The kernel recognizes it and queries the originating Env using the run-level shared cache, since `teacher_index` requires the native pending decision and cannot be reconstructed from features. Ordinary selectors and observers do not receive this capability.

## Scheduling and inference

Retain the existing synchronous loop: fill active handles in plan order, gather pending requests, compute selections, validate, commit one step per active game, notify and refill. Group by policy ID to avoid mixing weights or selection configurations. Traverse teacher groups before ordinary groups, ordered within each tier by first occurrence in current active pending requests. For all ordinary groups, chunk before filtering forced rows to preserve the legacy 2N2T batch shape; skip empty eligible chunks. Keep stable association between each request and its Env, selected index and metadata until step returns. Validate all choices for the round before stepping so malformed selector output cannot partially advance that round. This is not a transaction over Env steps or callbacks: failures after prevalidation cannot undo accepted actions or caller effects.

Reuse `CandidatePolicy`, `pad_decision_inputs`, `greedy_indices` and `sample_indices` rather than implementing another mask or model. FrozenPolicy's current greedy interface remains unchanged; expose sampling through a separate configured selector/adapter with explicit generator, T=1 and finite valid logits. The caller owns sampling state so external collectors can control replay and eventual recovery without scheduler-owned training state. Batch and group order affect RNG consumption, so replay requires the fixed scheduling configuration and deterministic execution conditions defined in spec. Preserve lazy native/PyTorch imports where the current API permits them.

The 2N2T adapter projects per-seat/policy counts and scalar traces into the existing neural/teacher counts, trace, `TerminalRecord` and `SchedulingResult`. Generic selector rows/batches count requests/invocations, not model forwards; only the legacy adapter maps its neural selector counts to inference accounting. Preserve existing timer boundaries including CUDA synchronization for official compare; generic diagnostics attribute selection costs per policy without promising old timing numbers. Default results omit feature arrays and opaque metadata so execution does not accumulate collector-owned trajectories.

## Evaluation and compatibility

In [cli.py](../../../online/mahjong_eval/cli.py), widen the role check from `best` to `best` or `policy` after normal loading. [inference.py](../../../online/mahjong_eval/inference.py) already permits both. The [checkpoint loader](../../../python/mahjong_bc/checkpoint.py) already supports standard policy exports and the [record identity](../../../online/mahjong_eval/records.py) already records actual role and optional training provenance. Therefore no checkpoint version change or special marker is needed. The evaluator reports results; checkpoint selection is external, whether based on offline or online evidence.

## Verification rationale

Fake Env tests isolate scheduling order, routing and failure behavior; native fixtures establish real action routing and settlement correctness. Capture fixed 2N2T baseline fixtures before loop extraction, including batch boundaries and forced rows; compare semantic traces and terminal results rather than timings or provenance bytes. Decode canonical indices using the existing fixture approach for comparison with native teacher replay. Keep mandatory CI CPU-capable; CUDA-specific checks are conditional on availability. Matched timing is diagnostic rather than a statistical performance experiment.
