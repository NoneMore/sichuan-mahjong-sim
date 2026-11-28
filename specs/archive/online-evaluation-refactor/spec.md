# Online evaluation refactor — Specification

[design.md](design.md) explains the architecture; [tasks.md](tasks.md) defines implementation slices. This is a prerequisite for RL self-play infrastructure, not an RL training or strategy experiment.

## Scope

- Extract one public, reusable synchronous multi-game execution kernel in `mahjong-eval`. Both existing 2N2T evaluation and future external collectors use it; there must not be two independent game loops.
- Preserve Rust `Game`, PyO3 `Env`, Encoding v1, `mahjong-bc` model and checkpoint contracts, and the existing `mahjong-eval compare` command, six-placement evaluation protocol and persisted format v1.
- Allow `compare` to accept compatible checkpoints with role `best` or `policy`; continue rejecting `last`. Selection of a checkpoint is the caller's responsibility. This is the only intended change to compare input eligibility.
- Do not implement RL, historical-policy pools, rollout proportions, trajectories, rewards, training recovery, model selection, new evaluation protocols/CLI commands, or general-purpose persisted evaluation formats.

## Public execution contract

### Plan and identity

A finite nonempty plan identifies each game by a unique caller-supplied game ID, a valid unsigned 64-bit seed and an ordered four-seat assignment of policy IDs. The same seed and assignment may be repeated under distinct game IDs. IDs remain stable through batching, completion and callbacks. Validate plan, policy registrations, positive batch/concurrency/decision limits and callback arguments before starting any environment. The kernel does not enforce corpus splits, balance placements or assign experimental roles.

A policy registry binds each referenced ID to an ordinary selector or a native teacher registration for the duration of a run. Distinct IDs can represent different model weights or selection configurations. Group by policy ID, not model architecture, checkpoint path or native decision ID. The caller owns policy lifetime and is responsible for keeping policy semantics fixed for a run. Selectors may hold explicit RNG state. Do not silently substitute policies.

### Selection and routing

- Each active game owns its Env. Reset and step use only the originating handle and its pending canonical decision ID. Every accepted step is counted once; no game ends on a seat's Hu alone. Only a complete Terminal ends a game.
- Selectors receive ordered, owned player-visible decisions with game ID, seat and policy identity. Return exactly one local integer candidate index per eligible request, optionally with opaque caller-owned metadata associated with that choice. The kernel validates result count, integer index type and candidate bounds, rejecting booleans and missing results. A one-candidate decision is always index zero without calling a selector or teacher; count it, mark it forced and report no selector metadata.
- Public callers can register native teacher policies. Teacher selection uses the original Rust decision through `Env.teacher_index` and one explicit `TeacherCache` shared by the run (disabled explicitly if none supplied). Expose no raw Env or hidden state to ordinary custom selectors or observers. Keep teacher query capability within the trusted execution/adapter boundary.
- Support independent neural selectors, greedy canonical tie-breaking and legal-mask stochastic sampling at T=1 using an explicit caller-supplied generator. Built-in neural selectors reject nonfinite logits at valid candidate positions before selecting an action; frozen selectors run in eval mode without gradients. Forced actions consume no sampling RNG. Scheduling order is deterministic for a fixed plan and configuration. Built-in sampling supports replay with identical policy and generator states under the same deterministic execution conditions, including device and dependencies. Custom selectors own their additional sources of randomness. Replay across different batching, devices or dependency versions is not required.
- Bound active handles and selector chunks. Fill and refill in plan order; within each round, traverse native teacher groups first, then ordinary selector groups, each tier ordered by first occurrence among current pending requests in active game order. Within ordinary groups, chunk in active game order before filtering forced requests; skip empty eligible chunks. Route choices by originating game/request position, never by native decision ID alone. Validate all choices for the round before any Env step; then commit in active game order. This preserves existing 2N2T teacher-first and neural chunk/forced filtering semantics.

### Observation and completion

An optional accepted-decision observer receives the game ID, seed, seat, policy ID, pre-action owned Decision, selected index, forced flag and optional opaque choice metadata only after a successful Env step, exactly once within a successful invocation. Observe the final accepted step before terminal notification.

Observers may retain owned player-visible snapshots; the kernel does not retain arrays, graphs or metadata in default results. Callbacks are synchronous and must not re-enter or mutate execution/policy state.

Return validated terminal facts (game identity, seed, assignments, four seat scores, winner seat indices, end reason, accepted and forced decision counts by seat, totals); validate native seed, zero-sum scores, winner/end consistency and decision limit/count agreement. Notify the terminal observer once for each validated completed game, before refill. Results are returned in plan order regardless of completion order.

Provide optional scalar trace and counts/timing diagnostics distinguishing accepted and forced decisions by seat/policy, ordinary selector requests (rows) and invocations (batches), and native teacher queries. Count only eligible requests in selector rows/batches; the kernel does not count internal model forwards. The 2N2T adapter preserves existing neural inference accounting.

Report contextual errors for invalid selections, native failures, limit exhaustion and callback failure. Stop immediately on error, with no synthetic terminal, successful partial result, retry or fallback. Round prevalidation does not roll back accepted steps or callback effects if a later step or callback fails. The caller owns partial callback side effects and any collector commit/recovery protocol.

## Existing evaluation compatibility

- Preserve `scheduler.GameSpec(seed, neural_seats)`, `run_games(..., neural=..., teacher_cache=...)`, result/trace/count/timing semantics and `env_factory` test seam through a thin 2N2T adapter. Keep `FrozenPolicy` greedy by default and its public selection API compatible. New public execution APIs must be documented and usable without importing RL.
- Keep CLI arguments/defaults, seed split checks, six placements, primary `neural_mean`, delta and seed-block SE, plan order, terminal persistence, costs and completion/failure semantics. Preserve existing format v1 readers/writers and old records; compare records remain 2N2T-specific. Record actual loaded checkpoint role and identity; policy-only provenance may be null. Do not label a `policy` export as selected `best` or fabricate training state.
- Existing `best` comparisons must retain semantic decisions and terminal results on fixed fixtures with the same device/configuration. Source identity and elapsed timings naturally differ. Formal evaluation remains greedy; stochastic selection is available to callers of the execution API only.

## Acceptance

Focused unit and contract tests cover input validation before Env creation, duplicate IDs, repeated seeds with distinct IDs, wrong-policy routing and colliding native decision IDs, bounded batches and round prevalidation, forced actions without selector calls or RNG draws, owned observations, metadata alignment, terminal callback order, plan-order results, immediate failure/no-success behavior and fixed-configuration sampling replay. Existing Rust/Python and online tests must continue to pass (update tests that intentionally rejected `policy`).

Native smoke tests complete 2N2T, 4N, 4T and two distinct neural policies in 2A2B; compare 4T semantic actions and terminal scores against native teacher replay. Include a real or scripted contract case for early Hu continuation and final settlement.

Check fixed-seed old/new greedy 2N2T trace, counts, summary and record compatibility; confirm v3 `policy` export from BC weights loads and completes existing compare with one valid validation seed (six games), and `last` still fails before publishing a run.

Record a small matched baseline/candidate timing and throughput check; document findings and investigate apparent regressions before handoff. No strength threshold, large-scale experiment or numerical performance acceptance gate.

Update the owning [online guide](../../../online/README.md), [neural architecture](../../../docs/architecture/neural.md), [data contracts](../../../docs/reference/data-contracts.md) and development/CI documentation where necessary. Archive after implementation and verification per [governance](../../../docs/governance.md). Future RL work depends on this documented execution API, not archived Change files.
