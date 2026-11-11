# Unimplemented research decisions

Keep only **evidence-supported decisions not incorporated into the current
project**. This is not a validation history: record the unresolved or explicit
non-adoption decision and the pinned source, **not** historical results, test
counts, timing tables, environment logs, or deleted raw artifacts. Remove an
entry when it no longer informs development. Current behavior and invariants
are owned by code, tests, [rules](baseline-rules.md) and
[architecture](architecture/overview.md).

## Still undecided

- **Exchange-three structural proxy.** The tested Full and U candidates did
  not meet their predeclared adoption conditions; this does not prove that
  the approach lacks all value. **Remaining decision:** abandon the approach
  explicitly or design and preregister a new comparison before replacing
  the current exchange policy. **Sources:**
  [outcome](https://github.com/NoneMore/sichuan-mahjong-sim/blob/d21504988b394072bba6205fa3e6b85e9fc3a1b7/specs/active/heuristic-exchange-structure-proxy/results.md),
  [alternatives](https://github.com/NoneMore/sichuan-mahjong-sim/blob/d21504988b394072bba6205fa3e6b85e9fc3a1b7/specs/active/heuristic-exchange-structure-proxy/design.md).
- **Needed-mask DP suffix lower bound.** Correctness checks supported the
  candidate, but the short/long-workload performance outcomes did not resolve
  its predeclared adoption criteria. **Remaining decision:** perform a new,
  independently preregistered complete performance acceptance or leave the
  production implementation unchanged. **Sources:**
  [outcome](https://github.com/NoneMore/sichuan-mahjong-sim/blob/5360ab6ce520569142341de8ceb814549c50fc07/specs/active/needed-suffix-bound/results.md),
  [acceptance criteria](https://github.com/NoneMore/sichuan-mahjong-sim/blob/5360ab6ce520569142341de8ceb814549c50fc07/specs/active/needed-suffix-bound/spec.md).

## Explicit non-adoption decisions

These apply to the **tested versions** of the candidates, not all possible
future designs. Reconsider only with a materially different approach or
newly registered acceptance criteria.

- **Batch Needed-mask bidirectional sensitivity:** do not adopt the tested
  candidate; it met its checked correctness conditions but failed the
  predefined end-to-end performance gate.
  [Source](https://github.com/NoneMore/sichuan-mahjong-sim/blob/cc1a32e30919d6775d952733328ca875080b5b86/specs/archive/batch-needed-sensitivity/results.md).
- **Masked DP `u16`:** keep the existing `u32` layout. The candidate was
  rolled back after failing its predeclared performance conditions.
  [Source](https://github.com/NoneMore/sichuan-mahjong-sim/blob/01b25dac1f975d60da5c78eb3f0a51cef1654297/specs/archive/masked-dp-u16/results.md).
- **Dual-mask suit DP:** do not adopt the tested shared Needed/Surplus
  candidate; its performance acceptance gate was not met.
  [Source](https://github.com/NoneMore/sichuan-mahjong-sim/blob/85e44a9570b0486ee724e5aa8af881cb07c7759a/specs/archive/dual-mask-suit-dp/results.md).

The [archival decision](https://github.com/NoneMore/sichuan-mahjong-sim/pull/23)
identifies protected historical branches and distinguishes decisions that were
rejected from those left undecided. Pinned links identify historical source
material; they do not imply raw local experiment samples still exist.
