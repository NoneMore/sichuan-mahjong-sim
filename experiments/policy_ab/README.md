# Reproducible heuristic / FirstLegal experiment

This optional experiment compares two imported ninone heuristic players (A)
with two simulator `FirstLegalPolicy` players (B). It measures **terminal net
score per player per hand**, including all settlement, for this 2A2B
population under the simulator's fixed rules. It is not a general playing
strength ranking: `FirstLegalPolicy` is a weak legality baseline. Players do
not cooperate or share private information.

## Components and prerequisites

- A stable Rust toolchain builds the `policy_ab` Cargo example. JSON support
  is a development-only dependency; the simulator library and native CLI
  remain independent of Python.
- Python 3 and the original `ninone.policies.agent_heuristic.choose_action`
  in a separate ninone checkout are needed for A. By default the checkout is
  the sibling `../ninone-mahjong-agent`; select a different path explicitly.
- `worker.py` imports that function, rather than translating its strategy.
  `experiment.py` owns sampling, run evidence, validation, and analysis.
- No changes to `Game`, the existing `Policy` trait, `Runner`, or game rules
  are required. This is one experiment, not a supported policy-plugin API.

Build the separate driver from the repository root:

```text
cargo build --release --example policy_ab
```

The executable is `target/release/examples/policy_ab` on Unix and
`target/release/examples/policy_ab.exe` on Windows. Use its path as `--driver`
when running the Python workflow. The core CLI is unchanged.

## Paired deals and policy states

Each distinct u64 seed produces six fresh `Game::new(seed)` games. A occupies
pairs `[0,1]`, `[0,2]`, `[0,3]`, `[1,2]`, `[1,3]`, `[2,3]`, in that order;
B occupies the remaining seats. Both policies occupy each seat three times,
including the fixed dealer. The assignment changes, not the dealer mechanism
or initial wall. Later trajectories can diverge as actions differ.

`FirstLegalPolicy` is fresh for each game and the imported heuristic is
stateless. A long-lived worker therefore does not carry logical game state
across games. Every decision passes through the public `next_decision` /
`submit` interface; only `Game` decides legality, transitions, and scores.

## Adapter identity and information boundary

The JSONL worker request contains only `hand`, `missing_suit`, and
`candidates`. Wan/Tong/Tiao are `m/p/s`; a tile is suit then rank, such as
`m1`, **not** the simulator's display format. The worker constructs the
heuristic's minimal `observation` and `legal_actions` request. It receives no
opponent concealed hand or wall order.

Ordinary candidates preserve the pending decision's order and all options.
Exchange candidates enumerate distinct nondecreasing same-suit triples,
respecting eligible suits and hand multiplicities. Candidate ordering and ID
construction are deterministic and part of the adapter identity; changing
either requires new run evidence because they can change tie-breaking.
IDs are `exchange_three:m1-m2-m2` for a sorted exchange triple,
`choose_missing_suit:m` for a declaration, `TYPE:CODE` for tile actions
(for example `discard:p4`, `pong:p5`, or `concealed_kong:s7`), and `win` /
`pass` for parameterless actions. They are unique within each decision.
The returned ID maps to exactly one original Rust action. Candidate types use
ninone names, including `exchange_three`, `choose_missing_suit`, `discard`,
`concealed_kong`, `added_kong`, `win`, `pong`, `kong`, and `pass`.

The adapter never removes peng or kong options to imitate the heuristic's
preferences. Its preference for discards over kongs and passes over claims
belongs to the imported policy itself. Worker stdout is reserved for protocol
responses; diagnostics go to stderr. Each complete request/response exchange,
including pipe writes, has a 60-second timeout by default. Set
`POLICY_AB_WORKER_TIMEOUT_SECS` to a positive integer number of seconds to
override it. Timeout invalidates the run and kills the worker; it is a failure
safety bound, not a policy time budget or part of score identity.

## Pilot, frozen plan, and main run

Choose a new work directory under `experiments/policy_ab/runs/` (ignored by
Git). `plan`, `main`, and `summarize` require an explicit `--work-dir`;
`pilot` and `run` may instead create a new directory automatically. Do not
store generated output among source files. The staged workflow lets you
inspect the pilot before authorizing a main sample:

```sh
# Bash example; use the .exe driver path on Windows.
DRIVER=target/release/examples/policy_ab
RUN=experiments/policy_ab/runs/my-experiment

python experiments/policy_ab/experiment.py pilot --driver "$DRIVER" --work-dir "$RUN" --pilot-size 8 --seed-procedure-seed 20261102

# Set TARGET_HALF_WIDTH yourself, in score units, after inspecting the pilot.
python experiments/policy_ab/experiment.py plan --driver "$DRIVER" --work-dir "$RUN" --target-precision "$TARGET_HALF_WIDTH" --bootstrap-seed 20261103 --bootstrap-resamples 10000

# Inspect plan.json: sample size, main seeds, limitations, and component identity.
# Only then launch the complete frozen sample. Settings are read from plan.json;
# explicitly repeated settings must match the plan.
python experiments/policy_ab/experiment.py main --driver "$DRIVER" --work-dir "$RUN"

# Recompute from saved records, without replay or the external worker.
python experiments/policy_ab/experiment.py summarize --work-dir "$RUN"
```

Instead of `--target-precision`, choose an explicit
`--runtime-budget SECONDS` for the main sample. You may supply both, and an
explicit `--max-main-seeds` cap; the plan records when a budget or cap limits
predicted precision. At least two pilot seeds and two main seeds are needed.
The precision calculation uses `ceil((1.96 * pilot_sd / half_width)^2)`;
it is a planning approximation, **not** a promise about the eventual
bootstrap width. A zero pilot SD does not establish zero uncertainty.

The seed corpus uses a recorded SplitMix64 procedure, drawing distinct u64
seeds in fixed order. Pilot seeds are the initial prefix; main seeds follow
that prefix and never overlap it. The actual lists, generator seed, bootstrap
seed, and bootstrap resample count are frozen in `plan.json` before main
execution. Pilot scores never enter the main estimate. Do not change the
corpus, add seeds after inspecting the main result, or stop when an interval
first looks favorable. Runtime is diagnostic, not an outcome for the A/B
comparison; pilot runtime does not guarantee the main wall-clock duration.

Use the same driver, worker, interpreter, ninone checkout, and settings for
pilot, planning, and main execution. The pilot's mode and decision limit must
match the frozen main plan. Main execution inherits settings omitted on its
CLI from that plan; explicitly supplied settings must match. Planning likewise
inherits the pilot's mode and decision limit when omitted, and rejects explicit
configuration drift or a seed list inconsistent with the recorded procedure.
Rebuild after changing Rust sources and start a new work directory after
changing experimental components. A frozen
plan must not silently adopt a different implementation. A failed attempt
must retain its evidence, not be overwritten by another attempt. A separately
authorized retry belongs in a new work directory with the original failure
retained.

The workflow retains `corpus.json`, pilot reports/results, `plan.json`,
`main/results.jsonl`, `main/run.json`, `manifest.json`, and derived
`summary.json` / `summary.md`. Before launching a run, it saves repository
captures and relevant source bytes under `evidence/`; the manifest uses those
pre-run captures rather than a later working tree. A pre/post identity check
invalidates pilot or main execution if experimental sources or repository
revisions change during the run. Do not edit components during a run, even if
you intend to restore them later.

The manifest records repository revisions and local changes, patches and
eligible untracked source snapshots, hashes of
Rust build inputs, the executed driver, worker, analysis script and Python
interpreter, actual seeds, analysis settings, invocation, and environment.
Keep the **entire run directory**, including provenance snapshots, rather
than only the summary. Failed main diagnostics remain in `main/failures.json`;
a failed recomputation writes `main/summarize_failures.json` without overwriting the
original failure. `summarize` accepts only `--work-dir` and reads analysis
settings from the frozen plan; it cannot silently replace the recorded
bootstrap configuration.

Build-order freshness is evidence, not a proof that an arbitrary executable
was built from those sources; build the documented
example yourself and retain the exact executable if binary-level replay is
needed. No performance or strategy-quality conclusion follows from the
smoke tests.

## Estimator

For seed `s` and placement `c`, average the two A scores and the two B scores
separately, and take their difference `d[s,c]`. Average all six differences to
obtain the seed-block contrast `D[s]`. The reported effect `Delta` is the mean
of `D[s]` over seeds. Policy means give equal weight to placements and seeds.

The sampling unit is the **whole seed block**, not one player or one game.
Standard error is `sample_sd(D) / sqrt(n)`. The 95% percentile bootstrap
resamples whole blocks with a separately recorded bootstrap seed and resample
count. At least two seeds are needed for this inferential summary. Zero-sum
settlement implies `B_mean = -A_mean` and `Delta = 2 * A_mean`; these are
consistency checks, not extra observations.

A replay of a fixed corpus has no sampling uncertainty. The interval concerns
seeds generated by the recorded sampling procedure, not replay variability.
Do not use an independent two-sample test or treat the six shared-deal games
as independent replicates.

## Failure contract and raw driver

A seeds file is a JSON array of distinct u64 integers. The raw driver interface
is useful for checks, not a replacement for the evidence-recording workflow:

```text
target/release/examples/policy_ab --seeds seeds.json --output games.jsonl --python python --worker experiments/policy_ab/worker.py --ninone-root ../ninone-mahjong-agent --mode mixed --decision-limit 10000
```

Use `.exe` on Windows. Modes are `mixed`, `first-only`, and `heuristic-only`.
The latter two still label the same six A pairs for the symmetry check, while
all four seats actually use the same policy.

Every successful game records `seed`, `a_seats`, four seat-ordered `scores`,
`end_reason` (`three_winners` or `wall_exhausted`), accepted `decisions`, and a
`trace_hash`. The trace hash is a versioned FNV-1a 64-bit identity of mode,
seed, placement, and ordered seat/action submissions (see the Rust driver's
encoding); it is a replay check, not a cryptographic signature. Timing is
diagnostic, not deterministic result identity. Complete results must contain
all six placements for every planned seed, zero-sum terminal
scores, and positive decision counts.

Protocol/process failures, unknown selections, illegal actions, and decision
limit exhaustion are fatal. The driver exits nonzero and retains a failure
record with seed, placement, decision index, and decision context. Partial
successful records are diagnostic only. Never replace a failed choice with B,
score an incomplete game as zero, omit a failed seed, or publish a normal
summary for an incomplete run. Existing result paths must not be overwritten.

## Verification

From the repository root:

```text
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
python -m unittest discover -s experiments/policy_ab -v
```

The end-to-end tests are opt-in so ordinary unit tests do not require an
external checkout or build. Build the example first, then set
`POLICY_AB_DRIVER` to its executable path and optionally
`POLICY_AB_NINONE_ROOT` to the external checkout. For example, in Bash:

```sh
POLICY_AB_DRIVER=target/release/examples/policy_ab python -m unittest discover -s experiments/policy_ab -p test_integration.py -v
```

In PowerShell, set `$env:POLICY_AB_DRIVER` to
`target/release/examples/policy_ab.exe`, then run the same unittest command.
The integration checks run two-seed smoke corpora for complete coverage,
zero-sum settlement, deterministic replay, all-B symmetry, all-A symmetry,
and fatal unknown-selection / decision-limit failures. Both symmetric modes
must have zero contrasts after six-placement seat averaging. They do not
replace focused candidate enumeration, ID/order, and action-mapping tests.

These checks establish harness properties, **not** an experimental conclusion
that A outperforms B. A main run still needs a preselected precision target or
runtime budget and a frozen sampling plan.
