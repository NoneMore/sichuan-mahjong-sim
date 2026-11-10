# Python-driven neural policy evaluation specification

## Outcome and scope

Run complete games with frozen PyTorch policies against `HeuristicPolicy`. This
Change provides the Python binding, synchronous single-process multi-game
inference, and comparison records. Reuse the [offline BC package](../../../python/README.md);
game rules, teacher behavior, and encoding remain unchanged.
[Design](design.md) covers structure; [tasks](tasks.md) covers delivery.

Preserve the Rust 1.75 baseline and keep core/collector builds and offline training
independent of bindings. Evaluation requires only a checkpoint, not its corpus,
training cache, or run directory. RL transitions/training, parallel workers, async
actors, live weights, and environment recovery are outside scope.

## Environment

One handle owns one game. Explicit reset starts a fresh seeded game and returns
its first pending decision. Step accepts a decision identity and local candidate
index, returning the next decision or terminal result.

- Pending decisions expose identity, actor seat, stage, and owned float32 state/
  candidate arrays matching the existing encoder and canonical order. Python array
  mutation cannot alter game state or stored actions.
- Every seat and stage, including responses and forced choices, remains explicit.
  Each step commits one choice and advances Game's automatic transitions.
- Stale identities, including those from before reset, and out-of-range indices
  are rejected before mutation. Rust resolves indices to semantic actions.
- Model inputs contain only encoded player-visible features. Identity and seat
  route decisions; opponents' concealed hands and wall order remain hidden.
- Terminal results include four seat-ordered scores, winners, end reason, and
  accepted-decision count, including forced choices. One seat winning does not
  end the game; reset remains explicit after termination.
- Rule errors and decision-limit exhaustion stop execution with seed/decision
  context, distinct from normal termination.

## Frozen inference and teacher

Use the offline checkpoint contract and reject unsupported checkpoint/model/
encoding versions before play. Evaluate policy weights without restoring training
state or imposing training-resume identity checks.

Use eval mode without gradients, float32 inputs, and a Boolean candidate mask.
Padding preserves valid scores and canonical order. Masked greedy selection uses
canonical-first ties; non-finite valid logits stop evaluation. A single candidate
is submitted as index zero without inference, retaining counts and traces. Online
inputs need no supervised labels. Batches group decisions by assigned policy and
route indices back to their originating handle/decision.

The native teacher chooses on the original Rust Decision and returns its canonical
index without advancing the game. Step commits the choice. Cache use preserves
teacher choices.

## Comparison and records

For each seed, start six fresh games with neural seats in pairs `[0,1]`, `[0,2]`,
`[0,3]`, `[1,2]`, `[1,3]`, `[2,3]` and heuristic policies elsewhere. This is a
free-for-all comparison. Primary strength observation is mean terminal net score
per neural seat; also report score spread, finish reasons, and decision counts.
Estimate uncertainty over seed blocks.

Use the existing corpus seed split: validation buckets for repeated diagnostics,
test buckets for frozen final comparison. Offline validation NLL selects the
checkpoint; online scores do not select it.

Persist the plan before play, raw seed/placement and terminal records as games
complete, and derive summaries from those records. Capture:

- checkpoint content identity, role, format versions, and model configuration;
- actual Python policy/evaluator source and compiled Rust runtime source, separate
  from training provenance when available;
- encoding/model versions, split algorithm, seeds and placements;
- selection mode, device/batching, decision limit, teacher cache, and dependencies.

Source identity follows the installed package/compiled binary, not the process
working directory. Game, inference, output, or record-consistency failure stops
the run, retaining completed records without publishing a complete summary or
substituting actions.

## Acceptance

- Environment tests cover all stages, forced choices, terminal/reset behavior,
  request rejection without mutation, snapshot ownership, and observation isolation.
- Offline/online fixtures match feature values and candidate order. Same-seed
  native/Python teacher replay agrees on semantic actions, counts, and terminal
  results under the same Rust runtime.
- Inference tests cover routing, padding, partial batches, ties, non-finite logits,
  forced bypass, checkpoint compatibility, and independence from training data/run
  directories.
- A small checkpoint-driven comparison completes six placements per seed with
  recorded identities and independently recomputable seed-grouped summaries.
  Injected failures verify record retention and incomplete-run behavior.
- Build/install checks preserve core/collector MSRV and offline independence.
  Record environment, teacher, and inference costs before choosing CPU/GPU batching
  settings or claiming throughput.

Whether to expose diagnostic `last.pt` evaluation remains undecided; it is not
required for formal comparison. If offered, evaluate its policy weights rather
than embedded best weights.
