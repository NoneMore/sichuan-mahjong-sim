# Historical validation and research outcomes

This page records **point-in-time** validation and experimental conclusions,
not the present test count, CI status, or performance/strategy-strength
guarantee. For current build, test, Rust 1.75 MSRV and supported experiments
see [development](development.md) and
[CI](../.github/workflows/rust-ci.yml). Code and tests are the authority on
implemented behavior; [architecture](architecture/overview.md) describes
the implemented subsystem boundaries.

Detailed original measurements and methods remain in the
[pre-retirement snapshot](https://github.com/NoneMore/sichuan-mahjong-sim/tree/f57e48fdd39a9571371194b46d61bfba2b20d795).
The snapshot also contains completed `specs/archive/` Changes and historical
`docs/studies/` reports. Neither is a maintained description of the current
implementation. Earlier locally deleted raw samples, binaries, traces,
prototypes and scripts are **not** restored by the snapshot.

## Game and runner baseline

Initial verification at the original 1.98 Windows toolchain milestone passed
**113 Debug and Release tests**, covering hand shapes, fixed fan/payment facts,
256 exhaustive-draw seat classifications, response/claim arbitration, illegal
submission atomicity, seeded replay and checkpoint continuation. A further
**256 seeded game** stress corpus checked physical tile ownership and zero-sum
score invariants after accepted transitions. These counts describe the
original milestone, not today's suite.

The later decision-hot-path change reused the Game's legal-action buffer and
avoided repeating a winning-shape check for passed-Hu handling without
changing action semantics. A fixed 512-game cross-version replay matched
byte-for-byte. Local paired 100,000-game batches observed median throughput
increases of **10.36%** (FirstLegal) and **8.88%** (Random), without a
cross-machine guarantee. The original
[validation record](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/docs/validation.md#issue-1-decision-hot-path-buffer-and-legality-reuse)
preserves the exact environment, workload, measurements and coverage.

## Strategy comparisons

The early shanten/ukeire implementation milestone compared the heuristic
against FirstLegal and Random over just **four seeds / 24 games per contrast**.
The observed advantage was **+4.375** terminal net score units per player per
game in each contrast. Those small exploratory sets do not establish general
playing strength. [Original study](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/docs/studies/shanten-ukeire.md).

The subsequent balanced missing-suit ablation used **4,000 complete seed
blocks / 24,000 games per contrast**, 48,000 games altogether. With positive
values favoring the policy in B:

| Contrast (B over A) | Mean net score advantage per player/game | Seed-block SE | Descriptive 95% interval |
| --- | ---: | ---: | --- |
| MissingOnly over FirstLegal | **+4.026667** | 0.082964 | [3.864058, 4.189275] |
| Heuristic over MissingOnly | **+2.962542** | 0.028990 | [2.905720, 3.019363] |

The missing-suit choice contributed meaningfully in the tested mixtures,
but did not account for the full heuristic advantage. The two contrasts
have different opponent mixtures and **must not be added**; convenience
seeds and descriptive intervals do not demonstrate population
representativeness. Six seat placements sharing a seed are not independent
observations. [Original protocol, results and provenance](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/docs/studies/missing-only-ablation.md).

## Exact-analysis changes and profiling

The following milestones were **implemented and verified at their own
revisions**. Their correctness requirements and current behavior live in
source and permanent regression tests, not in old benchmark percentages.

- **Seven-pairs formula and A2 suit-table pruning:** adopted after
  independent exact-target/table comparisons, cache checks and seeded replay.
  Small release-batch performance differences were local observations only.
  [Original validation](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/docs/validation.md#heuristic-hand-analysis-optimization).
- **One-pass ukeire:** replaced per-hypothetical-draw distance recomputation
  with an exact optimal-target needed mask, preserving all tied mask bits.
  Historical paired 2,000-game comparisons observed approximately 62%
  total/policy time reductions in both seed groups; no present-day speedup
  is implied. [Results and proof](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/specs/archive/one-pass-ukeire/results.md).
- **One-pass discard:** derived exact candidate deficits from original-hand
  surplus-mask analysis without changing action selection or checkpoint
  semantics. [Original verification and timing](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/specs/archive/one-pass-discard/results.md).
- **Bounded shared masked-DP cache:** adopted exact byte-bounded reuse across
  decisions with disabled and allocation-failure fallbacks. The selected
  default was 26.5 MiB. Large-batch improvements and cold-start slowdowns
  were both observed in the original acceptance protocol; neither is a
  general performance promise. [Original results](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/specs/archive/bounded-masked-dp-cache/results.md).
- **Bounded Needed-mask DP:** retained a strict `g > d` cutoff with
  equal-cost optimal-mask paths intact. [Original acceptance](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/specs/archive/bounded-exact-needed-dp/results.md).

The initial, intermediate and post-discard CPU attribution studies measure
**different historical implementations**; their old hotspot rankings
must not be projected onto today's code. In the post-discard `06fb919`
snapshot, masked per-suit DP represented **79.09%** of sampled worker
cost (needed **57.25%**, surplus **21.84%**) and plain distance DP **3.50%**.
Sample proportions are not speedup predictions. For the full methods,
revision boundaries and evidence limits, see the
[initial](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/docs/studies/heuristic-cost-attribution.md),
[intermediate](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/docs/studies/heuristic-cost-attribution-refresh.md),
and [post-discard](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/docs/studies/heuristic-cost-attribution-discard.md)
reports and the committed
[measurement summary](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/docs/studies/heuristic-cost-attribution-refresh/measurements.json).

## Recovery and new measurements

The studies ran primarily on Windows 10 / Ryzen 7 3700X with Rust 1.98.
Many original samples, EXE/PDB files, local source snapshots and experimental
tools were intentionally removed before retirement. The committed records
document their limits; they do **not** let Git reconstruct missing raw data.
Other accepted, rejected and still-undecided experiment candidates and their
original branch SHA values are catalogued in
[historical retirement PR #23](https://github.com/NoneMore/sichuan-mahjong-sim/pull/23).
A correct-but-slow or inconclusive experiment must not be presented as an
accepted optimization.

To remeasure, check out the specified historical source revision, rebuild it
with the documented toolchain and workload, save new evidence and compare
controlled baseline/candidate runs. Current CI or a new benchmark cannot
retroactively validate the old numerical samples.
