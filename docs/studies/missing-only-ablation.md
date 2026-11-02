# Missing-suit ablation: 1000x A/B study (2026-11-02)

## Policy isolation and validation

`MissingOnlyPolicy` delegates every decision to `FirstLegalPolicy` except the
missing-suit declaration, which calls the exact helper used by
`HeuristicPolicy`: choose the fewest-tile suit after exchange, breaking ties by
lowest suit index. FirstLegal instead takes the first legal declaration, Wan.
All three policies already share exchange selection; MissingOnly and Heuristic
also share the declaration rule. Their remaining differences are discard
ranking and peng/kong gates, not exchange logic or game rules.

The added tests compare real decisions against the appropriate reference over
32 seeded games, check legal choices and zero-sum results, and verify replay
and continuation from a cloned game with fresh policies. Focused cases cover
suit counts, ties, fallback, exchange, turns, and Hu preference. Response and
claim opportunities are exercised, but the combined response counter does not
prove coverage of every response subtype. Independent static review found no
production-code isolation or checkpoint defects.

Formatting, all-target Clippy with warnings denied, and warning-free rustdoc
passed. All **186 tests** passed in both debug and release (130 library, 6 CLI,
19 integration, 31 A/B example). No rules or settlement behavior changed.

## Workload, budget, and measurement conditions

- The earlier implemented experiment used four seeds and six balanced 2A2B
  placements per seed: **24 games per contrast**. This study used seeds
  **1 through 4000**, all six placements per seed: **24,000 games per contrast**,
  exactly 1000x, or **48,000 main games** across the two requested contrasts.
- A occupies `[0,1] [0,2] [0,3] [1,2] [1,3] [2,3]`; B occupies the other seats.
  Every placement uses a fresh game, fresh seat policies, and fresh runner.
  Policy seed is 0 (ignored by these deterministic policies); per-game decision
  limit is 10,000. No seed blocks or outliers were dropped.
- The two contrasts and seed range were fixed before viewing their outcomes.
  Forty-seed pilots (240 games each, not added to the main sample) checked the
  new build and cost; no policy tuning or outcome-dependent scale change was
  performed.
- Budget: one hour. Historical heuristic throughput suggested roughly two
  minutes for the main workload. A pre-implementation release build measured
  2400 Heuristic/FirstLegal games in 10.623 s and 24,000 FirstLegal/FirstLegal games
  in 1.762 s. The new-policy pilots took 0.248 s and 1.205 s. No downscaling was
  necessary.
- Source: base revision `6f726dcf6c94e27e8975f32e39fdb9824845e8a0` plus the
  uncommitted MissingOnly implementation. A local artifact snapshot captured
  the measured tracked diff and new source files. Later review fixes changed
  comments/documentation only. The generated snapshot was subsequently removed
  during cleanup; results, hashes, and reproduction commands remain below.
- AMD Ryzen 7 3700X, 16 logical CPUs; Windows 10 build 19045 x64;
  `x86_64-pc-windows-msvc`; rustc 1.98.0 (`88d9e12ae`), Cargo 1.98.0.
  Default release profile; one single-threaded simulator process at a time,
  without affinity pinning or overlapping builds/tests. Timings are external
  subprocess wall clocks including metadata collection and artifact I/O, each
  measured once after the pilots, not a speed guarantee.

## Scores and descriptive uncertainty

Means are terminal net score units **per player per game**. The existing A/B
example prints `Delta = A_mean - B_mean`; the improvement below is the opposite
sign, **B minus A**, so positive favors the upgrade. For each seed, average the
six placement contrasts first; `SE = sample_sd(seed contrasts) / sqrt(4000)`.
Intervals are descriptive normal 95% intervals, improvement +/- `1.96 * SE`.

| Metric | A = FirstLegal, B = MissingOnly | A = MissingOnly, B = Heuristic |
|---|---:|---:|
| Seed blocks / games | 4000 / 24,000 | 4000 / 24,000 |
| A mean | -2.013333 | -1.481271 |
| B mean | +2.013333 | +1.481271 |
| B minus A | **+4.026667** | **+2.962542** |
| Seed-block SE | 0.082964 | 0.028990 |
| Descriptive 95% interval | [3.864058, 4.189275] | [2.905720, 3.019363] |
| Decisions | 1,750,866 | 1,700,133 |
| Three-winner / wall-exhaustion endings | 592 / 23,408 | 8790 / 15,210 |
| Terminal score totals by seat | `[7119, 4202, -547, -10774]` | `[2715, 550, -263, -3002]` |
| Aggregate score sum | 0 | 0 |
| Main subprocess wall time | 2.014 s | 105.134 s |

The two main subprocesses took **107.148 s combined**, confirming the scale
estimate well within the hour. Independent recomputation from both persisted
TSVs matched the example's means, SE, and all 4000 per-seed D values per
contrast within six-decimal rounding. Checks confirmed exactly six unique
placements for each expected seed, zero-sum scores in every game, and positive
decision counts within the limit. A seed-1 six-placement replay matched the
first contrast's raw records byte-for-byte.

**Interpretation:** replacing the arbitrary FirstLegal declaration with the
fewest-tile declaration has substantial value in this balanced mixed-policy
setting. Heuristic still has a substantial advantage over that ablation, so its
benefit is not solely the declaration rule. These are two different opponent
mixtures: the contrasts cannot be added or divided into a percentage attribution
of a direct FirstLegal-to-Heuristic gap. Consecutive convenience seeds and their
block SE do not establish population representativeness. Six placements within
a seed are dependent, not six independent samples. Scores and global ending
counts do not establish policy win rates or identify particular settlement
payments as the mechanism.

## Reproduction and artifact provenance

Build the measured policies and expand the seed list through subprocess argv
rather than a shell; the 4000-seed argument is 18,892 characters, below the
Windows process command-line limit. Use a fresh output parent/name:

```text
cargo build --release --example policy_ab --lib
```

```python
from pathlib import Path
import subprocess

root = Path("target/missing-only-reproduction")
root.mkdir()  # refuse an existing study rather than overwrite it
seeds = ",".join(map(str, range(1, 4001)))
for a, b in [("first", "missing-only"), ("missing-only", "heuristic")]:
    subprocess.run([
        str(Path("target/release/examples/policy_ab.exe").resolve()),
        "--seeds", seeds, "--policy-seed", "0", "--decision-limit", "10000",
        "--out", str(root / f"{a}-vs-{b}"), "--a", a, "--b", b,
    ], check=True)
```

Generated artifacts were written under `target/missing-only-study-20261102/`
(ignored by Git). They were intentionally removed during post-study cleanup,
including raw records, temporary scripts, logs, and the source snapshot. The
following inventory and hashes document the original outputs, not files still
available in the checkout; the results above and reproduction commands are
retained in the repository:

- `analysis-plan.json`: predeclared workload, contrasts, uncertainty convention,
  and budget; `analysis.json` and `per-seed-stats.tsv`: recomputed results.
- `artifacts/ab-first-vs-missing-only/` and
  `artifacts/ab-missing-only-vs-heuristic/`: `run.txt`, `games.tsv`, `summary.txt`.
- `driver.py`, `analysis.py`, `driver-timings.json`, and `logs/`: execution,
  independent analysis, timings, stdout/stderr, and validation evidence.
- `snapshot/`: base revision, measured source diff/new files, environment, and
  SHA-256 manifest. The measured example executable SHA-256 is
  `5f2e7982cbcff57b0b32c9a417e1f8ecf81a343572b3224f9d011fadaedaa855`.
- Raw `games.tsv` SHA-256, first/MissingOnly:
  `37618ac3b23922f18e5b319f97b2ee32dafe1478c1ea59db9e03a2dc3ec68617`;
  MissingOnly/heuristic:
  `7d3d6983d4b7f131a184a900ba4ec81e007c692cf1daa7f4123c8ca54a5d41d1`.
