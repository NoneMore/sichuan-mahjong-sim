"""Unit and smoke tests for the policy A/B experiment orchestration.

These tests use synthetic driver output and never start a real policy
experiment: they cover statistics, seed isolation, output completeness and
failure handling, plan freezing, manifest provenance, symmetric checks, and a
small end-to-end smoke run through a fake driver.

Run with::

    python -m unittest discover -s experiments/policy_ab -p "test_experiment.py" -v
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import experiment as ex  # noqa: E402

ANALYSIS = {
    "bootstrap_seed": 99,
    "bootstrap_resamples": 200,
    "confidence": 0.95,
}
FAKE_DRIVER = '''\
import argparse
import hashlib
import json
import os
import sys

PLACEMENTS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def trace_hash(mode, output_path, seed, index):
    if mode == "trace_by_path":
        material = f"{output_path}|{seed}|{index}".encode("utf-8")
    else:
        material = f"{seed}|{index}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    for flag in (
        "--seeds",
        "--output",
        "--python",
        "--worker",
        "--ninone-root",
        "--mode",
        "--decision-limit",
    ):
        parser.add_argument(flag)
    args = parser.parse_args()
    mode = os.environ.get("FAKE_DRIVER_MODE", "ok")
    stage = os.path.basename(os.path.dirname(os.path.abspath(args.output)))
    fail_here = mode == "exit" and stage == "main"
    fail_after_records = mode == "exit_after_records" and stage == "main"
    mutate_sources = mode == "mutate_sources" and stage == "main"
    mutate_pilot = mode == "mutate_sources_pilot" and stage == "pilot"
    failure_record_here = mode == "failure_record" and stage == "main"
    if mutate_sources or mutate_pilot:
        # Simulate a source/repository change while the driver is executing.
        with open(args.worker, "a", encoding="utf-8") as handle:
            handle.write("# mutated during the run\\n")
        package = os.path.join(args.ninone_root, "ninone")
        os.makedirs(package, exist_ok=True)
        with open(
            os.path.join(package, "agent_heuristic.py"), "w", encoding="utf-8"
        ) as handle:
            handle.write("changed_during_run = True\\n")
    with open(args.seeds, "r", encoding="utf-8") as handle:
        seeds = json.load(handle)
    records = []
    for seed in seeds:
        for index, placement in enumerate(PLACEMENTS):
            spread = 2 + int(seed) % 4 + index
            scores = [spread if seat in placement else -spread for seat in range(4)]
            records.append(
                {
                    "seed": int(seed),
                    "a_seats": list(placement),
                    "scores": scores,
                    "end_reason": "wall_exhausted" if seed % 2 else "three_winners",
                    "decisions": 5 + int(seed) % 3,
                    "trace_hash": trace_hash(mode, args.output, seed, index),
                }
            )
    if fail_here:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(records[0]) + "\\n")
        print("fake driver failure", file=sys.stderr)
        sys.exit(3)
    with open(args.output, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\\n")
        if failure_record_here:
            handle.write(
                json.dumps(
                    {
                        "seed": int(seeds[0]),
                        "a_seats": [0, 1],
                        "error": "worker_protocol_error",
                        "detail": "unknown selection id",
                    }
                )
                + "\\n"
            )
    if fail_after_records:
        print("fake driver failure after complete records", file=sys.stderr)
        sys.exit(4)


main()
'''


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def valid_record(seed: int, placement, *, scores=None, end_reason="wall_exhausted", decisions=7):
    return {
        "seed": seed,
        "a_seats": list(placement),
        "scores": list(scores if scores is not None else _zero_sum_scores(placement)),
        "end_reason": end_reason,
        "decisions": decisions,
    }


def _zero_sum_scores(placement):
    return [4 if seat in placement else -4 for seat in range(4)]


def complete_records(seed: int, *, scores_per_seed=None, end_reason="wall_exhausted", decisions=7):
    return [
        valid_record(
            seed,
            placement,
            scores=scores_per_seed,
            end_reason=end_reason,
            decisions=decisions,
        )
        for placement in ex.PLACEMENTS
    ]


def synthetic_results(block_values, *, end_reason="wall_exhausted", decisions=7):
    """Complete game results whose seed-block contrast equals each block value.

    Scores are floats here (statistics tests only); zero-sum holds exactly.
    """
    results = []
    for seed, value in block_values.items():
        half = value / 2.0
        for placement in ex.PLACEMENTS:
            scores = tuple(half if seat in placement else -half for seat in range(4))
            results.append(
                ex.GameResult(
                    seed=seed,
                    placement=placement,
                    scores=scores,
                    end_reason=end_reason,
                    decisions=decisions,
                )
            )
    return results


def grouped(results):
    return ex._group(results)


def plan_for_blocks(values, *, mode="mixed", target_precision=None, max_main_seeds=None):
    """Build a frozen plan whose main blocks correspond to ``values``."""
    block_values = list(values.values()) if isinstance(values, dict) else list(values)
    procedure = ex.seed_procedure(4242)
    pilot_seeds = ex.generate_seeds(procedure, 3)
    pilot_report = {
        "seeds": pilot_seeds,
        "count": len(pilot_seeds),
        "mode": mode,
        "decision_limit": 100,
        "block_sd": 1.0,
        "block_mean": 0.5,
        "mean_seconds_per_seed": 1.0,
        "elapsed_seconds": float(len(pilot_seeds)),
        "adapter_identity": {"fingerprint": "test-fingerprint"},
    }
    plan = ex.build_plan(
        pilot_report=pilot_report,
        procedure=procedure,
        mode=mode,
        decision_limit=100,
        target_precision=target_precision,
        runtime_budget=float(len(block_values)),
        max_main_seeds=max_main_seeds,
        analysis=dict(ANALYSIS),
        adapter_identity={"fingerprint": "test-fingerprint"},
    )
    seeds = plan["sample"]["main_seeds"]
    assert len(seeds) == len(block_values), (len(seeds), len(block_values))
    return plan, dict(zip(seeds, block_values))


def summary_for_blocks(values, *, mode="mixed", target_precision=None, end_reason="wall_exhausted"):
    plan, blocks = plan_for_blocks(values, mode=mode, target_precision=target_precision)
    results = synthetic_results(blocks, end_reason=end_reason)
    evaluation = ex.RunEvaluation(
        ok=True, results=results, problems=[], by_seed=grouped(results)
    )
    return ex.build_summary(
        plan=plan,
        evaluation=evaluation,
        timing={"elapsed_seconds": 1.0, "mean_seconds_per_seed": 0.5},
        deterministic_replay={"status": "verified"},
    )


def run_git(*args, cwd):
    completed = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {completed.stderr}")
    return completed.stdout.strip()


def init_git_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    run_git("init", "-q", cwd=root)
    run_git("config", "user.email", "test@example.com", cwd=root)
    run_git("config", "user.name", "Test", cwd=root)
    run_git("config", "commit.gpgsign", "false", cwd=root)


def fake_driver(tmp: Path) -> Path:
    path = tmp / "fake_driver.py"
    path.write_text(FAKE_DRIVER, encoding="utf-8")
    return path


def fake_environment(tmp: Path) -> dict[str, Path]:
    """Minimal repo layout for identity capture during smoke runs."""
    repo = tmp / "sim-repo"
    (repo / "src").mkdir(parents=True, exist_ok=True)
    (repo / "examples").mkdir(parents=True, exist_ok=True)
    (repo / "Cargo.toml").write_text("[package]\nname = \"fake\"\n", encoding="utf-8")
    (repo / "Cargo.lock").write_text("# lock\n", encoding="utf-8")
    (repo / "src" / "lib.rs").write_text("pub fn x() {}\n", encoding="utf-8")
    (repo / "examples" / "policy_ab.rs").write_text("fn main() {}\n", encoding="utf-8")
    ninone = tmp / "ninone"
    package = ninone / "ninone"
    package.mkdir(parents=True, exist_ok=True)
    (package / "agent_heuristic.py").write_text(
        "def choose_action(observation, actions):\n    return actions[0]\n",
        encoding="utf-8",
    )
    worker = tmp / "worker.py"
    worker.write_text("# fake worker\n", encoding="utf-8")
    return {"repo": repo, "ninone": ninone, "worker": worker}


def run_args(work: Path, tmp: Path, env: dict[str, Path], driver: Path, *extra: str):
    return [
        "run",
        "--work-dir",
        str(work),
        "--driver",
        str(driver),
        "--repo-root",
        str(env["repo"]),
        "--ninone-root",
        str(env["ninone"]),
        "--worker",
        str(env["worker"]),
        "--python",
        sys.executable,
        *extra,
    ]


# ---------------------------------------------------------------------------
# Seed corpus
# ---------------------------------------------------------------------------


class TestSeedCorpus(unittest.TestCase):
    def test_corpus_is_deterministic_and_distinct(self):
        procedure = ex.seed_procedure(7)
        first = ex.generate_seeds(procedure, 64)
        second = ex.generate_seeds(procedure, 64)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 64)
        self.assertTrue(all(0 <= seed <= ex.U64_MAX for seed in first))

    def test_pilot_and_main_do_not_overlap(self):
        procedure = ex.seed_procedure(1234)
        corpus = ex.generate_seeds(procedure, 12)
        main = ex.plan_main_seeds(procedure, 5, 7)
        self.assertEqual(main, corpus[5:])
        self.assertFalse(set(corpus[:5]) & set(main))
        self.assertEqual(len(main), 7)

    def test_different_master_seeds_give_different_corpora(self):
        first = ex.generate_seeds(ex.seed_procedure(1), 8)
        second = ex.generate_seeds(ex.seed_procedure(2), 8)
        self.assertNotEqual(first, second)

    def test_rejects_unknown_generator(self):
        with self.assertRaises(ex.ExperimentError):
            ex.generate_seeds({"generator": "python-random", "master_seed": 1}, 3)

    def test_index_is_bounded_uniform_sampling(self):
        rng = ex.SplitMix64(12345)
        for bound in (1, 2, 3, 5, 7, 10**18, ex.U64_MAX):
            values = [rng.index(bound) for _ in range(200)]
            self.assertTrue(all(0 <= value < bound for value in values))
            if bound <= 5:
                self.assertEqual(set(values), set(range(bound)))
        replay = ex.SplitMix64(12345)
        reference = ex.SplitMix64(12345)
        self.assertEqual(
            [replay.index(10**18) for _ in range(20)],
            [reference.index(10**18) for _ in range(20)],
        )
        self.assertEqual(ex.SplitMix64(9).index(1), 0)

    def test_prepare_corpus_reuses_and_verifies_artifact(self):
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            procedure = ex.seed_procedure(5)
            seeds = ex.prepare_corpus(work, procedure=procedure, pilot_size=4)
            self.assertEqual(len(seeds), 4)
            self.assertEqual(
                ex.prepare_corpus(work, procedure=procedure, pilot_size=4), seeds
            )
            with self.assertRaises(ex.ExperimentError):
                ex.prepare_corpus(
                    work, procedure=ex.seed_procedure(6), pilot_size=4
                )

    def test_pilot_needs_two_seeds(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(ex.ExperimentError):
                ex.prepare_corpus(
                    Path(raw), procedure=ex.seed_procedure(1), pilot_size=1
                )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestContrastStatistics(unittest.TestCase):
    def test_known_two_block_sample(self):
        stats = ex.compute_contrast_stats(
            grouped(synthetic_results({10: 3.0, 20: 1.0})), [10, 20]
        )
        self.assertEqual(stats.block_contrasts, (3.0, 1.0))
        self.assertAlmostEqual(stats.delta, 2.0)
        self.assertAlmostEqual(stats.a_mean, 1.0)
        self.assertAlmostEqual(stats.b_mean, -1.0)
        self.assertAlmostEqual(stats.a_mean - stats.b_mean, 2.0)
        self.assertAlmostEqual(stats.sample_sd, math.sqrt(2.0))
        self.assertAlmostEqual(stats.standard_error, 1.0)

    def test_sample_sd_known(self):
        self.assertAlmostEqual(ex.sample_sd([1, 2, 3, 4]), math.sqrt(5.0 / 3.0))
        with self.assertRaises(ValueError):
            ex.sample_sd([1.0])

    def test_placements_weighted_equally(self):
        by_seed = {}
        for index, placement in enumerate(ex.PLACEMENTS):
            value = float(index)
            half = value / 2.0
            scores = tuple(half if seat in placement else -half for seat in range(4))
            by_seed.setdefault(1, {})[placement] = ex.GameResult(
                seed=1, placement=placement, scores=scores, end_reason="wall_exhausted", decisions=1
            )
        second = synthetic_results({2: 2.5})
        for result in second:
            by_seed.setdefault(2, {})[result.placement] = result
        stats = ex.compute_contrast_stats(by_seed, [1, 2])
        self.assertAlmostEqual(stats.block_contrasts[0], 2.5)
        self.assertAlmostEqual(stats.block_contrasts[1], 2.5)

    def test_requires_two_blocks(self):
        with self.assertRaises(ex.SummaryError):
            ex.compute_contrast_stats(grouped(synthetic_results({1: 1.0})), [1])

    def test_missing_placement_is_an_error(self):
        results = synthetic_results({1: 1.0, 2: 2.0})
        by_seed = grouped(results)
        del by_seed[2][ex.PLACEMENTS[0]]
        with self.assertRaises(ex.SummaryError):
            ex.compute_contrast_stats(by_seed, [1, 2])


class TestBootstrap(unittest.TestCase):
    def test_bootstrap_is_deterministic(self):
        blocks = [0.5, -0.25, 1.5, 2.0, -0.75]
        first = ex.bootstrap_delta_ci(blocks, resamples=500, seed=123)
        second = ex.bootstrap_delta_ci(blocks, resamples=500, seed=123)
        self.assertEqual(first, second)

    def test_bootstrap_of_constant_blocks_is_constant(self):
        low, high = ex.bootstrap_delta_ci([2.0, 2.0, 2.0], resamples=200, seed=7)
        self.assertAlmostEqual(low, 2.0)
        self.assertAlmostEqual(high, 2.0)

    def test_bootstrap_interval_brackets_sample_mean(self):
        blocks = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        low, high = ex.bootstrap_delta_ci(blocks, resamples=2000, seed=11)
        centre = sum(blocks) / len(blocks)
        self.assertLess(low, centre)
        self.assertGreater(high, centre)
        self.assertLess(low, high)

    def test_percentile_known_values(self):
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(ex.percentile(values, 0.0), 1.0)
        self.assertEqual(ex.percentile(values, 1.0), 4.0)
        self.assertAlmostEqual(ex.percentile(values, 0.5), 2.5)

    def test_bootstrap_rejects_degenerate_inputs(self):
        with self.assertRaises(ex.SummaryError):
            ex.bootstrap_delta_ci([1.0], resamples=10, seed=1)
        with self.assertRaises(ex.ExperimentError):
            ex.bootstrap_delta_ci([1.0, 2.0], resamples=0, seed=1)

    def test_zero_block_check(self):
        stats = ex.compute_contrast_stats(
            grouped(synthetic_results({1: 0.0, 2: 0.0})), [1, 2]
        )
        check = ex.zero_block_check(stats)
        self.assertTrue(check["all_zero"])
        self.assertLessEqual(check["max_abs_block_contrast"], 1e-12)

        stats = ex.compute_contrast_stats(
            grouped(synthetic_results({1: 1.0, 2: 0.0})), [1, 2]
        )
        check = ex.zero_block_check(stats)
        self.assertFalse(check["all_zero"])
        self.assertEqual(check["nonzero_blocks"], [{"seed": 1, "block_contrast": 1.0}])


# ---------------------------------------------------------------------------
# Output validation: completeness, failures, seed isolation
# ---------------------------------------------------------------------------


class TestOutputValidation(unittest.TestCase):
    def _load(self, tmp: Path, records, seeds, name="results.jsonl", raw_lines=None):
        path = tmp / name
        if raw_lines is not None:
            path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
        else:
            write_jsonl(path, records)
        return ex.load_output(seeds, path)

    def test_complete_output_is_valid(self):
        with tempfile.TemporaryDirectory() as raw:
            records = complete_records(11) + complete_records(22)
            evaluation = self._load(Path(raw), records, [11, 22])
            self.assertTrue(evaluation.ok, [p.to_json() for p in evaluation.problems])
            self.assertEqual(len(evaluation.results), 12)
            self.assertEqual(evaluation.by_seed[11][(0, 1)].scores, (4, 4, -4, -4))

    def test_missing_placement_is_preserved(self):
        with tempfile.TemporaryDirectory() as raw:
            records = [r for r in complete_records(11) if tuple(r["a_seats"]) != (1, 3)]
            evaluation = self._load(Path(raw), records, [11])
            kinds = {p.kind for p in evaluation.problems}
            self.assertIn("missing_placement", kinds)
            problem = next(p for p in evaluation.problems if p.kind == "missing_placement")
            self.assertEqual(problem.seed, 11)
            self.assertEqual(problem.placement, (1, 3))

    def test_missing_seed_is_preserved(self):
        with tempfile.TemporaryDirectory() as raw:
            evaluation = self._load(Path(raw), complete_records(11), [11, 22])
            self.assertIn("missing_seed", {p.kind for p in evaluation.problems})
            self.assertIn(22, {p.seed for p in evaluation.problems})

    def test_duplicate_entry_is_invalid(self):
        with tempfile.TemporaryDirectory() as raw:
            records = complete_records(11) + [complete_records(11)[0]]
            evaluation = self._load(Path(raw), records, [11])
            self.assertIn("duplicate_entry", {p.kind for p in evaluation.problems})

    def test_unexpected_seed_is_invalid(self):
        with tempfile.TemporaryDirectory() as raw:
            records = complete_records(11) + complete_records(99)
            evaluation = self._load(Path(raw), records, [11])
            self.assertIn("unexpected_seed", {p.kind for p in evaluation.problems})

    def test_runner_failure_record_invalidates_run(self):
        with tempfile.TemporaryDirectory() as raw:
            records = complete_records(11)
            failure = {
                "seed": 11,
                "a_seats": [0, 2],
                "error": "worker_protocol_error",
                "detail": "unknown selection id",
            }
            evaluation = self._load(Path(raw), records + [failure], [11])
            self.assertFalse(evaluation.ok)
            problem = next(
                p for p in evaluation.problems if p.kind == "runner_failure_record"
            )
            self.assertEqual(problem.seed, 11)
            self.assertEqual(problem.placement, (0, 2))
            self.assertIn("worker_protocol_error", json.dumps(problem.context))

            payload = ex.failures_payload(evaluation, stage="main")
            self.assertFalse(payload["normal_summary_published"])
            self.assertIn(11, payload["failing_seeds"])
            self.assertIn([0, 2], payload["failing_placements"])
            self.assertTrue(payload["preserved_results"])

    def test_malformed_line_is_invalid(self):
        with tempfile.TemporaryDirectory() as raw:
            lines = [json.dumps(record) for record in complete_records(11)]
            lines.append("{not json")
            evaluation = self._load(Path(raw), [], [11], raw_lines=lines)
            self.assertIn("malformed_record", {p.kind for p in evaluation.problems})

    def test_nonzero_score_sum_is_invalid(self):
        with tempfile.TemporaryDirectory() as raw:
            records = complete_records(11)
            records[0]["scores"] = [1, 2, 3, 4]
            evaluation = self._load(Path(raw), records, [11])
            self.assertIn("nonzero_score_sum", {p.kind for p in evaluation.problems})

    def test_invalid_placement_is_invalid(self):
        with tempfile.TemporaryDirectory() as raw:
            records = complete_records(11)
            records[0]["a_seats"] = [0, 4]
            evaluation = self._load(Path(raw), records, [11])
            self.assertIn("invalid_placement", {p.kind for p in evaluation.problems})

    def test_reversed_seat_pair_is_normalized(self):
        with tempfile.TemporaryDirectory() as raw:
            records = complete_records(11)
            target = next(
                record for record in records if record["a_seats"] == [0, 2]
            )
            target["a_seats"] = [2, 0]
            evaluation = self._load(Path(raw), records, [11])
            self.assertTrue(evaluation.ok, [p.to_json() for p in evaluation.problems])
            self.assertIn((0, 2), evaluation.by_seed[11])

    def test_non_integer_scores_are_invalid(self):
        with tempfile.TemporaryDirectory() as raw:
            records = complete_records(11)
            records[0]["scores"] = [1.5, -1.5, 0, 0]
            evaluation = self._load(Path(raw), records, [11])
            self.assertIn("invalid_scores", {p.kind for p in evaluation.problems})

    def test_trace_hash_is_validated_and_preserved(self):
        with tempfile.TemporaryDirectory() as raw:
            records = complete_records(11)
            records[0]["trace_hash"] = "0123456789abcdef"
            evaluation = self._load(Path(raw), records, [11])
            self.assertTrue(evaluation.ok, [p.to_json() for p in evaluation.problems])
            self.assertEqual(
                evaluation.by_seed[11][(0, 1)].trace_hash, "0123456789abcdef"
            )
            self.assertIn("trace_hash", evaluation.results[0].to_json())

        for bad in ("0123456789ABCDEF", "0123", "zzzzzzzzzzzzzzzz", 12):
            with tempfile.TemporaryDirectory() as raw:
                records = complete_records(11)
                records[0]["trace_hash"] = bad
                evaluation = self._load(Path(raw), records, [11])
                self.assertFalse(evaluation.ok)
                self.assertIn(
                    "invalid_trace_hash",
                    {p.kind for p in evaluation.problems},
                )

        with tempfile.TemporaryDirectory() as raw:
            records = complete_records(11)
            records[0]["trace_hash"] = None
            evaluation = self._load(Path(raw), records, [11])
            self.assertTrue(evaluation.ok)
            self.assertIsNone(evaluation.results[0].trace_hash)

    def test_absent_trace_hash_is_allowed(self):
        with tempfile.TemporaryDirectory() as raw:
            evaluation = self._load(Path(raw), complete_records(11), [11])
            self.assertTrue(evaluation.ok)
            self.assertIsNone(evaluation.results[0].trace_hash)
            self.assertNotIn("trace_hash", evaluation.results[0].to_json())

    def test_zero_decisions_is_invalid(self):
        with tempfile.TemporaryDirectory() as raw:
            records = complete_records(11, decisions=0)
            evaluation = self._load(Path(raw), records, [11])
            self.assertIn("invalid_decisions", {p.kind for p in evaluation.problems})

    def test_unknown_end_reason_is_invalid(self):
        for reason in ("hu_win", "three_winners ", "", "WallExhausted"):
            with tempfile.TemporaryDirectory() as raw:
                records = complete_records(11, end_reason=reason)
                evaluation = self._load(Path(raw), records, [11])
                self.assertIn(
                    "invalid_end_reason", {p.kind for p in evaluation.problems}
                )

    def test_failure_marker_with_complete_game_fields_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            records = complete_records(11)
            poisoned = dict(records[0])
            poisoned["error"] = "worker_protocol_error"
            evaluation = self._load(Path(raw), records + [poisoned], [11])
            self.assertFalse(evaluation.ok)
            problem = next(
                p for p in evaluation.problems if p.kind == "runner_failure_record"
            )
            self.assertEqual(problem.seed, 11)
            self.assertIn("error", json.dumps(problem.context))

    def test_missing_output_file(self):
        with tempfile.TemporaryDirectory() as raw:
            evaluation = ex.load_output([1], Path(raw) / "absent.jsonl")
            self.assertFalse(evaluation.ok)
            self.assertEqual(evaluation.problems[0].kind, "missing_output")

    def test_empty_output_is_missing_seeds(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            evaluation = ex.load_output([1, 2], path)
            self.assertFalse(evaluation.ok)
            self.assertEqual(
                sorted(p.seed for p in evaluation.problems if p.kind == "missing_seed"),
                [1, 2],
            )

    def test_nonzero_exit_invalidates_run(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            records = complete_records(11)
            write_jsonl(tmp / "out.jsonl", records)
            run = ex.DriverRun(
                command=["fake"],
                returncode=3,
                elapsed_seconds=0.1,
                timed_out=False,
                stdout_tail="",
                stderr_tail="adapter crashed",
                output_path=tmp / "out.jsonl",
            )
            evaluation = ex.evaluate_run([11], run)
            self.assertFalse(evaluation.ok)
            kinds = {p.kind for p in evaluation.problems}
            self.assertIn("nonzero_exit", kinds)
            problem = next(p for p in evaluation.problems if p.kind == "nonzero_exit")
            self.assertIn("adapter crashed", problem.context["stderr_tail"])

    def test_timeout_records_the_actual_configured_seconds(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            write_jsonl(tmp / "out.jsonl", complete_records(11))
            run = ex.DriverRun(
                command=["fake"],
                returncode=-1,
                elapsed_seconds=2.6,
                timed_out=True,
                stdout_tail="",
                stderr_tail="",
                output_path=tmp / "out.jsonl",
                timeout_seconds=2.5,
            )
            evaluation = ex.evaluate_run([11], run)
            self.assertFalse(evaluation.ok)
            problem = next(p for p in evaluation.problems if p.kind == "driver_timeout")
            self.assertEqual(problem.context["timeout_seconds"], 2.5)
            self.assertEqual(problem.context["elapsed_seconds"], 2.6)


# ---------------------------------------------------------------------------
# Plan freezing
# ---------------------------------------------------------------------------


class TestPlan(unittest.TestCase):
    def _pilot_report(self, *, block_sd=2.0, seconds=1.0, count=4):
        procedure = ex.seed_procedure(31)
        seeds = ex.generate_seeds(procedure, count)
        return {
            "seeds": seeds,
            "count": count,
            "mode": "mixed",
            "decision_limit": 500,
            "block_sd": block_sd,
            "block_mean": 0.0,
            "mean_seconds_per_seed": seconds,
            "elapsed_seconds": seconds * count,
            "adapter_identity": {"fingerprint": "fp"},
        }

    def _plan(self, report, **overrides):
        params = {
            "pilot_report": report,
            "procedure": ex.seed_procedure(31),
            "mode": "mixed",
            "decision_limit": 500,
            "target_precision": None,
            "runtime_budget": None,
            "max_main_seeds": None,
            "analysis": dict(ANALYSIS),
            "adapter_identity": {"fingerprint": "fp"},
        }
        params.update(overrides)
        return ex.build_plan(**params)

    def test_precision_target_sets_sample_size(self):
        plan = self._plan(self._pilot_report(), target_precision=1.0)
        expected = math.ceil((ex.Z_95 * 2.0 / 1.0) ** 2)
        self.assertEqual(plan["sample"]["n_main"], expected)
        self.assertEqual(len(plan["sample"]["main_seeds"]), expected)
        self.assertTrue(plan["sample"]["precision_target_met"])
        self.assertEqual(plan["status"], "frozen")
        self.assertFalse(plan["rules"]["early_stopping"])

    def test_runtime_budget_sets_sample_size(self):
        plan = self._plan(self._pilot_report(seconds=2.0), runtime_budget=20.0)
        self.assertEqual(plan["sample"]["n_main"], 10)

    def test_both_constraints_take_the_smaller(self):
        plan = self._plan(
            self._pilot_report(block_sd=2.0, seconds=1.0),
            target_precision=1.0,
            runtime_budget=10.0,
        )
        self.assertEqual(plan["sample"]["n_main"], 10)
        self.assertFalse(plan["sample"]["precision_target_met"])
        self.assertTrue(
            any("runtime budget binds" in text for text in plan["limitations"])
        )

    def test_budget_too_small_is_infeasible(self):
        with self.assertRaises(ex.PlanningError):
            self._plan(self._pilot_report(seconds=1.0), runtime_budget=1.5)

    def test_max_main_seeds_caps_and_reports(self):
        plan = self._plan(
            self._pilot_report(), target_precision=1.0, max_main_seeds=5
        )
        self.assertEqual(plan["sample"]["n_main"], 5)
        self.assertFalse(plan["sample"]["precision_target_met"])
        self.assertTrue(any("capped" in text for text in plan["limitations"]))

    def test_plan_requires_target_or_budget(self):
        with self.assertRaises(ex.ExperimentError):
            self._plan(self._pilot_report())

    def test_zero_pilot_sd_uses_minimum_sample_without_claiming_certainty(self):
        plan = self._plan(self._pilot_report(block_sd=0.0), target_precision=1.0)
        self.assertEqual(plan["sample"]["n_main"], ex.MIN_INFERENCE_SEEDS)
        self.assertIsNone(plan["sample"]["precision_target_met"])
        self.assertTrue(
            any("zero uncertainty is not claimed" in text for text in plan["limitations"])
        )

    def test_precision_is_recorded_as_an_estimate(self):
        plan = self._plan(self._pilot_report(), target_precision=1.0)
        self.assertTrue(plan["sample"]["predicted_half_width_is_estimate"])
        self.assertIn("normal approximation", plan["sample"]["prediction_method"])
        self.assertIn("not guaranteed", plan["target"]["precision_note"])

    def test_plan_is_reproducible_and_pilot_disjoint(self):
        first = self._plan(self._pilot_report(), runtime_budget=6.0)
        second = self._plan(self._pilot_report(), runtime_budget=6.0)
        self.assertEqual(first["sample"]["main_seeds"], second["sample"]["main_seeds"])
        self.assertFalse(
            set(first["pilot"]["seeds"]) & set(first["sample"]["main_seeds"])
        )
        self.assertEqual(first["seed_procedure"]["generator"], ex.SEED_PROCEDURE_GENERATOR)
        self.assertEqual(first["analysis"]["bootstrap_seed"], ANALYSIS["bootstrap_seed"])

    def test_plan_rejects_changed_pilot_identity(self):
        report = self._pilot_report()
        with self.assertRaises(ex.PlanningError):
            self._plan(
                report,
                runtime_budget=4.0,
                adapter_identity={"fingerprint": "other"},
            )
        report.pop("adapter_identity")
        with self.assertRaises(ex.PlanningError):
            self._plan(report, runtime_budget=4.0)

    def test_plan_inherits_pilot_mode_and_decision_limit(self):
        plan = self._plan(
            self._pilot_report(), mode=None, decision_limit=None, runtime_budget=4.0
        )
        self.assertEqual(plan["mode"], "mixed")
        self.assertEqual(plan["decision_limit"], 500)

    def test_plan_rejects_changed_pilot_configuration(self):
        report = self._pilot_report()
        with self.assertRaises(ex.PlanningError):
            self._plan(report, mode="first-only", runtime_budget=4.0)
        with self.assertRaises(ex.PlanningError):
            self._plan(report, decision_limit=501, runtime_budget=4.0)

    def test_plan_rejects_pilot_report_without_configuration(self):
        report = self._pilot_report()
        report.pop("mode")
        with self.assertRaises(ex.PlanningError):
            self._plan(report, runtime_budget=4.0)
        report = self._pilot_report()
        report.pop("decision_limit")
        with self.assertRaises(ex.PlanningError):
            self._plan(report, runtime_budget=4.0)

    def test_plan_rejects_pilot_seeds_outside_recorded_prefix(self):
        report = self._pilot_report()
        tampered = [seed + 1 for seed in report["seeds"]]
        report["seeds"] = tampered
        report["count"] = len(tampered)
        with self.assertRaises(ex.PlanningError):
            self._plan(report, runtime_budget=4.0)

    def test_plan_procedure_requires_recorded_corpus(self):
        report = self._pilot_report()
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            ex.write_json(work / "pilot" / "report.json", report)
            with self.assertRaises(ex.ExperimentError):
                ex.plan_procedure_from(work, report)

            procedure = ex.seed_procedure(31)
            ex.write_json(
                work / "corpus.json",
                {"procedure": procedure, "pilot_seeds": list(report["seeds"])},
            )
            self.assertEqual(ex.plan_procedure_from(work, report), procedure)

            ex.write_json(
                work / "corpus.json",
                {
                    "procedure": procedure,
                    "pilot_seeds": list(reversed(report["seeds"])),
                },
            )
            with self.assertRaises(ex.ExperimentError):
                ex.plan_procedure_from(work, report)

    def test_write_plan_refuses_to_overwrite_frozen_plan(self):
        plan = self._plan(self._pilot_report(), runtime_budget=4.0)
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            ex.write_plan(work, plan)
            with self.assertRaises(ex.ExperimentError):
                ex.write_plan(work, plan)

            # A started main attempt also blocks replacing a non-frozen plan.
            (work / "main").mkdir()
            (work / "main" / "results.jsonl").write_text("", encoding="utf-8")
            ex.write_json(work / "plan.json", {"status": "draft"})
            with self.assertRaises(ex.ExperimentError):
                ex.write_plan(work, plan)

    def test_plan_round_trip_and_seed_validation(self):
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            plan = self._plan(self._pilot_report(), runtime_budget=4.0)
            ex.write_plan(work, plan)
            loaded = ex.load_plan(work)
            seeds = ex.resolve_plan_seeds(loaded)
            self.assertEqual(seeds, plan["sample"]["main_seeds"])

            broken = json.loads(json.dumps(loaded))
            broken["sample"]["main_seeds"] = [plan["pilot"]["seeds"][0]] + seeds[1:]
            with self.assertRaises(ex.ExperimentError):
                ex.resolve_plan_seeds(broken)

            broken = json.loads(json.dumps(loaded))
            broken["sample"]["main_seeds"] = [seeds[0], seeds[0]]
            with self.assertRaises(ex.ExperimentError):
                ex.resolve_plan_seeds(broken)

            broken = json.loads(json.dumps(loaded))
            broken["status"] = "draft"
            ex.write_json(work / "plan.json", broken)
            with self.assertRaises(ex.ExperimentError):
                ex.load_plan(work)

    def test_verify_plan_detects_drift(self):
        plan = self._plan(self._pilot_report(), runtime_budget=4.0)
        seeds = ex.verify_plan_for_main(
            plan,
            mode="mixed",
            decision_limit=500,
            analysis=dict(ANALYSIS),
            adapter_identity={"fingerprint": "fp"},
        )
        self.assertEqual(seeds, plan["sample"]["main_seeds"])

        with self.assertRaises(ex.ExperimentError):
            ex.verify_plan_for_main(
                plan,
                mode="first-only",
                decision_limit=500,
                analysis=dict(ANALYSIS),
                adapter_identity={"fingerprint": "fp"},
            )
        with self.assertRaises(ex.ExperimentError):
            ex.verify_plan_for_main(
                plan,
                mode="mixed",
                decision_limit=501,
                analysis=dict(ANALYSIS),
                adapter_identity={"fingerprint": "fp"},
            )
        drifted_analysis = dict(ANALYSIS, bootstrap_seed=ANALYSIS["bootstrap_seed"] + 1)
        with self.assertRaises(ex.ExperimentError):
            ex.verify_plan_for_main(
                plan,
                mode="mixed",
                decision_limit=500,
                analysis=drifted_analysis,
                adapter_identity={"fingerprint": "fp"},
            )
        with self.assertRaises(ex.ExperimentError):
            ex.verify_plan_for_main(
                plan,
                mode="mixed",
                decision_limit=500,
                analysis=dict(ANALYSIS),
                adapter_identity={"fingerprint": "other"},
            )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary(unittest.TestCase):
    def test_summary_reports_known_estimates(self):
        summary = summary_for_blocks({"a": 3.0, "b": 1.0})
        self.assertEqual(summary["status"], "ok")
        self.assertAlmostEqual(summary["means_per_player_per_game"]["delta"], 2.0)
        self.assertAlmostEqual(summary["means_per_player_per_game"]["a"], 1.0)
        self.assertAlmostEqual(summary["means_per_player_per_game"]["b"], -1.0)
        self.assertAlmostEqual(summary["uncertainty"]["sample_sd"], math.sqrt(2.0))
        self.assertAlmostEqual(summary["uncertainty"]["standard_error"], 1.0)
        self.assertTrue(summary["checks"]["delta_matches_policy_means"])
        self.assertFalse(summary["checks"]["symmetric_zero_blocks"]["required"])
        self.assertFalse(summary["analysis"]["early_stopping"])
        self.assertTrue(summary["seeds"]["pilot_excluded"])

    def test_symmetric_mode_requires_zero_blocks(self):
        summary = summary_for_blocks({"a": 0.0, "b": 0.0}, mode="first-only")
        self.assertTrue(summary["checks"]["symmetric_zero_blocks"]["required"])
        self.assertTrue(summary["checks"]["symmetric_zero_blocks"]["all_zero"])

    def test_symmetric_mode_rejects_nonzero_blocks(self):
        with self.assertRaises(ex.SummaryError):
            summary_for_blocks({"a": 1.0, "b": 0.0}, mode="first-only")
        with self.assertRaises(ex.SummaryError):
            summary_for_blocks({"a": 1.0, "b": 0.0}, mode="heuristic-only")

    def test_unexpected_end_reason_is_flagged(self):
        summary = summary_for_blocks(
            {"a": 1.0, "b": 2.0}, end_reason="hu_win"
        )
        self.assertEqual(summary["diagnostics"]["unexpected_end_reasons"], ["hu_win"])
        self.assertTrue(
            any("unexpected end_reason" in warning for warning in summary["warnings"])
        )

    def test_known_end_reasons_are_catalogued(self):
        self.assertEqual(ex.KNOWN_END_REASONS, ("three_winners", "wall_exhausted"))
        summary = summary_for_blocks(
            {"a": 1.0, "b": 2.0}, end_reason="three_winners"
        )
        self.assertEqual(summary["diagnostics"]["unexpected_end_reasons"], [])
        self.assertEqual(summary["diagnostics"]["end_reasons"], {"three_winners": 12})

    def test_summary_refuses_invalid_evaluation(self):
        plan, blocks = plan_for_blocks({"a": 1.0, "b": 2.0})
        results = synthetic_results(blocks)
        evaluation = ex.RunEvaluation(
            ok=False,
            results=results,
            problems=[ex.Problem("missing_seed", "seed missing", seed=blocks_tip(blocks))],
            by_seed=grouped(results),
        )
        with self.assertRaises(ex.SummaryError):
            ex.build_summary(
                plan=plan,
                evaluation=evaluation,
                timing=None,
                deterministic_replay=None,
            )

    def test_summary_marks_unverified_replay(self):
        summary = summary_for_blocks({"a": 1.0, "b": 2.0})
        self.assertEqual(summary["checks"]["deterministic_replay"], "verified")
        plan, blocks = plan_for_blocks({"a": 1.0, "b": 2.0})
        results = synthetic_results(blocks)
        evaluation = ex.RunEvaluation(
            ok=True, results=results, problems=[], by_seed=grouped(results)
        )
        summary = ex.build_summary(
            plan=plan, evaluation=evaluation, timing=None, deterministic_replay=None
        )
        self.assertEqual(summary["checks"]["deterministic_replay"], "not-checked")
        self.assertTrue(any("replay" in warning for warning in summary["warnings"]))

    def test_summary_reports_planning_estimate_and_zero_sd_warning(self):
        summary = summary_for_blocks({"a": 1.0, "b": 2.0}, target_precision=1.0)
        planning = summary["planning"]
        self.assertTrue(planning["predicted_half_width_is_estimate"])
        self.assertIn("planning estimate", planning["note"])
        self.assertGreater(planning["predicted_half_width"], 0.0)
        self.assertGreater(summary["uncertainty"]["ci_half_width"], 0.0)
        self.assertEqual(
            planning["measured_ci_half_width"],
            summary["uncertainty"]["ci_half_width"],
        )

        plan, blocks = plan_for_blocks({"a": 1.0, "b": 2.0})
        plan["pilot"]["block_sd"] = 0.0
        results = synthetic_results(blocks)
        evaluation = ex.RunEvaluation(
            ok=True, results=results, problems=[], by_seed=grouped(results)
        )
        summary = ex.build_summary(
            plan=plan,
            evaluation=evaluation,
            timing=None,
            deterministic_replay={"status": "verified"},
        )
        self.assertTrue(
            any("zero uncertainty" in warning for warning in summary["warnings"])
        )


def blocks_tip(blocks):
    return next(iter(blocks))


# ---------------------------------------------------------------------------
# Manifest and adapter identity
# ---------------------------------------------------------------------------


class TestRepositoryCapture(unittest.TestCase):
    def _repo_with_changes(self, root: Path) -> None:
        init_git_repo(root)
        (root / "a.txt").write_bytes(b"one\n")
        run_git("add", ".", cwd=root)
        run_git("commit", "-qm", "init", cwd=root)
        (root / "a.txt").write_bytes(b"two\n")
        (root / "b.txt").write_bytes(b"untracked evidence\n")
        (root / "target").mkdir()
        (root / "target" / "junk.bin").write_bytes(b"\x00\x01")
        runs = root / "experiments" / "policy_ab" / "runs"
        runs.mkdir(parents=True)
        (runs / "run.json").write_bytes(b"{}\n")

    def test_capture_revision_diff_and_untracked(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            repo = tmp / "repo"
            self._repo_with_changes(repo)
            snapshot = tmp / "snap"
            info = ex.capture_repository(
                repo, name="repo", reference_dir=tmp, snapshot_dir=snapshot
            )
            self.assertTrue(info["is_git_repo"])
            self.assertEqual(info["head"], run_git("rev-parse", "HEAD", cwd=repo))
            self.assertTrue(info["dirty"])
            patch = (snapshot / "changes.patch").read_text(encoding="utf-8")
            self.assertIn("two", patch)
            self.assertEqual(
                info["diff_sha256"],
                ex.sha256_bytes((snapshot / "changes.patch").read_bytes()),
            )

            untracked = {entry["path"]: entry for entry in info["untracked"]}
            self.assertIn("b.txt", untracked)
            self.assertEqual(
                untracked["b.txt"]["sha256"], ex.sha256_bytes(b"untracked evidence\n")
            )
            self.assertEqual(
                (snapshot / "untracked" / "b.txt").read_bytes(),
                b"untracked evidence\n",
            )
            # Generated build/experiment outputs never enter provenance.
            self.assertNotIn("target/junk.bin", untracked)
            self.assertNotIn("experiments/policy_ab/runs/run.json", untracked)
            self.assertEqual(info["omitted_generated_entries"], 2)
            self.assertFalse((snapshot / "untracked" / "target").exists())
            self.assertFalse((snapshot / "untracked" / "experiments").exists())

    def test_capture_non_git_directory(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            info = ex.capture_repository(tmp / "plain", name="plain", reference_dir=tmp)
            self.assertFalse(info["is_git_repo"])
            self.assertIn("note", info)

    def test_generated_path_classifier(self):
        for path in (
            "target/debug/foo",
            "experiments/policy_ab/runs/x/summary.json",
            "src/__pycache__/x.pyc",
            "./target/examples/policy_ab",
        ):
            self.assertTrue(ex._is_generated_path(path), path)
        for path in ("src/game.rs", "experiments/policy_ab/experiment.py", "Cargo.lock"):
            self.assertFalse(ex._is_generated_path(path), path)


class TestAdapterIdentity(unittest.TestCase):
    def _layout(self, tmp: Path) -> dict[str, Path]:
        repo = tmp / "repo"
        (repo / "src").mkdir(parents=True)
        (repo / "examples").mkdir(parents=True)
        (repo / "Cargo.toml").write_text("[package]\nname=\"x\"\n", encoding="utf-8")
        (repo / "Cargo.lock").write_text("# lock\n", encoding="utf-8")
        (repo / "src" / "lib.rs").write_text("pub fn a() {}\n", encoding="utf-8")
        (repo / "examples" / "policy_ab.rs").write_text("fn main() {}\n", encoding="utf-8")
        driver = repo / "policy_ab.exe"
        driver.write_bytes(b"binary-v1")
        worker = tmp / "worker.py"
        worker.write_text("# worker\n", encoding="utf-8")
        return {"repo": repo, "driver": driver, "worker": worker}

    def test_identity_records_build_inputs_and_executable(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            layout = self._layout(tmp)
            rust = ex.capture_rust_sources(layout["repo"])
            identity = ex.adapter_identity(
                driver=layout["driver"],
                worker=layout["worker"],
                python_exe=Path(sys.executable),
                repo_root=layout["repo"],
                ninone_capture={
                    "root": str(tmp),
                    "is_git_repo": False,
                    "head": None,
                    "diff_sha256": None,
                    "dirty": False,
                },
                rust_sources=rust,
            )
            self.assertEqual(identity["driver"]["sha256"], ex.sha256_bytes(b"binary-v1"))
            self.assertEqual(
                identity["rust_sources"]["cargo_lock_sha256"],
                ex.sha256_file(layout["repo"] / "Cargo.lock"),
            )
            self.assertEqual(
                identity["rust_sources"]["cargo_toml_sha256"],
                ex.sha256_file(layout["repo"] / "Cargo.toml"),
            )
            self.assertEqual(identity["rust_sources"]["rust_source_count"], 2)
            self.assertEqual(identity["worker"]["sha256"], ex.sha256_file(layout["worker"]))
            self.assertEqual(identity["script"]["sha256"], ex.sha256_file(ex.SCRIPT_PATH))
            self.assertEqual(
                identity["python"]["resolved_path"], str(Path(sys.executable).resolve())
            )
            self.assertIsNotNone(identity["python"]["sha256"])
            self.assertTrue(identity["driver_freshness"]["checked"])
            self.assertFalse(identity["driver_freshness"]["stale"])
            self.assertIsNotNone(identity["driver_freshness"]["newest_source_mtime_utc"])

    def test_identity_fingerprint_changes_with_sources(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            layout = self._layout(tmp)
            base = ex.adapter_identity(
                driver=layout["driver"],
                worker=layout["worker"],
                python_exe=Path(sys.executable),
                repo_root=layout["repo"],
                ninone_capture={"root": str(tmp), "head": None, "diff_sha256": None},
            )
            (layout["repo"] / "src" / "lib.rs").write_text("pub fn b() {}\n", encoding="utf-8")
            changed = ex.adapter_identity(
                driver=layout["driver"],
                worker=layout["worker"],
                python_exe=Path(sys.executable),
                repo_root=layout["repo"],
                ninone_capture={"root": str(tmp), "head": None, "diff_sha256": None},
                allow_stale_driver=True,
            )
            self.assertNotEqual(base["fingerprint"], changed["fingerprint"])
            self.assertNotEqual(
                base["rust_sources"]["rust_source_aggregate_sha256"],
                changed["rust_sources"]["rust_source_aggregate_sha256"],
            )

    def test_stale_driver_is_rejected_unless_allowed(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            layout = self._layout(tmp)
            sources = ex.capture_rust_sources(layout["repo"])
            newest = sources["newest_source_epoch"]
            os.utime(layout["driver"], (newest - 60.0, newest - 60.0))
            with self.assertRaises(ex.ExperimentError):
                ex.adapter_identity(
                    driver=layout["driver"],
                    worker=layout["worker"],
                    python_exe=Path(sys.executable),
                    repo_root=layout["repo"],
                    ninone_capture={"root": str(tmp), "head": None, "diff_sha256": None},
                    rust_sources=sources,
                )
            identity = ex.adapter_identity(
                driver=layout["driver"],
                worker=layout["worker"],
                python_exe=Path(sys.executable),
                repo_root=layout["repo"],
                ninone_capture={"root": str(tmp), "head": None, "diff_sha256": None},
                rust_sources=sources,
                allow_stale_driver=True,
            )
            self.assertTrue(identity["driver_freshness"]["stale"])
            self.assertTrue(identity["driver_freshness"]["waived"])

    def test_python_identity_probes_worker_interpreter(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            layout = self._layout(tmp)
            identity = ex.adapter_identity(
                driver=layout["driver"],
                worker=layout["worker"],
                python_exe=Path(sys.executable),
                repo_root=layout["repo"],
                ninone_capture={"root": str(tmp), "head": None, "diff_sha256": None},
            )
            python = identity["python"]
            self.assertEqual(python["version"], platform.python_version())
            self.assertEqual(python["implementation"], platform.python_implementation())
            self.assertEqual(python["version_source"], "worker interpreter probe")
            self.assertEqual(python["analysis_python_version"], platform.python_version())
            self.assertIsNone(python["probe_error"])
            self.assertEqual(python["resolved_path"], str(Path(sys.executable).resolve()))

    def test_python_command_name_is_resolved(self):
        name = Path(sys.executable).name
        if shutil.which(name) is None:
            self.skipTest(f"{name!r} is not resolvable on PATH")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            layout = self._layout(tmp)
            identity = ex.adapter_identity(
                driver=layout["driver"],
                worker=layout["worker"],
                python_exe=Path(name),
                repo_root=layout["repo"],
                ninone_capture={"root": str(tmp), "head": None, "diff_sha256": None},
            )
            python = identity["python"]
            self.assertEqual(python["requested"], name)
            self.assertTrue(python["resolved_path"].lower().endswith(name.lower()))
            self.assertTrue(python["present"])
            self.assertIsNotNone(python["sha256"])
            self.assertIsNotNone(python["version"])

    def test_unresolvable_python_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            layout = self._layout(tmp)
            with self.assertRaises(ex.ExperimentError):
                ex.adapter_identity(
                    driver=layout["driver"],
                    worker=layout["worker"],
                    python_exe=Path("definitely-not-a-python-interpreter-xyz"),
                    repo_root=layout["repo"],
                    ninone_capture={"root": str(tmp), "head": None, "diff_sha256": None},
                )

    def test_ninone_heuristic_source_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            layout = self._layout(tmp)
            ninone = tmp / "ninone-repo"
            heuristic = ninone / "ninone" / "policies" / "agent_heuristic.py"
            heuristic.parent.mkdir(parents=True)
            heuristic.write_text("CHOICE = 1\n", encoding="utf-8")
            base = ex.adapter_identity(
                driver=layout["driver"],
                worker=layout["worker"],
                python_exe=Path(sys.executable),
                repo_root=layout["repo"],
                ninone_capture={"root": str(ninone), "head": None, "diff_sha256": None},
            )
            self.assertEqual(base["ninone"]["sources"]["file_count"], 1)
            aggregate = base["ninone"]["sources"]["aggregate_sha256"]
            self.assertIsNotNone(aggregate)

            # An untracked/ignored edit changes the heuristic, so identity must change.
            heuristic.write_text("CHOICE = 2\n", encoding="utf-8")
            changed = ex.adapter_identity(
                driver=layout["driver"],
                worker=layout["worker"],
                python_exe=Path(sys.executable),
                repo_root=layout["repo"],
                ninone_capture={"root": str(ninone), "head": None, "diff_sha256": None},
            )
            self.assertNotEqual(base["fingerprint"], changed["fingerprint"])
            self.assertNotEqual(
                aggregate, changed["ninone"]["sources"]["aggregate_sha256"]
            )

    def test_missing_worker_is_recorded_not_fatal(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            layout = self._layout(tmp)
            identity = ex.adapter_identity(
                driver=layout["driver"],
                worker=tmp / "absent_worker.py",
                python_exe=Path(sys.executable),
                repo_root=layout["repo"],
                ninone_capture={"root": str(tmp), "head": None, "diff_sha256": None},
            )
            self.assertFalse(identity["worker"]["present"])
            self.assertIsNone(identity["worker"]["sha256"])
            self.assertTrue(identity["fingerprint"])


class TestManifest(unittest.TestCase):
    def test_manifest_binds_sources_and_repositories(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            repo = tmp / "sim"
            init_git_repo(repo)
            (repo / "src").mkdir()
            (repo / "examples").mkdir()
            (repo / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
            (repo / "Cargo.lock").write_text("# lock\n", encoding="utf-8")
            (repo / "src" / "lib.rs").write_text("pub fn a() {}\n", encoding="utf-8")
            (repo / "examples" / "policy_ab.rs").write_text("fn main() {}\n", encoding="utf-8")
            run_git("add", ".", cwd=repo)
            run_git("commit", "-qm", "init", cwd=repo)
            driver = repo / "policy_ab.exe"
            driver.write_bytes(b"binary-v1")

            ninone = tmp / "ninone"
            ninone.mkdir()
            worker = tmp / "worker.py"
            worker.write_text("# worker\n", encoding="utf-8")

            work = tmp / "work"
            plan, _blocks = plan_for_blocks({"a": 1.0, "b": 2.0})
            adapter = ex.adapter_identity(
                driver=driver,
                worker=worker,
                python_exe=Path(sys.executable),
                repo_root=repo,
                ninone_capture={"root": str(ninone), "head": None, "diff_sha256": None},
            )
            manifest = ex.build_manifest(
                work_dir=work,
                repo_root=repo,
                ninone_root=ninone,
                plan=plan,
                adapter=adapter,
                invocation={"command": ["policy_ab"], "mode": "mixed"},
                timing={"elapsed_seconds": 1.0},
                deterministic_replay={"status": "verified", "seeds": []},
                argv=["run"],
            )
            ex.write_json(work / "manifest.json", manifest)
            reloaded = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                set(reloaded["repositories"]),
                {"sichuan-mahjong-sim", "ninone"},
            )
            binding = reloaded["source_binding"]
            self.assertEqual(binding["driver_executable_sha256"], ex.sha256_bytes(b"binary-v1"))
            self.assertEqual(
                binding["cargo_lock_sha256"], ex.sha256_file(repo / "Cargo.lock")
            )
            self.assertEqual(binding["worker_sha256"], ex.sha256_file(worker))
            self.assertEqual(
                binding["analysis_script_sha256"], ex.sha256_file(ex.SCRIPT_PATH)
            )
            self.assertEqual(binding["identity_fingerprint"], adapter["fingerprint"])
            self.assertEqual(
                binding["python_executable"], str(Path(sys.executable).resolve())
            )
            self.assertEqual(
                reloaded["repositories"]["sichuan-mahjong-sim"]["head"],
                run_git("rev-parse", "HEAD", cwd=repo),
            )
            # Generated outputs are not recursively captured.
            self.assertFalse(
                (work / "repos" / "sichuan-mahjong-sim" / "untracked" / "target").exists()
            )
            self.assertEqual(reloaded["experiment"]["mode"], "mixed")
            self.assertEqual(
                reloaded["experiment"]["main"]["seeds"], plan["sample"]["main_seeds"]
            )
            self.assertEqual(
                reloaded["analysis"]["bootstrap_seed"], ANALYSIS["bootstrap_seed"]
            )
            self.assertIsNotNone(reloaded["environment"]["python_version"])


# ---------------------------------------------------------------------------
# End-to-end smoke runs through a fake driver
# ---------------------------------------------------------------------------


class TestWorkflowSmoke(unittest.TestCase):
    def _read(self, path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    def _common_args(self, work: Path, env: dict[str, Path], driver: Path):
        return [
            "--work-dir",
            str(work),
            "--driver",
            str(driver),
            "--repo-root",
            str(env["repo"]),
            "--ninone-root",
            str(env["ninone"]),
            "--worker",
            str(env["worker"]),
            "--python",
            sys.executable,
        ]

    def _successful_run(self, tmp: Path, *extra: str):
        env = fake_environment(tmp)
        driver = fake_driver(tmp)
        work = tmp / "work"
        args = run_args(
            work,
            tmp,
            env,
            driver,
            "--pilot-size",
            "3",
            "--runtime-budget",
            "10.0",
            "--max-main-seeds",
            "2",
            "--determinism-check",
            "0",
            *extra,
        )
        self.assertEqual(ex.main(args), 0)
        return work, env, driver

    def _rewrite_run_record(self, work: Path, mutate) -> None:
        run_path = work / "main" / "run.json"
        record = self._read(run_path)
        mutate(record)
        ex.write_json(run_path, record)
        manifest_path = work / "manifest.json"
        manifest = self._read(manifest_path)
        manifest["artifacts"]["main_run_sha256"] = ex.sha256_file(run_path)
        ex.write_json(manifest_path, manifest)

    def test_run_workflow_publishes_summary(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            env = fake_environment(tmp)
            driver = fake_driver(tmp)
            work = tmp / "work"
            args = run_args(
                work,
                tmp,
                env,
                driver,
                "--mode",
                "mixed",
                "--pilot-size",
                "3",
                "--target-precision",
                "1.0",
                "--max-main-seeds",
                "2",
                "--bootstrap-resamples",
                "200",
                "--bootstrap-seed",
                "7",
                "--determinism-check",
                "2",
                "--decision-limit",
                "50",
            )
            self.assertEqual(ex.main(args), 0)

            plan = self._read(work / "plan.json")
            summary = self._read(work / "summary.json")
            manifest = self._read(work / "manifest.json")
            corpus = self._read(work / "corpus.json")

            self.assertEqual(plan["status"], "frozen")
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["seeds"]["count"], 2)
            self.assertEqual(
                summary["seeds"]["list"], plan["sample"]["main_seeds"]
            )
            self.assertFalse(
                set(summary["seeds"]["list"]) & set(corpus["pilot_seeds"])
            )
            self.assertGreater(summary["seeds"]["count"], 1)
            self.assertTrue(summary["checks"]["complete_placement_coverage"])
            self.assertTrue(summary["checks"]["zero_sum_scores"])
            self.assertEqual(summary["checks"]["deterministic_replay"], "verified")
            self.assertIn("means_per_player_per_game", summary)
            self.assertTrue((work / "summary.md").is_file())
            self.assertFalse((work / "main" / "failures.json").exists())
            self.assertIn("source_binding", manifest)
            self.assertEqual(
                manifest["source_binding"]["identity_fingerprint"],
                plan["adapter_identity"]["fingerprint"],
            )
            self.assertEqual(
                manifest["experiment"]["main"]["seeds"], summary["seeds"]["list"]
            )
            self.assertTrue(summary["integrity"]["verified"])
            self.assertFalse(summary["integrity"]["replay_triggered"])
            self.assertEqual(
                summary["provenance"]["results_sha256"],
                ex.sha256_file(work / "main" / "results.jsonl"),
            )
            self.assertEqual(
                summary["provenance"]["manifest_sha256"],
                ex.sha256_file(work / "manifest.json"),
            )
            self.assertEqual(
                manifest["artifacts"]["main_results_sha256"],
                ex.sha256_file(work / "main" / "results.jsonl"),
            )
            self.assertEqual(
                manifest["artifacts"]["main_seeds_sha256"],
                ex.sha256_file(work / "main" / "seeds.json"),
            )
            # The frozen plan, not a CLI override, determines the executed sample.
            self.assertEqual(
                [
                    json.loads(line)["seed"]
                    for line in (work / "main" / "results.jsonl")
                    .read_text()
                    .splitlines()
                ],
                [seed for seed in plan["sample"]["main_seeds"] for _ in ex.PLACEMENTS],
            )

            # Summarizing again from stored results must be reproducible.
            before = summary["means_per_player_per_game"]["delta"]
            self.assertEqual(ex.main(["summarize", "--work-dir", str(work)]), 0)
            after = self._read(work / "summary.json")["means_per_player_per_game"]["delta"]
            self.assertEqual(before, after)

    def test_determinism_check_detects_trace_hash_only_mismatch(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            env = fake_environment(tmp)
            driver = fake_driver(tmp)
            work = tmp / "work"
            os.environ["FAKE_DRIVER_MODE"] = "trace_by_path"
            try:
                code = ex.main(
                    run_args(
                        work,
                        tmp,
                        env,
                        driver,
                        "--pilot-size",
                        "3",
                        "--runtime-budget",
                        "10.0",
                        "--max-main-seeds",
                        "2",
                        "--determinism-check",
                        "2",
                    )
                )
            finally:
                os.environ.pop("FAKE_DRIVER_MODE", None)
            self.assertEqual(code, 1)
            self.assertFalse((work / "summary.json").exists())
            failures = self._read(work / "main" / "failures.json")
            kinds = {p["kind"] for p in failures["problems"]}
            self.assertIn("deterministic_replay_mismatch", kinds)
            mismatch = failures["problems"][0]["context"]["mismatches"][0]
            self.assertEqual(mismatch["main"]["scores"], mismatch["replay"]["scores"])
            self.assertNotEqual(
                mismatch["main"]["trace_hash"], mismatch["replay"]["trace_hash"]
            )
            self.assertEqual(
                self._read(work / "main" / "run.json")["deterministic_replay"][
                    "status"
                ],
                "failed",
            )

    def test_rerun_refuses_existing_attempts(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            work, env, driver = self._successful_run(tmp)
            results_path = work / "main" / "results.jsonl"
            results_before = results_path.read_bytes()
            summary_before = (work / "summary.json").read_text(encoding="utf-8")

            repeat = ex.main(
                run_args(
                    work,
                    tmp,
                    env,
                    driver,
                    "--pilot-size",
                    "3",
                    "--runtime-budget",
                    "10.0",
                    "--max-main-seeds",
                    "2",
                    "--determinism-check",
                    "0",
                )
            )
            self.assertEqual(repeat, 1)
            self.assertEqual(results_path.read_bytes(), results_before)
            self.assertEqual(
                (work / "summary.json").read_text(encoding="utf-8"), summary_before
            )

            common = self._common_args(work, env, driver)
            self.assertEqual(ex.main(["main", *common]), 1)
            self.assertEqual(results_path.read_bytes(), results_before)
            self.assertEqual(
                (work / "summary.json").read_text(encoding="utf-8"), summary_before
            )

    def test_plan_refuses_changed_pilot_identity(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            env = fake_environment(tmp)
            driver = fake_driver(tmp)
            work = tmp / "work"
            common = self._common_args(work, env, driver)
            self.assertEqual(ex.main(["pilot", *common, "--pilot-size", "3"]), 0)
            env["worker"].write_text("# worker changed\n", encoding="utf-8")
            self.assertEqual(
                ex.main(
                    [
                        "plan",
                        *common,
                        "--target-precision",
                        "1.0",
                        "--max-main-seeds",
                        "2",
                    ]
                ),
                1,
            )
            self.assertFalse((work / "plan.json").exists())

    def test_summarize_refuses_nonzero_run_record(self):
        with tempfile.TemporaryDirectory() as raw:
            work, _env, _driver = self._successful_run(Path(raw))

            def mutate(record):
                record["returncode"] = 3

            self._rewrite_run_record(work, mutate)
            self.assertEqual(ex.main(["summarize", "--work-dir", str(work)]), 1)
            self.assertFalse((work / "summary.json").exists())

    def test_summarize_refuses_timed_out_run_record(self):
        with tempfile.TemporaryDirectory() as raw:
            work, _env, _driver = self._successful_run(Path(raw))

            def mutate(record):
                record["timed_out"] = True

            self._rewrite_run_record(work, mutate)
            self.assertEqual(ex.main(["summarize", "--work-dir", str(work)]), 1)
            self.assertFalse((work / "summary.json").exists())

    def test_summarize_refuses_failed_replay_record(self):
        with tempfile.TemporaryDirectory() as raw:
            work, _env, _driver = self._successful_run(Path(raw))

            def mutate(record):
                record["deterministic_replay"]["status"] = "failed"

            self._rewrite_run_record(work, mutate)
            self.assertEqual(ex.main(["summarize", "--work-dir", str(work)]), 1)
            self.assertFalse((work / "summary.json").exists())

    def test_summarize_refuses_missing_run_evidence(self):
        with tempfile.TemporaryDirectory() as raw:
            work, _env, _driver = self._successful_run(Path(raw))
            manifest_path = work / "manifest.json"
            manifest = self._read(manifest_path)
            manifest["artifacts"].pop("main_run_sha256")
            ex.write_json(manifest_path, manifest)
            self.assertEqual(ex.main(["summarize", "--work-dir", str(work)]), 1)
            self.assertFalse((work / "summary.json").exists())

    def test_run_workflow_rejects_failing_driver(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            env = fake_environment(tmp)
            driver = fake_driver(tmp)
            work = tmp / "work"
            os.environ["FAKE_DRIVER_MODE"] = "exit"
            try:
                code = ex.main(
                    run_args(
                        work,
                        tmp,
                        env,
                        driver,
                        "--pilot-size",
                        "3",
                        "--runtime-budget",
                        "10.0",
                        "--max-main-seeds",
                        "2",
                        "--determinism-check",
                        "0",
                    )
                )
            finally:
                os.environ.pop("FAKE_DRIVER_MODE", None)
            self.assertEqual(code, 1)
            self.assertFalse((work / "summary.json").exists())
            failures = self._read(work / "main" / "failures.json")
            self.assertFalse(failures["normal_summary_published"])
            self.assertIn("nonzero_exit", {p["kind"] for p in failures["problems"]})
            self.assertTrue(failures["failing_seeds"])
            self.assertTrue(failures["preserved_results"])
            self.assertTrue((work / "manifest.json").is_file())

    def test_run_workflow_rejects_runner_failure_record(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            env = fake_environment(tmp)
            driver = fake_driver(tmp)
            work = tmp / "work"
            os.environ["FAKE_DRIVER_MODE"] = "failure_record"
            try:
                code = ex.main(
                    run_args(
                        work,
                        tmp,
                        env,
                        driver,
                        "--pilot-size",
                        "3",
                        "--runtime-budget",
                        "10.0",
                        "--max-main-seeds",
                        "2",
                        "--determinism-check",
                        "0",
                    )
                )
            finally:
                os.environ.pop("FAKE_DRIVER_MODE", None)
            self.assertEqual(code, 1)
            self.assertFalse((work / "summary.json").exists())
            failures = self._read(work / "main" / "failures.json")
            kinds = {p["kind"] for p in failures["problems"]}
            self.assertIn("runner_failure_record", kinds)
            self.assertIn(
                "worker_protocol_error", json.dumps(failures["problems"])
            )

    def test_split_workflow_pilot_plan_main_summarize(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            env = fake_environment(tmp)
            driver = fake_driver(tmp)
            work = tmp / "work"
            common = [
                "--work-dir",
                str(work),
                "--driver",
                str(driver),
                "--repo-root",
                str(env["repo"]),
                "--ninone-root",
                str(env["ninone"]),
                "--worker",
                str(env["worker"]),
                "--python",
                sys.executable,
            ]
            self.assertEqual(
                ex.main(["pilot", *common, "--pilot-size", "3", "--mode", "mixed"]), 0
            )
            self.assertTrue((work / "pilot" / "report.json").is_file())
            self.assertFalse((work / "plan.json").exists())

            self.assertEqual(
                ex.main(
                    [
                        "plan",
                        *common,
                        "--target-precision",
                        "1.0",
                        "--max-main-seeds",
                        "2",
                    ]
                ),
                0,
            )
            plan = self._read(work / "plan.json")
            planned_seeds = list(plan["sample"]["main_seeds"])
            self.assertEqual(plan["status"], "frozen")

            self.assertEqual(
                ex.main(["main", *common, "--determinism-check", "2"]), 0
            )
            summary = self._read(work / "summary.json")
            self.assertEqual(summary["seeds"]["list"], planned_seeds)
            self.assertEqual(
                self._read(work / "plan.json")["sample"]["main_seeds"], planned_seeds
            )

            self.assertEqual(ex.main(["summarize", "--work-dir", str(work)]), 0)
            resummarized = self._read(work / "summary.json")
            self.assertEqual(
                resummarized["means_per_player_per_game"],
                summary["means_per_player_per_game"],
            )
            self.assertTrue(resummarized["integrity"]["verified"])
            self.assertFalse(resummarized["integrity"]["replay_triggered"])

    def test_summarize_refuses_tampered_results(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            env = fake_environment(tmp)
            driver = fake_driver(tmp)
            work = tmp / "work"
            self.assertEqual(
                ex.main(
                    run_args(
                        work,
                        tmp,
                        env,
                        driver,
                        "--pilot-size",
                        "3",
                        "--runtime-budget",
                        "10.0",
                        "--max-main-seeds",
                        "2",
                        "--determinism-check",
                        "0",
                    )
                ),
                0,
            )
            results_path = work / "main" / "results.jsonl"
            records = [json.loads(line) for line in results_path.read_text().splitlines()]
            records[0]["scores"] = [1, -1, 0, 0]
            write_jsonl(results_path, records)

            self.assertEqual(ex.main(["summarize", "--work-dir", str(work)]), 1)
            self.assertFalse((work / "summary.json").exists())
            self.assertFalse((work / "summary.md").exists())

    def test_summarize_refuses_missing_manifest(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            env = fake_environment(tmp)
            driver = fake_driver(tmp)
            work = tmp / "work"
            self.assertEqual(
                ex.main(
                    run_args(
                        work,
                        tmp,
                        env,
                        driver,
                        "--pilot-size",
                        "3",
                        "--runtime-budget",
                        "10.0",
                        "--max-main-seeds",
                        "2",
                    )
                ),
                0,
            )
            (work / "manifest.json").unlink()
            self.assertEqual(ex.main(["summarize", "--work-dir", str(work)]), 1)
            self.assertFalse((work / "summary.json").exists())

    def test_main_refuses_plan_setting_changes(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            env = fake_environment(tmp)
            driver = fake_driver(tmp)
            work = tmp / "work"
            common = [
                "--work-dir",
                str(work),
                "--driver",
                str(driver),
                "--repo-root",
                str(env["repo"]),
                "--ninone-root",
                str(env["ninone"]),
                "--worker",
                str(env["worker"]),
                "--python",
                sys.executable,
            ]
            self.assertEqual(
                ex.main(["pilot", *common, "--pilot-size", "3"]), 0
            )
            self.assertEqual(
                ex.main(
                    [
                        "plan",
                        *common,
                        "--target-precision",
                        "1.0",
                        "--max-main-seeds",
                        "2",
                    ]
                ),
                0,
            )
            self.assertEqual(
                ex.main(["main", *common, "--decision-limit", "999"]), 1
            )
            self.assertFalse((work / "summary.json").exists())
            self.assertEqual(
                ex.main(["main", *common, "--mode", "first-only"]), 1
            )
            self.assertFalse((work / "summary.json").exists())

    def test_main_refuses_changed_adapter_identity(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            env = fake_environment(tmp)
            driver = fake_driver(tmp)
            work = tmp / "work"
            common = [
                "--work-dir",
                str(work),
                "--driver",
                str(driver),
                "--repo-root",
                str(env["repo"]),
                "--ninone-root",
                str(env["ninone"]),
                "--worker",
                str(env["worker"]),
                "--python",
                sys.executable,
            ]
            self.assertEqual(ex.main(["pilot", *common, "--pilot-size", "3"]), 0)
            self.assertEqual(
                ex.main(
                    [
                        "plan",
                        *common,
                        "--target-precision",
                        "1.0",
                        "--max-main-seeds",
                        "2",
                    ]
                ),
                0,
            )
            env["worker"].write_text("# worker changed after the plan\n", encoding="utf-8")
            self.assertEqual(ex.main(["main", *common]), 1)
            self.assertFalse((work / "summary.json").exists())

    def test_main_refuses_unfrozen_plan(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            env = fake_environment(tmp)
            driver = fake_driver(tmp)
            work = tmp / "work"
            work.mkdir()
            ex.write_json(work / "plan.json", {"status": "draft", "sample": {}})
            code = ex.main(
                [
                    "main",
                    "--work-dir",
                    str(work),
                    "--driver",
                    str(driver),
                    "--repo-root",
                    str(env["repo"]),
                    "--ninone-root",
                    str(env["ninone"]),
                    "--worker",
                    str(env["worker"]),
                ]
            )
            self.assertEqual(code, 1)
            self.assertFalse((work / "summary.json").exists())

    def test_main_inherits_plan_bootstrap_settings(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            env = fake_environment(tmp)
            driver = fake_driver(tmp)
            work = tmp / "work"
            common = self._common_args(work, env, driver)
            self.assertEqual(ex.main(["pilot", *common, "--pilot-size", "3"]), 0)
            self.assertEqual(
                ex.main(
                    [
                        "plan",
                        *common,
                        "--target-precision",
                        "1.0",
                        "--max-main-seeds",
                        "2",
                        "--bootstrap-seed",
                        "7",
                        "--bootstrap-resamples",
                        "200",
                    ]
                ),
                0,
            )
            plan = self._read(work / "plan.json")
            self.assertEqual(plan["analysis"]["bootstrap_seed"], 7)
            self.assertEqual(plan["analysis"]["bootstrap_resamples"], 200)

            # An explicitly different value is still rejected before the driver runs.
            self.assertEqual(
                ex.main(
                    [
                        "main",
                        *common,
                        "--determinism-check",
                        "0",
                        "--bootstrap-seed",
                        "8",
                    ]
                ),
                1,
            )
            self.assertFalse((work / "main" / "results.jsonl").exists())
            self.assertFalse((work / "summary.json").exists())

            # Omitted analysis settings inherit the frozen plan.
            self.assertEqual(
                ex.main(["main", *common, "--determinism-check", "0"]), 0
            )
            summary = self._read(work / "summary.json")
            self.assertEqual(summary["uncertainty"]["bootstrap_seed"], 7)
            self.assertEqual(summary["uncertainty"]["resamples"], 200)
            self.assertFalse((work / "main" / "failures.json").exists())

    def test_plan_inherits_pilot_configuration_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            env = fake_environment(tmp)
            driver = fake_driver(tmp)
            work = tmp / "work"
            common = self._common_args(work, env, driver)
            self.assertEqual(
                ex.main(
                    [
                        "pilot",
                        *common,
                        "--pilot-size",
                        "3",
                        "--mode",
                        "mixed",
                        "--decision-limit",
                        "50",
                    ]
                ),
                0,
            )
            self.assertEqual(
                ex.main(
                    [
                        "plan",
                        *common,
                        "--target-precision",
                        "1.0",
                        "--max-main-seeds",
                        "2",
                    ]
                ),
                0,
            )
            plan = self._read(work / "plan.json")
            self.assertEqual(plan["mode"], "mixed")
            self.assertEqual(plan["decision_limit"], 50)
            self.assertEqual(plan["pilot"]["count"], 3)

            drift = tmp / "drift"
            drift_common = self._common_args(drift, env, driver)
            self.assertEqual(
                ex.main(
                    [
                        "pilot",
                        *drift_common,
                        "--pilot-size",
                        "3",
                        "--mode",
                        "mixed",
                        "--decision-limit",
                        "50",
                    ]
                ),
                0,
            )
            self.assertEqual(
                ex.main(
                    [
                        "plan",
                        *drift_common,
                        "--target-precision",
                        "1.0",
                        "--max-main-seeds",
                        "2",
                        "--mode",
                        "first-only",
                    ]
                ),
                1,
            )
            self.assertFalse((drift / "plan.json").exists())
            self.assertEqual(
                ex.main(
                    [
                        "plan",
                        *drift_common,
                        "--target-precision",
                        "1.0",
                        "--max-main-seeds",
                        "2",
                        "--decision-limit",
                        "51",
                    ]
                ),
                1,
            )
            self.assertFalse((drift / "plan.json").exists())

    def test_source_change_during_pilot_invalidates_pilot(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            env = fake_environment(tmp)
            driver = fake_driver(tmp)
            work = tmp / "work"
            common = self._common_args(work, env, driver)
            os.environ["FAKE_DRIVER_MODE"] = "mutate_sources_pilot"
            try:
                code = ex.main(["pilot", *common, "--pilot-size", "3"])
            finally:
                os.environ.pop("FAKE_DRIVER_MODE", None)
            self.assertEqual(code, 1)
            self.assertFalse((work / "pilot" / "report.json").exists())
            failures = self._read(work / "pilot" / "failures.json")
            self.assertIn("identity_drifted", {p["kind"] for p in failures["problems"]})
            self.assertEqual(failures["stage"], "pilot-identity-drift")
            # A pilot without a report cannot support a plan.
            with redirect_stderr(StringIO()):
                self.assertEqual(
                    ex.main(
                        [
                            "plan",
                            *common,
                            "--target-precision",
                            "1.0",
                            "--max-main-seeds",
                            "2",
                        ]
                    ),
                    1,
                )
            self.assertFalse((work / "plan.json").exists())

    def test_source_change_during_main_invalidates_run(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            env = fake_environment(tmp)
            driver = fake_driver(tmp)
            work = tmp / "work"
            common = self._common_args(work, env, driver)
            self.assertEqual(ex.main(["pilot", *common, "--pilot-size", "3"]), 0)
            self.assertEqual(
                ex.main(
                    [
                        "plan",
                        *common,
                        "--target-precision",
                        "1.0",
                        "--max-main-seeds",
                        "2",
                    ]
                ),
                0,
            )
            original_worker = env["worker"].read_bytes()
            original_ninone = (
                env["ninone"] / "ninone" / "agent_heuristic.py"
            ).read_bytes()
            os.environ["FAKE_DRIVER_MODE"] = "mutate_sources"
            try:
                code = ex.main(["main", *common, "--determinism-check", "0"])
            finally:
                os.environ.pop("FAKE_DRIVER_MODE", None)
            self.assertEqual(code, 1)
            self.assertFalse((work / "summary.json").exists())
            self.assertFalse((work / "summary.md").exists())
            failures = self._read(work / "main" / "failures.json")
            self.assertIn("identity_drifted", {p["kind"] for p in failures["problems"]})
            run_record = self._read(work / "main" / "run.json")
            self.assertEqual(run_record["identity_check"]["status"], "drifted")

            pre_worker = (
                work / "evidence" / "main-pre-run" / "sources" / "worker" / "worker.py"
            )
            self.assertEqual(pre_worker.read_bytes(), original_worker)
            pre_ninone = (
                work
                / "evidence"
                / "main-pre-run"
                / "sources"
                / "ninone_sources"
                / "ninone"
                / "agent_heuristic.py"
            )
            self.assertEqual(pre_ninone.read_bytes(), original_ninone)
            post_worker = (
                work / "evidence" / "main-post-run" / "sources" / "worker" / "worker.py"
            )
            self.assertIn(
                "mutated during the run", post_worker.read_text(encoding="utf-8")
            )
            self.assertNotEqual(
                (env["ninone"] / "ninone" / "agent_heuristic.py").read_bytes(),
                original_ninone,
            )

            manifest = self._read(work / "manifest.json")
            self.assertEqual(
                manifest["evidence"]["repository_capture_stage"], "main-pre-run"
            )
            self.assertEqual(
                manifest["evidence"]["identity_check"]["status"], "drifted"
            )
            files = {
                (item["label"], item["path"]): item
                for item in manifest["evidence"]["source_snapshots"]["files"]
            }
            worker_entry = next(
                item for (label, _path), item in files.items() if label == "worker"
            )
            self.assertEqual(worker_entry["sha256"], ex.sha256_bytes(original_worker))
            self.assertEqual(
                worker_entry["snapshot"],
                "evidence/main-pre-run/sources/worker/worker.py",
            )
            self.assertFalse(
                manifest["evidence"]["source_snapshots"]["missing_evidence"]
            )

            # A later summarize must not rescue the drifted run either.
            failures_path = work / "main" / "failures.json"
            preserved = failures_path.read_bytes()
            with redirect_stderr(StringIO()):
                self.assertEqual(ex.main(["summarize", "--work-dir", str(work)]), 1)
            self.assertEqual(failures_path.read_bytes(), preserved)
            self.assertTrue((work / "main" / "summarize_failures.json").is_file())
            self.assertFalse((work / "summary.json").exists())
            self.assertFalse((work / "summary.md").exists())

    def test_summarize_preserves_original_main_failure(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            env = fake_environment(tmp)
            driver = fake_driver(tmp)
            work = tmp / "work"
            os.environ["FAKE_DRIVER_MODE"] = "exit"
            try:
                code = ex.main(
                    run_args(
                        work,
                        tmp,
                        env,
                        driver,
                        "--pilot-size",
                        "3",
                        "--runtime-budget",
                        "10.0",
                        "--max-main-seeds",
                        "2",
                        "--determinism-check",
                        "0",
                    )
                )
            finally:
                os.environ.pop("FAKE_DRIVER_MODE", None)
            self.assertEqual(code, 1)
            failures_path = work / "main" / "failures.json"
            preserved = failures_path.read_bytes()
            self.assertIn("fake driver failure", preserved.decode("utf-8"))

            self.assertEqual(ex.main(["summarize", "--work-dir", str(work)]), 1)
            self.assertEqual(failures_path.read_bytes(), preserved)
            derived = self._read(work / "main" / "summarize_failures.json")
            self.assertEqual(derived["status"], "failed")
            self.assertEqual(derived["stage"], "summarize")
            self.assertFalse(derived["normal_summary_published"])
            self.assertTrue(derived["problems"])
            self.assertFalse((work / "summary.json").exists())
            self.assertFalse((work / "summary.md").exists())

    def test_successful_recomputation_clears_stale_derived_failure(self):
        with tempfile.TemporaryDirectory() as raw:
            work, _, _ = self._successful_run(Path(raw))
            results_path = work / "main" / "results.jsonl"
            original = results_path.read_bytes()
            results_path.write_bytes(original + b"not valid JSON\n")
            self.assertEqual(ex.main(["summarize", "--work-dir", str(work)]), 1)
            diagnostic = work / "main" / "summarize_failures.json"
            self.assertTrue(diagnostic.exists())
            self.assertFalse((work / "summary.json").exists())

            # Restoring the exact original bytes also restores the manifest
            # binding; this is not permission to repair a failed main run.
            results_path.write_bytes(original)
            self.assertEqual(ex.main(["summarize", "--work-dir", str(work)]), 0)
            self.assertFalse(diagnostic.exists())
            self.assertEqual(self._read(work / "summary.json")["status"], "ok")

    def test_summarize_rejects_flags_it_does_not_use(self):
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw) / "work"
            work.mkdir()
            for extra in (
                ["--mode", "mixed"],
                ["--bootstrap-seed", "5"],
                ["--bootstrap-resamples", "200"],
                ["--driver", "x"],
                ["--pilot-size", "3"],
            ):
                with self.assertRaises(SystemExit) as caught, redirect_stderr(
                    StringIO()
                ):
                    ex.main(["summarize", "--work-dir", str(work), *extra])
                self.assertEqual(caught.exception.code, 2)

    def test_plan_main_summarize_require_explicit_work_dir(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            env = fake_environment(tmp)
            driver = fake_driver(tmp)
            base = [
                "--driver",
                str(driver),
                "--repo-root",
                str(env["repo"]),
                "--ninone-root",
                str(env["ninone"]),
                "--worker",
                str(env["worker"]),
                "--python",
                sys.executable,
            ]
            with redirect_stderr(StringIO()):
                self.assertEqual(ex.main(["plan", *base]), 1)
                self.assertEqual(ex.main(["main", *base]), 1)
                self.assertEqual(ex.main(["summarize"]), 1)

    def test_complete_records_with_nonzero_exit_publish_no_summary(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            env = fake_environment(tmp)
            driver = fake_driver(tmp)
            work = tmp / "work"
            os.environ["FAKE_DRIVER_MODE"] = "exit_after_records"
            try:
                code = ex.main(
                    run_args(
                        work,
                        tmp,
                        env,
                        driver,
                        "--pilot-size",
                        "3",
                        "--runtime-budget",
                        "10.0",
                        "--max-main-seeds",
                        "2",
                        "--determinism-check",
                        "0",
                    )
                )
            finally:
                os.environ.pop("FAKE_DRIVER_MODE", None)
            self.assertEqual(code, 1)
            self.assertFalse((work / "summary.json").exists())
            self.assertFalse((work / "summary.md").exists())
            records = [
                json.loads(line)
                for line in (work / "main" / "results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(records), 2 * len(ex.PLACEMENTS))
            failures = self._read(work / "main" / "failures.json")
            kinds = {p["kind"] for p in failures["problems"]}
            self.assertIn("nonzero_exit", kinds)
            self.assertEqual(len(failures["preserved_results"]), len(records))
            self.assertIn(
                "after complete records", json.dumps(failures["problems"])
            )


if __name__ == "__main__":
    unittest.main()
