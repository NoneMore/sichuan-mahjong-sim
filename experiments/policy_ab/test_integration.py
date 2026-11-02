"""Opt-in end-to-end checks against a built policy_ab Cargo example.

Set POLICY_AB_DRIVER to the executable path. The heuristic checks additionally
need the adjacent ninone checkout, or POLICY_AB_NINONE_ROOT.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DRIVER = os.environ.get("POLICY_AB_DRIVER")
NINONE = Path(os.environ.get("POLICY_AB_NINONE_ROOT", ROOT.parent / "ninone-mahjong-agent"))
PLACEMENTS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


@unittest.skipUnless(DRIVER, "set POLICY_AB_DRIVER to a built policy_ab example")
class IntegrationTests(unittest.TestCase):
    def run_driver(self, directory, name, mode, *, worker=None, limit=10000):
        seeds = directory / "seeds.json"
        seeds.write_text(json.dumps([42, 107]), encoding="utf-8")
        output = directory / f"{name}.jsonl"
        command = [
            str(Path(DRIVER).resolve()), "--seeds", str(seeds),
            "--output", str(output), "--python", sys.executable,
            "--worker", str(worker or HERE / "worker.py"),
            "--ninone-root", str(NINONE), "--mode", mode,
            "--decision-limit", str(limit),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] if output.exists() else []
        return result, records

    def check_complete(self, records, symmetric=False):
        self.assertEqual(len(records), 12)
        for seed in [42, 107]:
            block = [record for record in records if record["seed"] == seed]
            self.assertEqual([tuple(record["a_seats"]) for record in block], PLACEMENTS)
            contrast_sum = 0
            for record in block:
                self.assertNotIn("error", record)
                scores = record["scores"]
                self.assertEqual(len(scores), 4)
                self.assertEqual(sum(scores), 0)
                self.assertGreater(record["decisions"], 0)
                self.assertTrue(record["end_reason"])
                a_seats = record["a_seats"]
                contrast_sum += sum(scores[i] for i in a_seats) - sum(scores[i] for i in range(4) if i not in a_seats)
            if symmetric:
                self.assertEqual(contrast_sum, 0)
                self.assertTrue(all(record["scores"] == block[0]["scores"] for record in block))

    @staticmethod
    def identity(records):
        # Keep only deterministic result fields; wall-clock diagnostics may vary.
        keys = ("seed", "a_seats", "scores", "end_reason", "decisions", "trace_hash", "trace_digest")
        return [{key: record[key] for key in keys if key in record} for record in records]

    def test_first_only_complete_symmetric_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, records = self.run_driver(directory, "first", "first-only")
            self.assertEqual(first.returncode, 0, first.stderr)
            self.check_complete(records, symmetric=True)
            second, replay = self.run_driver(directory, "replay", "first-only")
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(self.identity(records), self.identity(replay))

    @unittest.skipUnless((NINONE / "ninone/policies/agent_heuristic.py").is_file(), "original ninone checkout unavailable")
    def test_imported_heuristic_symmetry_and_mixed_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            symmetric, records = self.run_driver(directory, "symmetric", "heuristic-only")
            self.assertEqual(symmetric.returncode, 0, symmetric.stderr)
            self.check_complete(records, symmetric=True)
            mixed, records = self.run_driver(directory, "mixed", "mixed")
            self.assertEqual(mixed.returncode, 0, mixed.stderr)
            self.check_complete(records)
            repeat, replay = self.run_driver(directory, "repeat", "mixed")
            self.assertEqual(repeat.returncode, 0, repeat.stderr)
            self.assertEqual(self.identity(records), self.identity(replay))

    def test_unknown_selection_is_fatal_with_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            worker = directory / "unknown.py"
            worker.write_text(
                'import json, sys\nfor line in sys.stdin:\n print(json.dumps({"id": "not-a-candidate"}), flush=True)\n',
                encoding="utf-8",
            )
            result, records = self.run_driver(directory, "failure", "mixed", worker=worker)
            self.assertNotEqual(result.returncode, 0)
            failures = [record for record in records if "error" in record]
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["seed"], 42)
            self.assertEqual(failures[0]["a_seats"], [0, 1])
            self.assertIn("context", failures[0])
            self.assertFalse(any("scores" in record for record in failures))

    def test_limit_exhaustion_is_not_a_terminal_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, records = self.run_driver(Path(temporary), "limit", "first-only", limit=1)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(len(records), 1)
            self.assertIn("error", records[0])
            self.assertIn("context", records[0])
            self.assertNotIn("scores", records[0])


if __name__ == "__main__":
    unittest.main()
