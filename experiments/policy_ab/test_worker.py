"""Tests for the one-off policy A/B JSON Lines worker.

Standard library only. Protocol tests run without the external repository; the
integration tests import the original heuristic from `../ninone-mahjong-agent`
(or `$POLICY_AB_NINONE_ROOT`, then `$NINONE_ROOT`) and skip explicitly when
that checkout is absent.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import worker  # noqa: E402 - sibling module, imported after the path insert

WORKER_PATH = HERE / "worker.py"


def _default_ninone_root() -> Path:
    for name in ("POLICY_AB_NINONE_ROOT", "NINONE_ROOT"):
        override = os.environ.get(name)
        if override:
            return Path(override)
    # The checkout is a sibling of this repository, not a package dependency.
    return HERE.parents[2] / "ninone-mahjong-agent"


NINONE_ROOT = _default_ninone_root()
NINONE_HEURISTIC = NINONE_ROOT.joinpath("ninone", "policies", "agent_heuristic.py")
NINONE_SKIP = f"original ninone heuristic not found at {NINONE_HEURISTIC}"
NINONE_AVAILABLE = NINONE_HEURISTIC.is_file()

_POLICY: worker.PolicyCall | None = None


def _policy() -> worker.PolicyCall:
    global _POLICY
    if _POLICY is None:
        _POLICY = worker.load_policy(str(NINONE_ROOT))
    return _POLICY


def _payload(candidates, hand=None, missing_suit=None) -> dict:
    return {
        "hand": list(hand if hand is not None else ["m1", "m2"]),
        "missing_suit": missing_suit,
        "candidates": candidates,
    }


def _discards(hand) -> list[dict]:
    return [
        {"id": f"discard:{index}", "type": "discard", "channel": "turn", "code": code, "tile_id": index + 1}
        for index, code in enumerate(hand)
    ]


EXCHANGE_HAND = ["m1", "m2", "m3", "m4", "p1", "p2", "p3", "s1", "s2", "s3", "s3", "s4", "s5", "s6"]
DISCRIMINATING_HAND = ["p1", "p1", "p2", "p2", "p3", "m7", "m8", "m8",
                     "m1", "m2", "m3", "m4", "m5", "m6"]
KONG_HAND = ["m1", "m1", "m1", "m2", "m3", "m4", "p1", "p2", "p3", "p7", "p8", "p9", "m9"]
CONCEALED_HAND = ["m1", "m1", "m1", "m1", "m2", "m3", "m4", "p1", "p2", "p3", "p7", "p8", "p9", "s5"]
ADDED_HAND = ["m1", "m1", "m1", "m2", "m3", "m4", "p1", "p2", "p3", "p7", "p8", "p9", "s5", "m9"]

# One case per decision phase, with the id the original heuristic must choose.
# Exchange and missing-suit prefer the least represented suit in EXCHANGE_HAND
# (m=4, p=3, s=7). Discard prefers the own missing suit before tile utility: in
# DISCRIMINATING_HAND the best p discard beats the best m discard on that rule
# even though p3's utility (3.0) is worse than the best m discard's (1.5).
PHASE_CASES = [
    (
        "exchange_three",
        _payload(
            [
                {"id": "exchange:0", "type": "exchange_three", "channel": "phase", "suit": "m",
                 "tile_ids": [1, 2, 3], "tile_kinds": ["m1", "m2", "m3"]},
                {"id": "exchange:1", "type": "exchange_three", "channel": "phase", "suit": "p",
                 "tile_ids": [5, 6, 7], "tile_kinds": ["p1", "p2", "p3"]},
                {"id": "exchange:2", "type": "exchange_three", "channel": "phase", "suit": "s",
                 "tile_ids": [8, 9, 10], "tile_kinds": ["s1", "s2", "s3"]},
            ],
            hand=EXCHANGE_HAND,
        ),
        "exchange:1",
    ),
    (
        "choose_missing_suit",
        _payload(
            [
                {"id": "missing:m", "type": "choose_missing_suit", "channel": "phase", "suit": "m"},
                {"id": "missing:p", "type": "choose_missing_suit", "channel": "phase", "suit": "p"},
                {"id": "missing:s", "type": "choose_missing_suit", "channel": "phase", "suit": "s"},
            ],
            hand=EXCHANGE_HAND,
        ),
        "missing:p",
    ),
    (
        "discard",
        _payload(_discards(DISCRIMINATING_HAND), hand=DISCRIMINATING_HAND, missing_suit="p"),
        "discard:4",
    ),
    (
        "win",
        _payload(
            [
                {"id": "discard:0", "type": "discard", "channel": "turn", "code": "m1", "tile_id": 1},
                {"id": "win:0", "type": "win", "channel": "response",
                 "score": {"legal": True, "fan": 1, "capped_fan": 1, "value": 0}},
            ],
            hand=KONG_HAND,
        ),
        "win:0",
    ),
    (
        "pong_or_pass",
        _payload(
            [
                {"id": "pong:0", "type": "pong", "channel": "response", "code": "m1", "consume_ids": [1, 2]},
                {"id": "pass:0", "type": "pass", "channel": "response"},
            ],
            hand=KONG_HAND,
        ),
        "pass:0",
    ),
    (
        "kong_or_pass",
        _payload(
            [
                {"id": "kong:0", "type": "kong", "channel": "response", "code": "m1", "consume_ids": [1, 2, 3]},
                {"id": "pass:0", "type": "pass", "channel": "response"},
            ],
            hand=KONG_HAND,
        ),
        "pass:0",
    ),
    (
        "kong_only",
        _payload(
            [{"id": "kong:0", "type": "kong", "channel": "response", "code": "m1", "consume_ids": [1, 2, 3]}],
            hand=KONG_HAND,
        ),
        "kong:0",
    ),
    (
        "concealed_kong_or_discard",
        _payload(
            [
                {"id": "concealed:0", "type": "concealed_kong", "channel": "turn", "code": "m1",
                 "consume_ids": [1, 2, 3, 4]},
                {"id": "discard:0", "type": "discard", "channel": "turn", "code": "s5", "tile_id": 14},
            ],
            hand=CONCEALED_HAND,
        ),
        "discard:0",
    ),
    (
        "added_kong_or_discard",
        _payload(
            [
                {"id": "added:0", "type": "added_kong", "channel": "turn", "code": "m1", "tile_id": 14,
                 "meld_index": 1},
                {"id": "discard:0", "type": "discard", "channel": "turn", "code": "m9", "tile_id": 13},
            ],
            hand=ADDED_HAND,
        ),
        "discard:0",
    ),
    (
        "added_kong_only",
        _payload(
            [{"id": "added:0", "type": "added_kong", "channel": "turn", "code": "m1", "tile_id": 14,
              "meld_index": 1}],
            hand=ADDED_HAND,
        ),
        "added:0",
    ),
]

ALL_TYPES = [
    "exchange_three",
    "choose_missing_suit",
    "discard",
    "pong",
    "kong",
    "concealed_kong",
    "added_kong",
    "pass",
    "win",
]

KNOWN_TYPE_FIELDS = {
    "exchange_three": {"suit": "m", "tile_kinds": ["m1", "m2", "m3"]},
    "choose_missing_suit": {"suit": "p"},
    "discard": {"code": "m1"},
    "pong": {"code": "m1"},
    "kong": {"code": "m1"},
    "concealed_kong": {"code": "m1"},
    "added_kong": {"code": "m1"},
}

FAKE_NOISY_HEURISTIC = '''\
import sys

print('{"id": "import-noise"}')


def choose_action(request):
    print('{"id": "call-noise"}')
    print("call diagnostic", file=sys.stderr)
    return request["legal_actions"][0]
'''

INVALID_PAYLOADS = [
    ("not an object", []),
    ("missing hand", {"missing_suit": None, "candidates": [{"id": "a", "type": "pass"}]}),
    ("hand is not a list", {"hand": "m1", "missing_suit": None,
                            "candidates": [{"id": "a", "type": "pass"}]}),
    ("hand has a non-tile", {"hand": ["m10"], "missing_suit": None,
                             "candidates": [{"id": "a", "type": "pass"}]}),
    ("hand has a non-string", {"hand": [1], "missing_suit": None,
                               "candidates": [{"id": "a", "type": "pass"}]}),
    ("missing missing_suit", {"hand": ["m1"], "candidates": [{"id": "a", "type": "pass"}]}),
    ("bad missing_suit", {"hand": ["m1"], "missing_suit": "x",
                          "candidates": [{"id": "a", "type": "pass"}]}),
    ("non-string missing_suit", {"hand": ["m1"], "missing_suit": 3,
                                 "candidates": [{"id": "a", "type": "pass"}]}),
    ("missing candidates", {"hand": ["m1"], "missing_suit": None}),
    ("empty candidates", {"hand": ["m1"], "missing_suit": None, "candidates": []}),
    ("candidates is not a list", {"hand": ["m1"], "missing_suit": None, "candidates": "pass"}),
    ("candidate is not an object", {"hand": ["m1"], "missing_suit": None, "candidates": [1]}),
    ("candidate has no id", {"hand": ["m1"], "missing_suit": None,
                             "candidates": [{"type": "pass"}]}),
    ("candidate id is empty", {"hand": ["m1"], "missing_suit": None,
                               "candidates": [{"id": "", "type": "pass"}]}),
    ("candidate id is not a string", {"hand": ["m1"], "missing_suit": None,
                                      "candidates": [{"id": 1, "type": "pass"}]}),
    ("duplicate candidate id", {"hand": ["m1"], "missing_suit": None,
                                "candidates": [{"id": "a", "type": "pass"},
                                               {"id": "a", "type": "pass"}]}),
    ("candidate has no type", {"hand": ["m1"], "missing_suit": None,
                               "candidates": [{"id": "a"}]}),
    ("candidate type is not a string", {"hand": ["m1"], "missing_suit": None,
                                        "candidates": [{"id": "a", "type": 3}]}),
    ("exchange has no suit", {"hand": ["m1"], "missing_suit": None,
                              "candidates": [{"id": "e", "type": "exchange_three",
                                               "tile_kinds": ["m1", "m2", "m3"]}]}),
    ("exchange has a bad suit", {"hand": ["m1"], "missing_suit": None,
                                 "candidates": [{"id": "e", "type": "exchange_three", "suit": "x",
                                                  "tile_kinds": ["m1", "m2", "m3"]}]}),
    ("exchange has too few tile kinds", {"hand": ["m1"], "missing_suit": None,
                                         "candidates": [{"id": "e", "type": "exchange_three", "suit": "m",
                                                          "tile_kinds": ["m1", "m2"]}]}),
    ("exchange tile kinds are not a list", {"hand": ["m1"], "missing_suit": None,
                                            "candidates": [{"id": "e", "type": "exchange_three", "suit": "m",
                                                             "tile_kinds": "m1m2m3"}]}),
    ("exchange tile kinds mix suits", {"hand": ["m1"], "missing_suit": None,
                                       "candidates": [{"id": "e", "type": "exchange_three", "suit": "m",
                                                        "tile_kinds": ["m1", "p1", "m3"]}]}),
    ("exchange tile kinds leave the declared suit", {"hand": ["m1"], "missing_suit": None,
                                                      "candidates": [{"id": "e", "type": "exchange_three", "suit": "p",
                                                                       "tile_kinds": ["m1", "m2", "m3"]}]}),
    ("exchange tile kinds hold a bad code", {"hand": ["m1"], "missing_suit": None,
                                              "candidates": [{"id": "e", "type": "exchange_three", "suit": "m",
                                                               "tile_kinds": ["m1", "m2", "m10"]}]}),
    ("choose_missing_suit has no suit", {"hand": ["m1"], "missing_suit": None,
                                         "candidates": [{"id": "c", "type": "choose_missing_suit"}]}),
    ("choose_missing_suit has a bad suit", {"hand": ["m1"], "missing_suit": None,
                                            "candidates": [{"id": "c", "type": "choose_missing_suit", "suit": "z"}]}),
    ("discard has no code", {"hand": ["m1"], "missing_suit": None,
                             "candidates": [{"id": "d", "type": "discard"}]}),
    ("discard has a bad code", {"hand": ["m1"], "missing_suit": None,
                                "candidates": [{"id": "d", "type": "discard", "code": "m10"}]}),
    ("pong has a bad code", {"hand": ["m1"], "missing_suit": None,
                             "candidates": [{"id": "g", "type": "pong", "code": "10m"}]}),
    ("added_kong has no code", {"hand": ["m1"], "missing_suit": None,
                                "candidates": [{"id": "k", "type": "added_kong", "meld_index": 1}]}),
]


class BuildRequestTests(unittest.TestCase):
    def test_preserves_hand_own_missing_suit_and_candidate_list_identity(self):
        candidates = [
            {"id": "exchange:0", "type": "exchange_three", "suit": "m",
             "tile_ids": [1, 2, 3], "tile_kinds": ["m1", "m2", "m3"]},
            {"id": "exchange:1", "type": "exchange_three", "suit": "p",
             "tile_ids": [4, 5, 6], "tile_kinds": ["p1", "p2", "p3"]},
        ]
        payload = _payload(candidates, hand=EXCHANGE_HAND, missing_suit="s")

        request = worker.build_request(payload)

        self.assertEqual(request["observation"]["own_hand"], EXCHANGE_HAND)
        self.assertEqual(request["observation"]["own_missing_suit"], "s")
        self.assertIs(request["legal_actions"], candidates)
        for actual, expected in zip(request["legal_actions"], candidates):
            self.assertIs(actual, expected)

    def test_request_carries_only_the_heuristics_inputs(self):
        request = worker.build_request(_payload([{"id": "a", "type": "pass"}]))

        self.assertEqual(set(request), {"legal_actions", "observation"})
        self.assertEqual(set(request["observation"]), {"own_hand", "own_missing_suit"})
        self.assertIsNone(request["observation"]["own_missing_suit"])

    def test_all_ninone_types_and_unknown_types_pass_through_in_order(self):
        candidates = []
        for name in ALL_TYPES:
            candidate = {"id": f"{name}:0", "type": name}
            candidate.update(KNOWN_TYPE_FIELDS.get(name, {}))
            candidates.append(candidate)
        candidates.append({"id": "future_type:0", "type": "future_type"})

        request = worker.build_request(_payload(candidates))

        self.assertEqual([candidate["id"] for candidate in request["legal_actions"]],
                         [candidate["id"] for candidate in candidates])
        self.assertEqual([candidate["type"] for candidate in request["legal_actions"]],
                         [candidate["type"] for candidate in candidates])
        for actual, expected in zip(request["legal_actions"], candidates):
            self.assertIs(actual, expected)

    def test_extra_top_level_keys_stay_out_of_the_request(self):
        payload = _payload([{"id": "a", "type": "pass"}])
        payload["wall"] = ["m1", "m2", "m3"]
        payload["opponent_hands"] = {"seat2": ["p1"], "seat3": ["s1"]}

        request = worker.build_request(payload)

        self.assertEqual(set(request), {"legal_actions", "observation"})
        serialized = json.dumps(request)
        for leaked in ("wall", "opponent_hands", "wall\"", "p1", "s1"):
            self.assertNotIn(leaked, serialized)

    def test_invalid_payloads_raise_worker_error(self):
        for label, payload in INVALID_PAYLOADS:
            with self.subTest(label):
                with self.assertRaises(worker.WorkerError):
                    worker.build_request(payload)


class DecideTests(unittest.TestCase):
    def test_selected_candidate_maps_to_bare_id(self):
        chosen = {"id": "discard:m3", "type": "discard", "code": "m3", "tile_id": 3}
        payload = _payload([{"id": "discard:m1", "type": "discard", "code": "m1", "tile_id": 1}, chosen])
        seen = []

        def choose(request):
            seen.append(request)
            return chosen

        response = worker.decide(payload, choose)

        self.assertEqual(response, {"id": "discard:m3"})
        self.assertEqual(set(response), {"id"})
        self.assertIs(seen[0]["legal_actions"], payload["candidates"])

    def test_repeated_request_replays_the_same_selection(self):
        candidates = [{"id": "pass:0", "type": "pass"}, {"id": "pong:0", "type": "pong", "code": "m1"}]
        payload = _payload(candidates)

        def choose(request):
            return request["legal_actions"][0]

        self.assertEqual(worker.decide(payload, choose), worker.decide(payload, choose))

    def test_mutated_candidate_id_is_an_explicit_error(self):
        candidate = {"id": "a", "type": "pass"}
        payload = _payload([candidate])

        def choose(request):
            request["legal_actions"][0]["id"] = "b"
            return request["legal_actions"][0]

        with self.assertRaises(worker.WorkerError) as raised:
            worker.decide(payload, choose)

        message = str(raised.exception)
        self.assertIn("mutated", message)
        self.assertIn("'a'", message)
        self.assertIn("'b'", message)

    def test_selected_candidate_keeps_its_original_id(self):
        candidate = {"id": "pass:0", "type": "pass"}
        payload = _payload([candidate])

        self.assertEqual(worker.decide(payload, lambda request: request["legal_actions"][0]),
                         {"id": "pass:0"})
        self.assertEqual(candidate["id"], "pass:0")

    def test_heuristic_stdout_does_not_escape_the_worker(self):
        payload = _payload([{"id": "a", "type": "pass"}])

        def choose(request):
            print('{"id": "call-noise"}')
            return request["legal_actions"][0]

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            response = worker.decide(payload, choose)

        self.assertEqual(response, {"id": "a"})
        self.assertEqual(captured.getvalue(), "")

    def test_unknown_selection_is_an_error(self):
        payload = _payload([{"id": "a", "type": "pass"}])

        with self.assertRaises(worker.WorkerError):
            worker.decide(payload, lambda request: {"id": "unknown", "type": "pass"})

    def test_copy_of_a_candidate_is_not_an_original_candidate(self):
        payload = _payload([{"id": "a", "type": "pass"}])

        with self.assertRaises(worker.WorkerError):
            worker.decide(payload, lambda request: dict(request["legal_actions"][0]))

    def test_non_object_selection_is_an_error(self):
        payload = _payload([{"id": "a", "type": "pass"}])

        for value in (None, [], "a"):
            with self.subTest(value):
                with self.assertRaises(worker.WorkerError):
                    worker.decide(payload, lambda request, value=value: value)

    def test_heuristic_exception_is_an_error(self):
        payload = _payload([{"id": "a", "type": "pass"}])

        def choose(request):
            raise KeyError("code")

        with self.assertRaises(worker.WorkerError) as raised:
            worker.decide(payload, choose)

        self.assertIn("KeyError", str(raised.exception))


class ServeTests(unittest.TestCase):
    def _serve(self, lines, choose):
        stdin = io.StringIO("".join(line + "\n" for line in lines))
        stdout = io.StringIO()
        status = worker.serve(choose, stdin, stdout)
        return status, stdout.getvalue().splitlines()

    def test_one_process_serves_many_requests(self):
        payloads = [_payload([{"id": name, "type": "pass"}]) for name in ("a", "b", "c")]
        lines = [json.dumps(payload) for payload in payloads]

        status, output = self._serve(lines, lambda request: request["legal_actions"][0])

        self.assertEqual(status, 0)
        self.assertEqual([json.loads(line) for line in output], [{"id": "a"}, {"id": "b"}, {"id": "c"}])

    def test_blank_lines_are_ignored(self):
        line = json.dumps(_payload([{"id": "a", "type": "pass"}]))
        status, output = self._serve(["", "   ", line], lambda request: request["legal_actions"][0])

        self.assertEqual(status, 0)
        self.assertEqual([json.loads(item) for item in output], [{"id": "a"}])

    def test_invalid_json_writes_one_error_and_stops(self):
        line = json.dumps(_payload([{"id": "a", "type": "pass"}]))
        status, output = self._serve([line, "{not json", line], lambda request: request["legal_actions"][0])

        self.assertEqual(status, 1)
        self.assertEqual(len(output), 2)
        self.assertEqual(json.loads(output[0]), {"id": "a"})
        error = json.loads(output[1])
        self.assertEqual(set(error), {"error"})
        self.assertTrue(error["error"])

    def test_policy_failure_never_falls_back(self):
        line = json.dumps(_payload([{"id": "a", "type": "pass"}]))

        def choose(request):
            raise RuntimeError("boom")

        status, output = self._serve([line, line], choose)

        self.assertEqual(status, 1)
        self.assertEqual(len(output), 1)
        self.assertEqual(set(json.loads(output[0])), {"error"})


class WorkerStartupTests(unittest.TestCase):
    def _run(self, args, stdin_text=""):
        return subprocess.run(
            [sys.executable, str(WORKER_PATH), *args],
            input=stdin_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )

    def test_missing_root_directory_fails_without_stdout(self):
        completed = self._run(["--ninone-root", str(HERE / "no-such-checkout")])

        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, "")
        self.assertIn("not a directory", completed.stderr)

    def test_root_without_the_original_heuristic_fails(self):
        with tempfile.TemporaryDirectory(prefix="idle-ninone-") as root:
            completed = self._run(["--ninone-root", root])

        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, "")
        self.assertIn("agent_heuristic.py", completed.stderr)

    def test_requires_ninone_root_argument(self):
        completed = self._run([])

        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, "")
        self.assertIn("--ninone-root", completed.stderr)


class FakeCheckoutProcessTests(unittest.TestCase):
    def _write_checkout(self, root: Path) -> None:
        package = root / "ninone" / "policies"
        package.mkdir(parents=True)
        (root / "ninone" / "__init__.py").write_text("", encoding="utf-8")
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "agent_heuristic.py").write_text(FAKE_NOISY_HEURISTIC, encoding="utf-8")

    def test_module_stdout_cannot_disguise_itself_as_protocol_output(self):
        line = json.dumps(_payload([{"id": "a", "type": "pass"}]))
        with tempfile.TemporaryDirectory(prefix="noisy-ninone-") as directory:
            root = Path(directory)
            self._write_checkout(root)
            completed = subprocess.run(
                [sys.executable, str(WORKER_PATH), "--ninone-root", str(root)],
                input=line + "\n",
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=120,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual([json.loads(item) for item in completed.stdout.splitlines()], [{"id": "a"}])
        self.assertEqual(len(completed.stdout.splitlines()), 1, completed.stdout)
        # Import-time and call-time module prints are preserved on stderr only.
        self.assertIn('{"id": "import-noise"}', completed.stderr)
        self.assertIn('{"id": "call-noise"}', completed.stderr)
        self.assertIn("call diagnostic", completed.stderr)

    def test_broken_checkout_fails_on_stderr_without_protocol_output(self):
        line = json.dumps(_payload([{"id": "a", "type": "pass"}]))
        with tempfile.TemporaryDirectory(prefix="broken-ninone-") as directory:
            root = Path(directory)
            self._write_checkout(root)
            (root / "ninone" / "policies" / "agent_heuristic.py").write_text(
                "def choose_action(request)\n    return {}\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [sys.executable, str(WORKER_PATH), "--ninone-root", str(root)],
                input=line + "\n",
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=120,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, "")
        self.assertIn("agent_heuristic.py", completed.stderr)


@unittest.skipUnless(NINONE_AVAILABLE, NINONE_SKIP)
class OriginalHeuristicTests(unittest.TestCase):
    def test_imports_the_original_module_file(self):
        choose = _policy()

        self.assertEqual(choose.__module__, "ninone.policies.agent_heuristic")
        self.assertEqual(Path(choose.__code__.co_filename).resolve(), NINONE_HEURISTIC.resolve())

    def test_every_decision_phase_selects_its_expected_candidate(self):
        for name, payload, expected in PHASE_CASES:
            with self.subTest(name):
                self.assertEqual(worker.decide(payload, _policy()), {"id": expected})

    def test_candidates_reach_the_heuristic_unfiltered(self):
        choose = _policy()
        recorded = []

        def recording(request):
            recorded.append(request)
            return choose(request)

        for name, payload, expected in PHASE_CASES:
            with self.subTest(name):
                recorded.clear()
                response = worker.decide(payload, recording)
                self.assertEqual(response, {"id": expected})
                self.assertIs(recorded[0]["legal_actions"], payload["candidates"])
                self.assertEqual([candidate["id"] for candidate in recorded[0]["legal_actions"]],
                                 [candidate["id"] for candidate in payload["candidates"]])
                self.assertEqual([candidate["type"] for candidate in recorded[0]["legal_actions"]],
                                 [candidate["type"] for candidate in payload["candidates"]])

    def test_observation_state_stays_on_the_requested_hand(self):
        choose = _policy()
        recorded = []

        def recording(request):
            recorded.append(request)
            return choose(request)

        payload = PHASE_CASES[2][1]  # discard with missing_suit = "p" and hand-local tiles
        worker.decide(payload, recording)

        self.assertEqual(set(recorded[0]["observation"]), {"own_hand", "own_missing_suit"})
        self.assertEqual(recorded[0]["observation"]["own_hand"], payload["hand"])
        self.assertEqual(recorded[0]["observation"]["own_missing_suit"], "p")

    def test_requests_replay_deterministically(self):
        for name, payload, expected in PHASE_CASES:
            with self.subTest(name):
                self.assertEqual(worker.decide(payload, _policy()), worker.decide(payload, _policy()))


@unittest.skipUnless(NINONE_AVAILABLE, NINONE_SKIP)
class WorkerProcessTests(unittest.TestCase):
    def _run(self, stdin_text):
        return subprocess.run(
            [sys.executable, str(WORKER_PATH), "--ninone-root", str(NINONE_ROOT)],
            input=stdin_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )

    def test_long_lived_process_serves_and_replays_requests(self):
        cases = [PHASE_CASES[index] for index in (0, 2, 3, 4, 5, 7)]
        requests = [payload for _, payload, _ in cases] + [cases[0][1]]
        expected = [identifier for _, _, identifier in cases] + [cases[0][2]]

        completed = self._run("\n".join(json.dumps(payload) for payload in requests) + "\n")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        lines = completed.stdout.splitlines()
        self.assertEqual(len(lines), len(requests), completed.stdout)
        responses = [json.loads(line) for line in lines]
        for response, identifier in zip(responses, expected):
            self.assertEqual(response, {"id": identifier})
        self.assertEqual(responses[-1], responses[0])

    def test_process_failure_reports_error_and_exits_nonzero(self):
        good = json.dumps(PHASE_CASES[5][1])
        bad = json.dumps(_payload([{"id": "a", "type": "pass"}], missing_suit="x"))

        completed = self._run("\n".join([good, bad, good]) + "\n")

        self.assertNotEqual(completed.returncode, 0)
        lines = completed.stdout.splitlines()
        self.assertEqual(len(lines), 2, completed.stdout)
        self.assertEqual(json.loads(lines[0]), {"id": PHASE_CASES[5][2]})
        error = json.loads(lines[1])
        self.assertEqual(set(error), {"error"})
        self.assertTrue(error["error"])


if __name__ == "__main__":
    unittest.main()
