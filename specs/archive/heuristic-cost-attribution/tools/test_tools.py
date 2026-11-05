"""Unit tests for the archived heuristic cost attribution tools.

Run from this directory:

    python -m unittest test_tools -v

These tests mock the sampler's Win32/dbghelp surface, so they need no target
build, no profiler output, and no elevated or debugger privileges.
"""

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sample_cpu_windows as sampler
import symbolize_inline

EXE_BYTES = b"MZ-sampled-image" + b"\x90" * 32
PDB_BYTES = b"sampled-pdb" + b"\x11" * 32
SAMPLE = {"t": 0.0, "rip": {"addr": 0x140001000, "symbol": "rip", "module": "app.exe",
                            "rva": 0, "file": None, "line": None, "disp": 0},
          "frames": []}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class FakeProc:
    def __init__(self, rc=0):
        self.pid = 4242
        self.returncode = None
        self._rc = rc
        self.alive = True
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        return None if self.alive else self._rc

    def wait(self):
        self.wait_calls += 1
        self.alive = False
        self.returncode = self._rc
        return self._rc

    def kill(self):
        self.killed = True
        self.alive = False


class FakeKernel:
    def __init__(self):
        self.open_process = 0x111
        self.suspend_results = []
        self.context_results = []
        self.resume_results = []
        self.on_suspend = None
        self.on_context = None
        self.suspend_calls = 0
        self.context_calls = 0
        self.resume_calls = 0
        self.closed = []

    def OpenProcess(self, access, inherit, pid):
        return self.open_process

    def CloseHandle(self, handle):
        self.closed.append(handle)
        return 1

    def SuspendThread(self, handle):
        self.suspend_calls += 1
        result = self.suspend_results.pop(0) if self.suspend_results else 0
        if self.on_suspend is not None:
            self.on_suspend(self.suspend_calls, result)
        return result

    def GetThreadContext(self, handle, pointer):
        self.context_calls += 1
        ok = self.context_results.pop(0) if self.context_results else 1
        if self.on_context is not None:
            self.on_context(self.context_calls, ok)
        return ok

    def ResumeThread(self, handle):
        self.resume_calls += 1
        return self.resume_results.pop(0) if self.resume_results else 0


class FakeWalker:
    def __init__(self, pcs=(0x140001000, 0x140002000, 0x140003000)):
        self.pcs = list(pcs)
        self.error = None
        self.on_walk = None
        self.walks = 0
        self.cleaned = False

    def stack(self, ctx):
        self.walks += 1
        if self.on_walk is not None:
            self.on_walk(self.walks)
        if self.error is not None:
            raise self.error
        return list(self.pcs)

    def cleanup(self):
        self.cleaned = True


class FakeSymbolizer:
    def __init__(self):
        self.cleaned = False

    def resolve(self, addr):
        return {"addr": addr, "symbol": "symbol_%x" % addr, "file": None,
                "line": None, "disp": 0, "module": "app.exe",
                "rva": addr - 0x140000000}

    def cleanup(self):
        self.cleaned = True


def write_image(directory, exe_bytes=EXE_BYTES, pdb_bytes=PDB_BYTES):
    """Writes a fake EXE/PDB pair and returns their absolute paths."""
    os.makedirs(directory, exist_ok=True)
    exe = os.path.abspath(os.path.join(directory, "app.exe"))
    pdb = os.path.abspath(os.path.join(directory, "app.pdb"))
    with open(exe, "wb") as f:
        f.write(exe_bytes)
    with open(pdb, "wb") as f:
        f.write(pdb_bytes)
    return exe, pdb


def write_sample_dir(directory, exe_bytes=EXE_BYTES, pdb_bytes=PDB_BYTES,
                     include_image=True, status="ok", samples=(SAMPLE,)):
    """Writes a sampler output directory as the sampler would leave it."""
    exe, pdb = write_image(directory, exe_bytes, pdb_bytes)
    with open(os.path.join(directory, "modules.json"), "w", encoding="utf-8") as f:
        json.dump([{"name": "app.exe", "path": exe, "base": 0x140000000,
                    "size": 0x2000}], f)
    data = {"status": status, "command": [], "pid": 1, "interval_ms": 1.0,
            "startup_skip_ms": 0.0, "max_depth": 96, "sampling_wall_s": 0.0,
            "samples": list(samples)}
    if include_image:
        data["image"] = {
            "exe": {"path": exe, "size": len(exe_bytes), "sha256": sha256(exe_bytes)},
            "pdb": {"path": pdb, "size": len(pdb_bytes), "sha256": sha256(pdb_bytes)},
        }
    with open(os.path.join(directory, "samples.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)
    return exe, pdb


class SamplerRun:
    """One mocked sampler invocation."""

    def __init__(self, tmpdir, target_rc=0, samples_before_exit=None):
        self.proc = FakeProc(target_rc)
        self.kernel = FakeKernel()
        self.walker = FakeWalker()
        self.symbolizer = FakeSymbolizer()
        self.image_dir = os.path.join(tmpdir, "image")
        self.exe, self.pdb = write_image(self.image_dir)
        if samples_before_exit is not None:
            self.walker.on_walk = lambda n: setattr(
                self.proc, "alive", False) if n >= samples_before_exit else None

    def run(self, outdir, extra_env=None):
        modules = [{"name": "app.exe", "path": self.exe, "base": 0x140000000,
                    "size": 0x2000}]
        env = {"SAMPLE_INTERVAL_MS": "0", "STARTUP_SKIP_MS": "0",
               "HOLD_SUSPENDED": "0"}
        env.update(extra_env or {})
        argv = ["sample_cpu_windows.py", outdir, "--", self.exe]
        stderr = io.StringIO()
        with mock.patch.dict(os.environ, env), \
             mock.patch.object(sys, "argv", argv), \
             mock.patch.object(sampler, "k32", self.kernel), \
             mock.patch.object(sampler, "EXIT_GRACE_MS", 5.0), \
             mock.patch.object(sampler, "Walker", lambda *a, **k: self.walker), \
             mock.patch.object(sampler, "Symbolizer",
                               lambda *a, **k: self.symbolizer), \
             mock.patch.object(sampler, "list_modules", lambda pid: modules), \
             mock.patch.object(sampler, "open_threads", lambda pid: {7: 0x222}), \
             mock.patch.object(sampler, "pick_workload_thread",
                               lambda handles: (7, {7: 3.0})), \
             mock.patch.object(sampler, "thread_cpu_ms", lambda handle: 1.0), \
             mock.patch.object(sampler.subprocess, "Popen",
                               lambda *a, **k: self.proc), \
             redirect_stderr(stderr), redirect_stdout(io.StringIO()):
            code = sampler.main()
        return code, stderr.getvalue()


class SamplerOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, True))
        self.outdir = os.path.join(self.tmp, "run")

    def samples_json(self):
        return os.path.join(self.outdir, "samples.json")

    def test_success_writes_valid_samples(self):
        run = SamplerRun(self.tmp, target_rc=0, samples_before_exit=3)
        code, _ = run.run(self.outdir)
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(self.samples_json()))
        with open(self.samples_json(), encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["status"], "ok")
        self.assertEqual(len(data["samples"]), 3)
        self.assertEqual(len(data["image"]["exe"]["sha256"]), 64)
        self.assertEqual(len(data["image"]["pdb"]["sha256"]), 64)
        self.assertFalse(run.proc.killed)
        self.assertEqual(run.kernel.resume_calls, run.kernel.suspend_calls)
        self.assertEqual(run.kernel.resume_calls, 3)
        self.assertIn(0x111, run.kernel.closed)
        self.assertIn(0x222, run.kernel.closed)
        self.assertTrue(run.walker.cleaned)
        self.assertTrue(run.symbolizer.cleaned)

    def test_nonzero_target_exit_is_a_failure(self):
        run = SamplerRun(self.tmp, target_rc=7, samples_before_exit=2)
        code, err = run.run(self.outdir)
        self.assertNotEqual(code, 0)
        self.assertIn("exited with code 7", err)
        self.assertFalse(os.path.exists(self.samples_json()))
        self.assertTrue(run.walker.cleaned)
        self.assertIn(0x111, run.kernel.closed)

    def test_zero_samples_is_a_failure(self):
        run = SamplerRun(self.tmp, target_rc=0, samples_before_exit=3)
        run.walker.pcs = []  # unwinds never produce a usable chain
        code, err = run.run(self.outdir)
        self.assertNotEqual(code, 0)
        self.assertIn("no samples", err)
        self.assertFalse(os.path.exists(self.samples_json()))

    def test_context_failure_while_target_runs_fails_and_resumes(self):
        run = SamplerRun(self.tmp, target_rc=0)
        run.kernel.context_results = [0]
        code, err = run.run(self.outdir)
        self.assertNotEqual(code, 0)
        self.assertIn("GetThreadContext failed", err)
        self.assertEqual(run.kernel.resume_calls, 1)
        self.assertFalse(os.path.exists(self.samples_json()))
        # The still-running child must be terminated, not waited on forever.
        self.assertTrue(run.proc.killed)
        self.assertGreaterEqual(run.proc.wait_calls, 1)

    def test_stack_exception_resumes_thread_and_fails(self):
        run = SamplerRun(self.tmp, target_rc=0)
        run.walker.error = RuntimeError("StackWalk64 failed")
        code, err = run.run(self.outdir, {"HOLD_SUSPENDED": "1"})
        self.assertNotEqual(code, 0)
        self.assertIn("StackWalk64 failed", err)
        self.assertEqual(run.kernel.resume_calls, 1)
        self.assertFalse(os.path.exists(self.samples_json()))

    def test_win32_suspend_resume_use_unsigned_failure_sentinel(self):
        self.assertIs(sampler.k32.SuspendThread.restype, sampler.wt.DWORD)
        self.assertIs(sampler.k32.ResumeThread.restype, sampler.wt.DWORD)

    def test_resume_failure_fails_run_and_terminates_child(self):
        for hold in ("0", "1"):
            with self.subTest(hold_suspended=hold):
                run = SamplerRun(self.tmp)
                run.kernel.resume_results = [0xFFFFFFFF, 0xFFFFFFFF]
                outdir = os.path.join(self.tmp, "resume-failure-" + hold)
                code, err = run.run(outdir, {"HOLD_SUSPENDED": hold})
                self.assertNotEqual(code, 0)
                self.assertIn("ResumeThread failed", err)
                self.assertTrue(run.proc.killed)
                self.assertFalse(os.path.exists(os.path.join(outdir, "samples.json")))

    def test_symbolization_exception_fails_without_valid_output(self):
        run = SamplerRun(self.tmp)
        run.symbolizer.resolve = mock.Mock(side_effect=RuntimeError("resolve failed"))
        code, err = run.run(self.outdir)
        self.assertNotEqual(code, 0)
        self.assertIn("resolve failed", err)
        self.assertEqual(run.kernel.resume_calls, 1)
        self.assertTrue(run.proc.killed)
        self.assertFalse(os.path.exists(self.samples_json()))

    def test_suspend_failure_while_target_runs_is_a_failure(self):
        run = SamplerRun(self.tmp)
        run.kernel.suspend_results = [0xFFFFFFFF]
        code, err = run.run(self.outdir)
        self.assertNotEqual(code, 0)
        self.assertIn("SuspendThread failed", err)
        self.assertEqual(run.kernel.resume_calls, 0)
        self.assertTrue(run.proc.killed)
        self.assertFalse(os.path.exists(self.samples_json()))

    def test_suspend_failure_during_normal_exit_is_not_a_failure(self):
        run = SamplerRun(self.tmp, target_rc=0, samples_before_exit=99)
        run.kernel.suspend_results = [0, 0xFFFFFFFF]
        run.kernel.on_suspend = lambda call, result: (
            setattr(run.proc, "alive", False) if result == 0xFFFFFFFF else None)
        code, err = run.run(self.outdir)
        self.assertEqual(code, 0, err)
        with open(self.samples_json(), encoding="utf-8") as f:
            self.assertEqual(len(json.load(f)["samples"]), 1)

    def test_context_failure_during_normal_exit_is_not_a_failure(self):
        run = SamplerRun(self.tmp, target_rc=0, samples_before_exit=99)
        run.kernel.context_results = [1, 0]
        run.kernel.on_context = lambda call, ok: (
            setattr(run.proc, "alive", False) if ok == 0 else None)
        code, err = run.run(self.outdir)
        self.assertEqual(code, 0, err)
        with open(self.samples_json(), encoding="utf-8") as f:
            self.assertEqual(len(json.load(f)["samples"]), 1)

    def test_nonempty_output_directory_is_refused(self):
        os.makedirs(self.outdir)
        stale = os.path.join(self.outdir, "samples.json")
        with open(stale, "w", encoding="utf-8") as f:
            json.dump({"status": "ok", "samples": [SAMPLE]}, f)
        run = SamplerRun(self.tmp)
        with mock.patch.object(sampler.subprocess, "Popen",
                               side_effect=AssertionError("target must not start")):
            stderr = io.StringIO()
            with mock.patch.object(sys, "argv",
                                   ["sample_cpu_windows.py", self.outdir,
                                    "--", run.exe]), \
                 redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                code = sampler.main()
        self.assertNotEqual(code, 0)
        self.assertIn("not empty", stderr.getvalue())
        with open(stale, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["status"], "ok")

    def test_missing_pdb_fails_before_sampling(self):
        run = SamplerRun(self.tmp)
        os.remove(run.pdb)
        code, err = run.run(self.outdir)
        self.assertNotEqual(code, 0)
        self.assertIn("PDB", err)
        self.assertFalse(os.path.exists(self.samples_json()))


class BuildIdentityTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, True))
        self.tmp = tmp

    def dir(self, name):
        return os.path.join(self.tmp, name)

    def test_recorded_build_is_accepted(self):
        write_sample_dir(self.dir("a"))
        modules, image = symbolize_inline.load_image(self.dir("a"))
        self.assertEqual(modules[0]["name"], "app.exe")
        self.assertEqual(symbolize_inline.check_same_image(
            [self.dir("a")])[0]["name"], "app.exe")

    def test_same_path_same_size_exe_replacement_is_rejected(self):
        exe, _ = write_sample_dir(self.dir("a"))
        replaced = bytes(byte ^ 0xFF for byte in EXE_BYTES)
        self.assertEqual(len(replaced), len(EXE_BYTES))
        with open(exe, "wb") as f:
            f.write(replaced)
        with self.assertRaises(SystemExit) as caught:
            symbolize_inline.load_image(self.dir("a"))
        self.assertIn("image changed since sampling", str(caught.exception))

    def test_same_path_same_size_pdb_replacement_is_rejected(self):
        _, pdb = write_sample_dir(self.dir("a"))
        replaced = bytes(byte ^ 0xFF for byte in PDB_BYTES)
        self.assertEqual(len(replaced), len(PDB_BYTES))
        with open(pdb, "wb") as f:
            f.write(replaced)
        with self.assertRaises(SystemExit) as caught:
            symbolize_inline.load_image(self.dir("a"))
        self.assertIn("PDB changed since sampling", str(caught.exception))

    def test_samples_without_recorded_hashes_fail_closed(self):
        write_sample_dir(self.dir("a"), include_image=False)
        with self.assertRaises(SystemExit) as caught:
            symbolize_inline.load_image(self.dir("a"))
        message = str(caught.exception)
        self.assertIn("no EXE SHA256", message)
        self.assertIn("resample", message)

    def test_failed_sampling_run_is_rejected(self):
        write_sample_dir(self.dir("a"), status="failed")
        with self.assertRaises(SystemExit) as caught:
            symbolize_inline.load_image(self.dir("a"))
        self.assertIn("failed or partial", str(caught.exception))

    def test_empty_sample_set_is_rejected(self):
        write_sample_dir(self.dir("a"), samples=())
        with self.assertRaises(SystemExit) as caught:
            symbolize_inline.load_image(self.dir("a"))
        self.assertIn("no samples", str(caught.exception))

    def test_missing_samples_json_is_rejected(self):
        exe, pdb = write_sample_dir(self.dir("a"))
        os.remove(os.path.join(self.dir("a"), "samples.json"))
        with self.assertRaises(SystemExit) as caught:
            symbolize_inline.load_image(self.dir("a"))
        self.assertIn("samples.json is missing", str(caught.exception))

    def test_identical_builds_aggregate(self):
        write_sample_dir(self.dir("a"))
        write_sample_dir(self.dir("b"))
        modules = symbolize_inline.check_same_image([self.dir("a"), self.dir("b")])
        self.assertEqual(modules[0]["name"], "app.exe")

    def test_mixed_builds_are_rejected(self):
        write_sample_dir(self.dir("a"))
        write_sample_dir(self.dir("b"), pdb_bytes=PDB_BYTES + b"other")
        with self.assertRaises(SystemExit) as caught:
            symbolize_inline.check_same_image([self.dir("a"), self.dir("b")])
        self.assertIn("refusing to mix builds", str(caught.exception))

    def test_missing_hash_blocks_aggregation_too(self):
        write_sample_dir(self.dir("a"))
        write_sample_dir(self.dir("b"), include_image=False)
        with self.assertRaises(SystemExit):
            symbolize_inline.check_same_image([self.dir("a"), self.dir("b")])

    def test_both_analysis_scripts_reject_failed_sampling(self):
        import aggregate_attribution

        write_sample_dir(self.dir("a"), status="failed")
        for module in (symbolize_inline, aggregate_attribution):
            with mock.patch.object(sys, "argv", ["tool.py", self.dir("a")]), \
                 redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    module.main()


if __name__ == "__main__":
    unittest.main()
