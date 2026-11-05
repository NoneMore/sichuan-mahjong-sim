"""External, non-admin CPU sampler for one Windows process.

Produces, per sample of the workload thread:
  * instruction pointer (self cost) with symbol and source line,
  * the unwound call chain from `StackWalk64` (inclusive cost), which uses the
    image's x64 unwind data rather than frame pointers.

Two dbghelp contexts are used because each is bound to one process handle:
the walking context is initialized on the target process (so `StackWalk64`
reads target memory directly), and the symbolizing context is initialized on
the sampler process with the analysed image loaded at its live base address.
No debugger, elevation, or ETW kernel session is required.

Sampling a suspended thread at wall-clock intervals approximates CPU sampling
for the thread that is actually running, so the sampler picks the process
thread with the largest `GetThreadTimes` CPU delta and samples only that one.
The other threads of this single-threaded workload stay in near-zero-CPU
`ntdll` waits, and per-thread CPU totals are written to `sampler.log`.

`SAMPLE_INTERVAL_MS` is the sleep between samples, not the achieved cadence:
each sample also pays the suspend/context cost and, off-thread, the unwind and
symbolization cost. With 1 ms sleeps the measured cadence was about 4.5-5.1 ms
per sample, and with 5 ms sleeps plus unwinding while suspended about 12 ms.
Report the achieved cadence (sampling wall time / samples), not the sleep.

A run only counts as successful when the target itself exited with code 0 and
at least one sample was captured. A non-zero target exit, a sampling or
symbolization failure, and an empty sample set all exit non-zero and write no
`samples.json`, so the offline scripts cannot mistake a failed or partial run
for data. For the same reason the output directory must be fresh: a stale
`samples.json` from an earlier run would make a failed run look successful.
`samples.json` also records the SHA256 of the sampled EXE and PDB, and both
offline scripts refuse to symbolize files whose content no longer matches that
record.
"""


import ctypes
import ctypes.wintypes as wt
import hashlib
import json
import os
import subprocess
import sys
import time

def load_dbghelp():
    """Loads dbghelp from the first usable location, and reports which one."""
    candidates = []
    if os.environ.get("DBGHELP_DLL"):
        candidates.append(os.environ["DBGHELP_DLL"])
    candidates.append(r"C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\dbghelp.dll")
    candidates.append(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                                   "System32", "dbghelp.dll"))
    candidates.append("dbghelp.dll")
    errors = []
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate), candidate
        except OSError as error:
            errors.append("%s: %s" % (candidate, error))
    raise SystemExit("could not load dbghelp.dll; set DBGHELP_DLL\n  " +
                     "\n  ".join(errors))


k32 = ctypes.WinDLL("kernel32", use_last_error=True)
dbg, DBGHELP_PATH = load_dbghelp()

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
THREAD_SUSPEND_RESUME = 0x0002
THREAD_GET_CONTEXT = 0x0008
THREAD_QUERY_INFORMATION = 0x0040
CONTEXT_FULL = 0x0010000B  # CONTEXT_AMD64 | CONTROL | INTEGER | FLOATING_POINT
CONTEXT_SIZE = 0x4D0
OFF_RSP = 0x98
OFF_RBP = 0xA0
OFF_RIP = 0xF8
TH32CS_SNAPTHREAD = 0x00000004
TH32CS_SNAPMODULE = 0x00000008
TH32CS_SNAPMODULE32 = 0x00000010
MAX_PATH = 260
SYMOPT_UNDNAME = 0x00000002
SYMOPT_DEFERRED_LOADS = 0x00000004
SYMOPT_LOAD_LINES = 0x00000010
SYMBOL_INFO_NAME_OFFSET = 84
IMAGE_FILE_MACHINE_AMD64 = 0x8664
ADDR_MODE_FLAT = 3
STACKFRAME64_SIZE = 512
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "96"))
# Grace period used to tell "the target just exited" apart from a genuine
# suspend/context-capture failure.
EXIT_GRACE_MS = 500.0


class THREADENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wt.DWORD),
        ("cntUsage", wt.DWORD),
        ("th32ThreadID", wt.DWORD),
        ("th32OwnerProcessID", wt.DWORD),
        ("tpBasePri", wt.LONG),
        ("tpDeltaPri", wt.LONG),
        ("dwFlags", wt.DWORD),
    ]


class MODULEENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wt.DWORD),
        ("th32ModuleID", wt.DWORD),
        ("th32ProcessID", wt.DWORD),
        ("GlblcntUsage", wt.DWORD),
        ("ProccntUsage", wt.DWORD),
        ("modBaseAddr", ctypes.POINTER(ctypes.c_byte)),
        ("modBaseSize", wt.DWORD),
        ("hModule", wt.HMODULE),
        ("szModule", wt.WCHAR * 256),
        ("szExePath", wt.WCHAR * MAX_PATH),
    ]


class SYMBOL_INFOW(ctypes.Structure):
    _fields_ = [
        ("SizeOfStruct", wt.ULONG),
        ("TypeIndex", wt.ULONG),
        ("Reserved", ctypes.c_ulonglong * 2),
        ("Index", wt.ULONG),
        ("Size", wt.ULONG),
        ("ModBase", ctypes.c_ulonglong),
        ("Flags", wt.ULONG),
        ("Value", ctypes.c_ulonglong),
        ("Address", ctypes.c_ulonglong),
        ("Register", wt.ULONG),
        ("Scope", wt.ULONG),
        ("Tag", wt.ULONG),
        ("NameLen", wt.ULONG),
        ("MaxNameLen", wt.ULONG),
        ("Name", wt.WCHAR * 1),
    ]


class IMAGEHLP_LINEW64(ctypes.Structure):
    _fields_ = [
        ("SizeOfStruct", wt.DWORD),
        ("Key", ctypes.c_void_p),
        ("LineNumber", wt.DWORD),
        ("FileName", wt.LPWSTR),
        ("Address", ctypes.c_ulonglong),
    ]


k32.OpenProcess.restype = wt.HANDLE
k32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
k32.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
k32.Thread32First.argtypes = [wt.HANDLE, ctypes.POINTER(THREADENTRY32)]
k32.Thread32Next.argtypes = [wt.HANDLE, ctypes.POINTER(THREADENTRY32)]
k32.Module32FirstW.argtypes = [wt.HANDLE, ctypes.POINTER(MODULEENTRY32W)]
k32.Module32NextW.argtypes = [wt.HANDLE, ctypes.POINTER(MODULEENTRY32W)]
k32.OpenThread.restype = wt.HANDLE
k32.OpenThread.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
k32.SuspendThread.argtypes = [wt.HANDLE]
k32.SuspendThread.restype = wt.DWORD
k32.ResumeThread.argtypes = [wt.HANDLE]
k32.ResumeThread.restype = wt.DWORD
k32.GetThreadContext.argtypes = [wt.HANDLE, ctypes.c_void_p]
k32.GetThreadTimes.argtypes = [
    wt.HANDLE, ctypes.POINTER(wt.FILETIME), ctypes.POINTER(wt.FILETIME),
    ctypes.POINTER(wt.FILETIME), ctypes.POINTER(wt.FILETIME)]
k32.CloseHandle.argtypes = [wt.HANDLE]
k32.GetCurrentProcess.restype = wt.HANDLE

dbg.SymSetOptions.argtypes = [wt.DWORD]
dbg.SymSetOptions.restype = wt.DWORD
dbg.SymInitializeW.argtypes = [wt.HANDLE, wt.LPCWSTR, wt.BOOL]
dbg.SymLoadModuleExW.argtypes = [
    wt.HANDLE, wt.HANDLE, wt.LPCWSTR, wt.LPCWSTR, ctypes.c_ulonglong,
    wt.DWORD, ctypes.c_void_p, wt.DWORD]
dbg.SymLoadModuleExW.restype = ctypes.c_ulonglong
dbg.SymFromAddrW.argtypes = [
    wt.HANDLE, ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_ulonglong),
    ctypes.POINTER(SYMBOL_INFOW)]
dbg.SymGetLineFromAddrW64.argtypes = [
    wt.HANDLE, ctypes.c_ulonglong, ctypes.POINTER(wt.DWORD),
    ctypes.POINTER(IMAGEHLP_LINEW64)]
dbg.SymGetModuleBase64.argtypes = [wt.HANDLE, ctypes.c_ulonglong]
dbg.SymGetModuleBase64.restype = ctypes.c_ulonglong
dbg.SymFunctionTableAccess64.argtypes = [wt.HANDLE, ctypes.c_ulonglong]
dbg.SymFunctionTableAccess64.restype = ctypes.c_void_p
dbg.StackWalk64.argtypes = [
    wt.DWORD, wt.HANDLE, wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
dbg.StackWalk64.restype = wt.BOOL
dbg.SymCleanup.argtypes = [wt.HANDLE]


def list_threads(pid):
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    ids = []
    if not snap or snap == wt.HANDLE(-1).value:
        return ids
    entry = THREADENTRY32()
    entry.dwSize = ctypes.sizeof(THREADENTRY32)
    ok = k32.Thread32First(snap, ctypes.byref(entry))
    while ok:
        if entry.th32OwnerProcessID == pid:
            ids.append(entry.th32ThreadID)
        ok = k32.Thread32Next(snap, ctypes.byref(entry))
    k32.CloseHandle(snap)
    return ids


def list_modules(pid):
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid)
    mods = []
    if not snap or snap == wt.HANDLE(-1).value:
        return mods
    entry = MODULEENTRY32W()
    entry.dwSize = ctypes.sizeof(MODULEENTRY32W)
    ok = k32.Module32FirstW(snap, ctypes.byref(entry))
    while ok:
        mods.append({
            "name": entry.szModule,
            "path": entry.szExePath,
            "base": ctypes.cast(entry.modBaseAddr, ctypes.c_void_p).value,
            "size": entry.modBaseSize,
        })
        ok = k32.Module32NextW(snap, ctypes.byref(entry))
    k32.CloseHandle(snap)
    return mods


def thread_cpu_ms(handle):
    c = wt.FILETIME(); e = wt.FILETIME(); k = wt.FILETIME(); u = wt.FILETIME()
    if not k32.GetThreadTimes(handle, ctypes.byref(c), ctypes.byref(e),
                              ctypes.byref(k), ctypes.byref(u)):
        return None
    ticks = (k.dwHighDateTime << 32 | k.dwLowDateTime) + \
            (u.dwHighDateTime << 32 | u.dwLowDateTime)
    return ticks / 10_000.0


def open_threads(pid):
    handles = {}
    for tid in list_threads(pid):
        h = k32.OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT |
                           THREAD_QUERY_INFORMATION, False, tid)
        if h:
            handles[tid] = h
    return handles


def pick_workload_thread(handles, probe_ms=120):
    first = {tid: thread_cpu_ms(h) for tid, h in handles.items()}
    time.sleep(probe_ms / 1000.0)
    second = {tid: thread_cpu_ms(h) for tid, h in handles.items()}
    deltas = {}
    for tid in handles:
        if first[tid] is None or second[tid] is None:
            continue
        deltas[tid] = second[tid] - first[tid]
    if not deltas:
        return None, deltas
    return max(deltas, key=deltas.get), deltas


class SamplingError(Exception):
    """A sampling failure that must not leave analysable output behind."""


class ThreadGone(SamplingError):
    """Suspend/context failed; benign only when the target itself is exiting."""


def sha256_file(path, chunk_size=1 << 20):
    """Content hash of one file, streamed so large images stay cheap to hash."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    return {"path": os.path.abspath(path), "size": os.path.getsize(path),
            "sha256": sha256_file(path)}


def image_identity(exe_path):
    """Identifies the sampled EXE/PDB pair by content, not just by location.

    A rebuild normally keeps the path and a similar size, so path and size
    cannot tell the sampled binary from its replacement. These hashes are
    recorded in `samples.json` and re-checked by both offline scripts before
    any symbolization happens.
    """
    pdb_path = os.path.splitext(exe_path)[0] + ".pdb"
    missing = [p for p in (exe_path, pdb_path) if not os.path.isfile(p)]
    if missing:
        raise SamplingError("offline analysis needs the EXE and PDB together; "
                            "missing: %s" % ", ".join(missing))
    return {"exe": file_identity(exe_path), "pdb": file_identity(pdb_path)}


class Walker:
    """dbghelp walking context bound to the target process."""

    def __init__(self, hproc, modules, exe):
        self.hproc = hproc
        self.modules = modules
        self.frame_template = ctypes.create_string_buffer(STACKFRAME64_SIZE)
        dbg.SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME)
        if not dbg.SymInitializeW(hproc, None, False):
            raise RuntimeError("walk SymInitializeW failed: %d"
                               % ctypes.get_last_error())
        if not dbg.SymLoadModuleExW(hproc, None, exe["path"], None,
                                    exe["base"], exe["size"], None, 0):
            raise RuntimeError("walk SymLoadModuleExW failed: %d"
                               % ctypes.get_last_error())

    def stack(self, ctx):
        frame = ctypes.create_string_buffer(bytes(self.frame_template),
                                            STACKFRAME64_SIZE)
        rsp = int.from_bytes(bytes(ctx[OFF_RSP:OFF_RSP + 8]), "little")
        rbp = int.from_bytes(bytes(ctx[OFF_RBP:OFF_RBP + 8]), "little")
        rip = int.from_bytes(bytes(ctx[OFF_RIP:OFF_RIP + 8]), "little")

        def put(off, value, size=8):
            frame[off:off + size] = value.to_bytes(size, "little")

        # ADDRESS64 = { ULONG64 Offset; WORD Segment; ADDRESS_MODE Mode; }:
        # Offset at +0, Mode at +12, sizeof 16. AddrPC/AddrFrame/AddrStack are
        # the 1st, 3rd and 4th entries of STACKFRAME64.
        put(0, rip); put(12, ADDR_MODE_FLAT, 4)
        put(32, rbp); put(44, ADDR_MODE_FLAT, 4)
        put(48, rsp); put(60, ADDR_MODE_FLAT, 4)

        pcs = []
        for _ in range(MAX_DEPTH):
            ok = dbg.StackWalk64(
                IMAGE_FILE_MACHINE_AMD64, self.hproc, None,
                ctypes.byref(frame), ctypes.cast(ctx, ctypes.c_void_p),
                None, dbg.SymFunctionTableAccess64, dbg.SymGetModuleBase64,
                None)
            if not ok:
                break
            pcs.append(int.from_bytes(bytes(frame[0:8]), "little"))
        return pcs

    def cleanup(self):
        dbg.SymCleanup(self.hproc)


class Symbolizer:
    """dbghelp symbol context bound to the sampler process."""

    def __init__(self, modules, exe):
        self.modules = modules
        self.cache = {}
        self.proc = k32.GetCurrentProcess()
        dbg.SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME | SYMOPT_LOAD_LINES)
        if not dbg.SymInitializeW(self.proc, None, False):
            raise RuntimeError("sym SymInitializeW failed: %d"
                               % ctypes.get_last_error())
        if not dbg.SymLoadModuleExW(self.proc, None, exe["path"], None,
                                    exe["base"], exe["size"], None, 0):
            raise RuntimeError("sym SymLoadModuleExW failed: %d"
                               % ctypes.get_last_error())

    def module_of(self, addr):
        for m in self.modules:
            if m["base"] <= addr < m["base"] + m["size"]:
                return m
        return None

    def resolve(self, addr):
        hit = self.cache.get(addr)
        if hit is not None:
            return hit
        result = {"addr": addr, "symbol": None, "file": None, "line": None,
                  "disp": 0, "module": None, "rva": None}
        mod = self.module_of(addr)
        if mod is not None:
            result["module"] = mod["name"]
            result["rva"] = addr - mod["base"]
        disp = ctypes.c_ulonglong(0)
        buf = ctypes.create_string_buffer(ctypes.sizeof(SYMBOL_INFOW) + 1024)
        sym = ctypes.cast(buf, ctypes.POINTER(SYMBOL_INFOW))
        sym.contents.SizeOfStruct = ctypes.sizeof(SYMBOL_INFOW)
        sym.contents.MaxNameLen = 512
        if dbg.SymFromAddrW(self.proc, addr, ctypes.byref(disp), sym):
            result["symbol"] = ctypes.string_at(
                ctypes.addressof(buf) + SYMBOL_INFO_NAME_OFFSET,
                sym.contents.NameLen * 2).decode("utf-16-le", "replace")
            result["disp"] = int(disp.value)
            line = IMAGEHLP_LINEW64()
            line.SizeOfStruct = ctypes.sizeof(IMAGEHLP_LINEW64)
            ldisp = wt.DWORD(0)
            if dbg.SymGetLineFromAddrW64(self.proc, addr, ctypes.byref(ldisp),
                                         ctypes.byref(line)):
                result["line"] = int(line.LineNumber)
                if line.FileName:
                    result["file"] = os.path.basename(line.FileName)
        self.cache[addr] = result
        return result

    def cleanup(self):
        dbg.SymCleanup(self.proc)


def prepare_outdir(outdir):
    """Refuses to reuse a non-empty directory.

    A stale `samples.json` from an earlier run would let a failed run look
    successful to the offline scripts, so every run writes into a fresh
    directory.
    """
    if os.path.exists(outdir) and not os.path.isdir(outdir):
        raise SamplingError("output path exists and is not a directory: %s" % outdir)
    if os.path.isdir(outdir) and os.listdir(outdir):
        raise SamplingError(
            "output directory is not empty: %s\n"
            "use a fresh output directory for every run so that a stale "
            "samples.json/sampler.log cannot mask a failed run" % outdir)
    os.makedirs(outdir, exist_ok=True)


def sample_once(h, walker, hold_suspended):
    """Captures one sample; always attempts to resume a suspended thread."""
    if k32.SuspendThread(h) == 0xFFFFFFFF:
        raise ThreadGone("SuspendThread failed: %d" % ctypes.get_last_error())
    suspended = True
    try:
        ctx = ctypes.create_string_buffer(CONTEXT_SIZE)
        ctx[0x30:0x34] = CONTEXT_FULL.to_bytes(4, "little")
        if not k32.GetThreadContext(h, ctypes.cast(ctx, ctypes.c_void_p)):
            raise ThreadGone("GetThreadContext failed: %d"
                             % ctypes.get_last_error())
        if hold_suspended:
            # Optional protocol: keep the thread suspended for the walk so the
            # captured stack cannot change under the unwinder.
            return walker.stack(ctx)
        # Default: resume before unwinding. The walk only needs the captured
        # CONTEXT and target memory, so holding the thread suspended for it
        # would inflate the workload's wall time.
        if k32.ResumeThread(h) == 0xFFFFFFFF:
            raise SamplingError("ResumeThread failed: %d" % ctypes.get_last_error())
        suspended = False
        return walker.stack(ctx)
    finally:
        if suspended and k32.ResumeThread(h) == 0xFFFFFFFF:
            raise SamplingError("ResumeThread failed: %d" % ctypes.get_last_error())


def target_exiting(proc, grace_ms=None):
    """True when a suspend/context failure is really the target exiting."""
    grace_ms = EXIT_GRACE_MS if grace_ms is None else grace_ms
    deadline = time.perf_counter() + grace_ms / 1000.0
    while proc.poll() is None:
        if time.perf_counter() >= deadline:
            return False
        time.sleep(0.01)
    return True


def sample(outdir, cmd, interval_ms, startup_skip_ms, hold_suspended, log):
    """Samples the target once and returns this run's exit status.

    Status 0 requires a zero target exit code and at least one sample; 3 is a
    sampling/symbolization failure, 4 a non-zero target exit, and 5 an empty
    sample set. Failures write no `samples.json`.
    """
    proc = None
    hproc = None
    walker = None
    symbolizer = None
    handles = {}
    samples = []
    target_rc = None
    try:
        with open(os.path.join(outdir, "target.stdout.txt"), "w",
                  encoding="utf-8") as so, \
             open(os.path.join(outdir, "target.stderr.txt"), "w",
                  encoding="utf-8") as se:
            proc = subprocess.Popen(cmd, stdout=so, stderr=se)
            log.write(
                "command: %s\npid: %d\ninterval_ms: %s\nstartup_skip_ms: %s\n"
                "max_depth: %d\nhold_suspended: %d\n"
                % (" ".join(cmd), proc.pid, interval_ms, startup_skip_ms,
                   MAX_DEPTH, int(hold_suspended)))
            log.write("python: %s (%s)\ndbghelp: %s\n"
                      % (sys.version.split()[0], sys.executable, DBGHELP_PATH))
            log.flush()

            hproc = k32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
                                    False, proc.pid)
            if not hproc:
                raise SamplingError("OpenProcess failed: %d"
                                    % ctypes.get_last_error())
            time.sleep(startup_skip_ms / 1000.0)

            modules = list_modules(proc.pid)
            if not modules:
                raise SamplingError("no modules for pid %d" % proc.pid)
            exe = modules[0]
            image = image_identity(exe["path"])
            log.write("image: %s size=%d sha256=%s\n"
                      % (image["exe"]["path"], image["exe"]["size"],
                         image["exe"]["sha256"]))
            log.write("pdb: %s size=%d sha256=%s\n"
                      % (image["pdb"]["path"], image["pdb"]["size"],
                         image["pdb"]["sha256"]))
            log.flush()

            walker = Walker(hproc, modules, exe)
            symbolizer = Symbolizer(modules, exe)

            handles = open_threads(proc.pid)
            tid, deltas = pick_workload_thread(handles)
            log.write("threads: %s\ncpu_ms_delta_120ms: %s\nworkload_tid: %s\n"
                      % (sorted(handles), deltas, tid))
            if tid is None:
                raise SamplingError("no workload thread")
            h = handles[tid]
            cpu_start = thread_cpu_ms(h)

            short_walks = 0
            start = time.perf_counter()
            while proc.poll() is None:
                try:
                    pcs = sample_once(h, walker, hold_suspended)
                except ThreadGone as error:
                    if target_exiting(proc):
                        # The target exited between the poll and the suspend;
                        # that is an empty outcome, not a sampler failure.
                        log.write("sampling stopped when the target exited: %s\n"
                                  % error)
                        break
                    raise
                if not pcs:
                    continue
                if len(pcs) < 3:
                    short_walks += 1
                resolved = [symbolizer.resolve(pc) for pc in pcs]
                samples.append({"t": round(time.perf_counter() - start, 6),
                                "rip": resolved[0], "frames": resolved[1:]})
                time.sleep(interval_ms / 1000.0)
            elapsed = time.perf_counter() - start

            cpu_end = thread_cpu_ms(h)
            for t, th in handles.items():
                v = thread_cpu_ms(th)
                log.write("thread %d cpu_ms_total=%.3f\n"
                          % (t, v if v is not None else -1))
            log.write("workload_thread_cpu_ms_window=%.3f\n"
                      % ((cpu_end - cpu_start)
                         if cpu_start is not None and cpu_end is not None else -1))
            log.write("samples: %d\nsampling_wall_s: %.3f\n"
                      % (len(samples), elapsed))
            log.write("cadence_ms: %.2f (sleep is not the achieved cadence)\n"
                      % (1000.0 * elapsed / len(samples) if samples else 0.0))
            depths = (sorted(len(s["frames"]) + 1 for s in samples)
                      if samples else [0])
            log.write("chain_depth min/median/max: %d/%d/%d\n"
                      % (depths[0], depths[len(depths) // 2], depths[-1]))
            log.write("short_walks(<3 frames): %d\n" % short_walks)
            log.flush()

            target_rc = proc.wait()
            if target_rc != 0:
                print("target exited with code %d; discarded %d samples and "
                      "wrote no samples.json" % (target_rc, len(samples)),
                      file=sys.stderr)
                return 4
            if not samples:
                print("no samples captured; nothing is analysable", file=sys.stderr)
                return 5

            with open(os.path.join(outdir, "samples.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"status": "ok", "command": cmd, "pid": proc.pid,
                           "interval_ms": interval_ms,
                           "startup_skip_ms": startup_skip_ms,
                           "max_depth": MAX_DEPTH,
                           "sampling_wall_s": elapsed, "image": image,
                           "samples": samples}, f)
            with open(os.path.join(outdir, "modules.json"), "w",
                      encoding="utf-8") as f:
                json.dump(modules, f, indent=2)
            print("sampled %d in %.2fs -> %s" % (len(samples), elapsed, outdir))
            return 0
    except SamplingError as error:
        log.write("sampling_failed: %s\n" % error)
        log.flush()
        print("sampling failed: %s" % error, file=sys.stderr)
        return 3
    except Exception as error:  # unexpected API/walk failure fails the run
        log.write("sampling_error: %r\n" % (error,))
        log.flush()
        print("sampling failed: %r" % (error,), file=sys.stderr)
        return 3
    finally:
        for thread_handle in handles.values():
            k32.CloseHandle(thread_handle)
        for owned in (walker, symbolizer):
            if owned is not None:
                owned.cleanup()
        if hproc:
            k32.CloseHandle(hproc)
        if proc is not None:
            if target_rc is None:
                if proc.poll() is None:
                    # Never block on a child that may still be running or
                    # suspended: terminate it, then reap it.
                    proc.kill()
                target_rc = proc.wait()
            log.write("target_exit_code: %d\n" % target_rc)
            log.flush()


def main():
    if len(sys.argv) < 3 or "--" not in sys.argv:
        print("usage: sample_cpu_windows.py OUTDIR -- command args...",
              file=sys.stderr)
        return 2
    outdir = sys.argv[1]
    cmd = sys.argv[sys.argv.index("--") + 1:]
    if not cmd:
        print("usage: sample_cpu_windows.py OUTDIR -- command args...",
              file=sys.stderr)
        return 2
    interval_ms = float(os.environ.get("SAMPLE_INTERVAL_MS", "1.0"))
    startup_skip_ms = float(os.environ.get("STARTUP_SKIP_MS", "150"))
    hold_suspended = os.environ.get("HOLD_SUSPENDED", "0") == "1"
    try:
        prepare_outdir(outdir)
    except SamplingError as error:
        print("cannot use output directory: %s" % error, file=sys.stderr)
        return 2
    log = open(os.path.join(outdir, "sampler.log"), "w", encoding="utf-8")
    try:
        return sample(outdir, cmd, interval_ms, startup_skip_ms,
                      hold_suspended, log)
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())
