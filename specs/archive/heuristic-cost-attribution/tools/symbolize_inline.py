"""Symbolizes probe.py samples, including inlined-frame attribution.

dbghelp `SymFromAddr` reports the outermost symbol, which folds the fully
inlined exact DP (`suit_table`, `seven_pairs_deficit`) into
`DistanceEvaluator::deficit`. This script uses the PDB inline information
(`SymQueryInlineTrace` + `SymFromInlineContextW`) to report the innermost
inlined function (and its line) for the sampled instruction, so the DP loops,
hash-table probing and memo work can be separated by self cost.
"""

import ctypes
import ctypes.wintypes as wt
import json
import os
import sys
from collections import Counter

import sample_cpu_windows as P

MACHINE = 0x8664
IMAGE_FILE_MACHINE_AMD64 = 0x8664


def setup(image_path, base, size):
    for name, args in {
        "SymAddrIncludeInlineTrace": [wt.HANDLE, ctypes.c_ulonglong],
        "SymQueryInlineTrace": [
            wt.HANDLE, ctypes.c_ulonglong, wt.DWORD, ctypes.c_ulonglong,
            ctypes.c_ulonglong, ctypes.POINTER(wt.DWORD), ctypes.POINTER(wt.DWORD)],
        "SymFromInlineContextW": [
            wt.HANDLE, ctypes.c_ulonglong, wt.ULONG,
            ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(P.SYMBOL_INFOW)],
        "SymGetLineFromInlineContextW": [
            wt.HANDLE, ctypes.c_ulonglong, wt.ULONG, ctypes.c_ulonglong,
            ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(P.IMAGEHLP_LINEW64)],
    }.items():
        fn = getattr(P.dbg, name)
        fn.argtypes = args
    P.dbg.SymAddrIncludeInlineTrace.restype = wt.DWORD
    proc = P.k32.GetCurrentProcess()
    P.dbg.SymSetOptions(0x4 | 0x2 | 0x10)
    if not P.dbg.SymInitializeW(proc, None, False):
        raise RuntimeError("SymInitializeW failed")
    if not P.dbg.SymLoadModuleExW(proc, None, image_path, None, base, size, None, 0):
        raise RuntimeError("SymLoadModuleExW failed")
    return proc


def sym_name(proc, addr, inline_context=None):
    disp = ctypes.c_ulonglong(0)
    buf = ctypes.create_string_buffer(ctypes.sizeof(P.SYMBOL_INFOW) + 1024)
    sym = ctypes.cast(buf, ctypes.POINTER(P.SYMBOL_INFOW))
    sym.contents.SizeOfStruct = ctypes.sizeof(P.SYMBOL_INFOW)
    sym.contents.MaxNameLen = 512
    if inline_context is None:
        ok = P.dbg.SymFromAddrW(proc, addr, ctypes.byref(disp), sym)
    else:
        ok = P.dbg.SymFromInlineContextW(proc, addr, inline_context,
                                         ctypes.byref(disp), sym)
    if not ok:
        return None, 0, 0
    return (ctypes.string_at(ctypes.addressof(buf) + 84, sym.contents.NameLen * 2)
            .decode("utf-16-le", "replace"), int(disp.value), int(sym.contents.Address))


def line_of(proc, addr, inline_context=None):
    line = P.IMAGEHLP_LINEW64()
    line.SizeOfStruct = ctypes.sizeof(P.IMAGEHLP_LINEW64)
    disp = wt.DWORD(0)
    if inline_context is None:
        ok = P.dbg.SymGetLineFromAddrW64(proc, addr, ctypes.byref(disp),
                                         ctypes.byref(line))
    else:
        # Empirically the PDB inline line records respond to context 0 with
        # machine type 0 for the innermost inline frame of this build.
        ok = False
        for ctx in (inline_context, 0):
            for machine in (0, MACHINE):
                line = P.IMAGEHLP_LINEW64()
                line.SizeOfStruct = ctypes.sizeof(P.IMAGEHLP_LINEW64)
                d64 = ctypes.c_ulonglong(0)
                if P.dbg.SymGetLineFromInlineContextW(
                        proc, addr, ctx, machine, ctypes.byref(d64),
                        ctypes.byref(line)):
                    ok = True
                    break
            if ok:
                break
    if not ok:
        return None, None
    return (os.path.basename(line.FileName) if line.FileName else None,
            int(line.LineNumber))


class Inline:
    def __init__(self, image_path, base, size):
        self.proc = setup(image_path, base, size)
        self.cache = {}

    def innermost(self, addr):
        """(innermost inline function, dispacement, line, inline depth)."""
        hit = self.cache.get(addr)
        if hit is not None:
            return hit
        outer, odisp, _ = sym_name(self.proc, addr)
        if outer is None:
            result = (None, 0, (None, None), 0)
            self.cache[addr] = result
            return result
        frames = P.dbg.SymAddrIncludeInlineTrace(self.proc, addr)
        if not frames:
            result = (outer, odisp, line_of(self.proc, addr), 0)
            self.cache[addr] = result
            return result
        ctx = wt.DWORD(0)
        idx = wt.DWORD(0)
        _, _, fstart = sym_name(self.proc, addr)
        ok = P.dbg.SymQueryInlineTrace(self.proc, fstart, 0, fstart, addr,
                                       ctypes.byref(ctx), ctypes.byref(idx))
        if not ok:
            result = (outer, odisp, line_of(self.proc, addr), 0)
            self.cache[addr] = result
            return result
        name, idisp, _ = sym_name(self.proc, addr, ctx.value)
        if name is None:
            name = outer
        result = (name or outer, idisp, line_of(self.proc, addr, ctx.value),
                  int(frames))
        self.cache[addr] = result
        return result


def label(frame):
    if frame is None:
        return "<none>"
    if frame["symbol"]:
        return frame["symbol"]
    if frame["module"]:
        return "<%s+0x%x>" % (frame["module"], frame["rva"] or 0)
    return "<no-module>"


def short(name, limit=90):
    return name if len(name) <= limit else name[:limit] + "..."


def pct(v, n):
    return 100.0 * v / n if n else 0.0


def analyze(path, inline):
    with open(os.path.join(path, "samples.json"), encoding="utf-8") as f:
        data = json.load(f)
    with open(os.path.join(path, "modules.json"), encoding="utf-8") as f:
        modules = json.load(f)
    live = modules[0]["base"]
    pseudo = inline_base
    samples = data["samples"]
    n = len(samples)

    self_outer = Counter()
    self_inline = Counter()
    self_module = Counter()
    lines_inline = Counter()
    inclusive = Counter()
    callers = Counter()
    chain_full = 0
    chain_depth = []
    for s in samples:
        rip = s["rip"]
        self_outer[label(rip)] += 1
        self_module[rip["module"] or "<none>"] += 1
        if rip["module"] == modules[0]["name"]:
            name, disp, (file, line), depth = inline.innermost(
                pseudo + (rip["addr"] - live))
            key = name or label(rip)
            self_inline[key] += 1
            if line:
                lines_inline[(short(key, 60), file, line)] += 1
        else:
            self_inline[label(rip)] += 1
        names = [label(f) for f in s["frames"]]
        chain_depth.append(len(names) + 1)
        if "sichuan_mahjong_sim::runner::Runner::drive" in names and "main" in names:
            chain_full += 1
        for name in set(names):
            inclusive[name] += 1
        if rip["symbol"] and "DistanceEvaluator::deficit" in rip["symbol"] and names:
            callers[names[0]] += 1
    return {
        "samples": n,
        "self_outer": self_outer,
        "self_inline": self_inline,
        "self_module": self_module,
        "lines": lines_inline,
        "inclusive": inclusive,
        "callers": callers,
        "full": chain_full,
        "depth": sorted(chain_depth),
        "wall": data.get("sampling_wall_s"),
    }


def report(name, a, top=18):
    n = a["samples"]
    d = a["depth"]
    print("== %s: %d samples, complete chains %.1f%%, chain depth median %d"
          % (name, n, pct(a["full"], n), d[len(d) // 2] if d else 0))
    print("-- self by innermost inlined function")
    for k, v in a["self_inline"].most_common(top):
        print("   %6.2f%%  %5d  %s" % (pct(v, n), v, short(k)))
    print("-- self by outermost symbol")
    for k, v in a["self_outer"].most_common(8):
        print("   %6.2f%%  %5d  %s" % (pct(v, n), v, short(k)))
    print("-- self by module")
    for k, v in a["self_module"].most_common():
        print("   %6.2f%%  %5d  %s" % (pct(v, n), v, k))
    print("-- inclusive by symbol")
    for k, v in a["inclusive"].most_common(top):
        print("   %6.2f%%  %5d  %s" % (pct(v, n), v, short(k)))
    print("-- immediate caller of deficit (clean-chain samples only)")
    tot = a["callers"].total()
    for k, v in a["callers"].most_common(8):
        print("   %6.2f%%  %5d  %s" % (pct(v, tot), v, short(k)))
    print("-- self by inlined function and line (top)")
    for (fn, file, line), v in a["lines"].most_common(top):
        print("   %6.2f%%  %5d  %s %s:%s" % (pct(v, n), v, fn, file, line))


def load_image(path):
    """Loads one directory's module list and validates its image/PDB pair."""
    with open(os.path.join(path, "modules.json"), encoding="utf-8") as f:
        modules = json.load(f)
    if not modules:
        raise SystemExit("%s: modules.json is empty" % path)
    exe = modules[0]
    pdb = os.path.splitext(exe["path"])[0] + ".pdb"
    if not os.path.exists(exe["path"]):
        raise SystemExit("%s: analysed image is missing: %s" % (path, exe["path"]))
    if not os.path.exists(pdb):
        raise SystemExit("%s: PDB is missing next to the image: %s" % (path, pdb))
    identity = (exe["name"], exe["path"], exe["size"], os.path.getsize(pdb))
    return modules, identity


def check_same_image(dirs):
    """Refuses to aggregate directories sampled against different images/PDBs."""
    reference = None
    modules = None
    for d in dirs:
        mods, identity = load_image(d)
        if reference is None:
            reference, modules = identity, mods
        elif identity != reference:
            raise SystemExit(
                "refusing to mix images: %s used %s (image %d bytes, PDB %d bytes) "
                "but %s used %s (image %d bytes, PDB %d bytes)"
                % (dirs[0], reference[1], reference[2], reference[3],
                   d, identity[1], identity[2], identity[3]))
    return modules


def main():
    global inline_base
    dirs = sys.argv[1:]
    if not dirs:
        print("usage: symbolize_inline.py SAMPLEDIR...", file=sys.stderr)
        return 2
    modules = check_same_image(dirs)
    exe = modules[0]
    inline_base = 0x140000000
    inline = Inline(exe["path"], inline_base, exe["size"])
    total = None
    for d in dirs:
        a = analyze(d, inline)
        report(d, a)
        print()
        if total is None:
            total = a
        else:
            total["samples"] += a["samples"]
            total["full"] += a["full"]
            total["depth"] += a["depth"]
            for key in ("self_outer", "self_inline", "self_module", "lines",
                        "inclusive", "callers"):
                total[key].update(a[key])
    if len(dirs) > 1:
        print()
        report("COMBINED", total, 24)
    return 0


if __name__ == "__main__":
    sys.exit(main())
