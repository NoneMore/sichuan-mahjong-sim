"""Aggregates probe.py samples into the buckets used by the study record.

Input: one or more probe.py output directories that used the same image.
The PDB path and module list come from each directory, and every directory
must record a matching EXE/PDB SHA256 pair: samples without that record are
refused rather than trusted against the files currently on disk.

Outputs, per directory and combined:
  * self-cost buckets (innermost inlined function, symbol or module),
  * inclusive share by symbol including the sampled frame itself,
  * heuristic-stage attribution from the unwound chain,
  * integrity / context checks used to justify the attributions.
"""

import json
import os
import sys
from collections import Counter

import symbolize_inline as analyze2


STAGES = [
    "impl$1::choose",
    "choose_turn",
    "choose_discard_response",
    "evaluate_peng",
    "evaluate_open_kong",
    "rank_discards",
    "ukeire",
    "max_discard_fan",
    "select_kong",
    "kong_position",
    "choose_missing_suit",
]

BUCKETS = [
    ("dp.suit_table", lambda n, m: "analysis::suit_table" in n),
    ("dp.standard_deficit", lambda n, m: "standard_deficit" in n),
    ("dp.seven_pairs_deficit", lambda n, m: "seven_pairs_deficit" in n),
    ("dp.deficit_body", lambda n, m: "DistanceEvaluator::deficit" in n),
    ("hash", lambda n, m: ("core::hash::" in n or "std::hash::" in n
                           or "hashbrown::" in n)),
    ("shared_rules_eval", lambda n, m: ("sichuan_mahjong_sim::rules::" in n
                                        or "sichuan_mahjong_sim::eval::" in n)),
    ("policy_other", lambda n, m: "sichuan_mahjong_sim::policy::heuristic::" in n),
    ("tile_hand_helpers", lambda n, m: ("sichuan_mahjong_sim::tile::" in n
                                        or "sichuan_mahjong_sim::hand::" in n
                                        or "sichuan_mahjong_sim::action::" in n)),
    ("allocator", lambda n, m: (m or "").upper().startswith("VCRUNTIME")),
    ("os_wait", lambda n, m: (m or "").lower() in ("ntdll.dll", "kernelbase.dll",
                                                   "kernel32.dll")),
    ("mem_intrinsics", lambda n, m: ("core::ptr::" in n or "core::intrinsics::" in n
                                     or "core::slice::" in n or "core::mem::" in n
                                     or "core::core_arch::" in n)),
    ("num_cmp_helpers", lambda n, m: ("core::num::" in n or "core::cmp::" in n
                                      or "core::ops::" in n)),
]


def bucket(name, module):
    for label, test in BUCKETS:
        if test(name or "", module):
            return label
    return "other"


def stage_of(chain):
    """Nearest policy stage present in the ordered chain (leaf first)."""
    for frame in chain:
        if not frame["symbol"]:
            continue
        for stage in STAGES:
            if stage in frame["symbol"]:
                return stage
    return "none"


def pct(v, n):
    return 100.0 * v / n if n else 0.0


def load_log(path):
    values = {}
    log = os.path.join(path, "sampler.log")
    if not os.path.exists(log):
        return values
    with open(log, encoding="utf-8") as f:
        for line in f:
            if ":" in line and "=" not in line:
                k, _, v = line.partition(":")
                values[k.strip()] = v.strip()
            elif "=" in line:
                k, _, v = line.partition("=")
                values[k.strip()] = v.strip()
    return values


def analyze(path, inline):
    with open(os.path.join(path, "samples.json"), encoding="utf-8") as f:
        data = json.load(f)
    with open(os.path.join(path, "modules.json"), encoding="utf-8") as f:
        modules = json.load(f)
    live = modules[0]["base"]
    exe_name = modules[0]["name"]
    members = set(m["name"] for m in modules)
    samples = data["samples"]
    n = len(samples)

    self_bucket = Counter()
    self_inline = Counter()
    self_outer = Counter()
    inclusive = Counter()
    stages = Counter()
    rules_under_policy = 0
    rules_total = 0
    deficit_callers = Counter()
    chain_full = 0
    for s in samples:
        rip = s["rip"]
        if rip["module"] == exe_name:
            name, _disp, _line, _depth = inline.innermost(
                analyze2.inline_base + (rip["addr"] - live))
            key = name or analyze2.label(rip)
        else:
            key = analyze2.label(rip)
        self_inline[key] += 1
        self_bucket[bucket(key, rip["module"])] += 1
        self_outer[analyze2.label(rip)] += 1

        chain = [rip] + s["frames"]
        names = [analyze2.label(f) for f in chain]
        if ("sichuan_mahjong_sim::runner::Runner::drive" in names
                and "main" in names):
            chain_full += 1
        for name in set(names):
            inclusive[name] += 1
        stages[stage_of(chain)] += 1
        if "sichuan_mahjong_sim::rules::" in key or "sichuan_mahjong_sim::eval::" in key:
            rules_total += 1
            if any("impl$1::choose" in name for name in names):
                rules_under_policy += 1
        if "DistanceEvaluator::deficit" in (rip["symbol"] or "") and s["frames"]:
            deficit_callers[analyze2.label(s["frames"][0])] += 1
    return {
        "samples": n,
        "self_bucket": self_bucket,
        "self_inline": self_inline,
        "self_outer": self_outer,
        "inclusive": inclusive,
        "stages": stages,
        "rules_total": rules_total,
        "rules_under_policy": rules_under_policy,
        "deficit_callers": deficit_callers,
        "full": chain_full,
        "log": load_log(path),
        "wall": data.get("sampling_wall_s"),
    }


def report(name, a, top=12):
    n = a["samples"]
    print("### %s: %d samples (%.1f%% complete chains)" % (name, n, pct(a["full"], n)))
    print("self buckets:")
    for k, v in a["self_bucket"].most_common():
        print("   %6.2f%% %5d  %s" % (pct(v, n), v, k))
    print("top innermost functions:")
    for k, v in a["self_inline"].most_common(top):
        print("   %6.2f%% %5d  %s" % (pct(v, n), v, k[:100]))
    print("stage attribution (nearest stage in chain):")
    for k, v in a["stages"].most_common():
        print("   %6.2f%% %5d  %s" % (pct(v, n), v, k))
    print("inclusive (self frame included):")
    for k, v in a["inclusive"].most_common(top):
        print("   %6.2f%% %5d  %s" % (pct(v, n), v, k[:100]))
    if a["deficit_callers"]:
        tot = a["deficit_callers"].total()
        print("deficit direct-call contexts (%.0f%% of samples have deficit self):"
              % pct(tot, n))
        for k, v in a["deficit_callers"].most_common(6):
            print("   %6.2f%% %5d  %s" % (pct(v, tot), v, k[:100]))
    print("shared rules/eval samples inside the policy path: %d/%d (%.1f%%)"
          % (a["rules_under_policy"], a["rules_total"],
             pct(a["rules_under_policy"], a["rules_total"])))
    log = a["log"]
    for key in ("chain_depth min/median/max", "short_walks(<3 frames)",
                "samples", "sampling_wall_s", "workload_thread_cpu_ms_window"):
        if key in log:
            print("   log %s: %s" % (key, log[key]))


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print("usage: aggregate_attribution.py SAMPLEDIR...", file=sys.stderr)
        return 2
    modules = analyze2.check_same_image(dirs)
    exe = modules[0]
    analyze2.inline_base = 0x140000000
    inline = analyze2.Inline(exe["path"], analyze2.inline_base, exe["size"])

    per_run = {}
    shared_suit = {}
    for d in dirs:
        a = analyze(d, inline)
        per_run[d] = a
        report(d, a)
        n = a["samples"]
        suit = a["self_inline"].get(
            "sichuan_mahjong_sim::policy::analysis::suit_table", 0)
        shared_suit[d] = (suit, n)
        print()

    print("### suit_table self share per run")
    for d, (v, n) in shared_suit.items():
        print("   %-40s %5d/%5d = %5.2f%%" % (os.path.basename(d), v, n, pct(v, n)))
    if len(dirs) > 1:
        total = {"samples": 0}
        for key in ("self_bucket", "self_inline", "self_outer", "inclusive",
                    "stages", "deficit_callers"):
            total[key] = Counter()
        total["rules_total"] = total["rules_under_policy"] = total["full"] = 0
        total["log"] = {}
        for a in per_run.values():
            total["samples"] += a["samples"]
            total["rules_total"] += a["rules_total"]
            total["rules_under_policy"] += a["rules_under_policy"]
            total["full"] += a["full"]
            for key in ("self_bucket", "self_inline", "self_outer", "inclusive",
                        "stages", "deficit_callers"):
                total[key].update(a[key])
        print()
        report("COMBINED", total, 15)
    return 0


if __name__ == "__main__":
    sys.exit(main())
