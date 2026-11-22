"""Time or profile a full BC epoch on a prepared cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from collections import Counter, defaultdict
from contextlib import ExitStack, nullcontext
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import mahjong_bc._input_pipeline as pipeline
import mahjong_bc.train as training
import torch
from mahjong_bc import (
    BatchTransferConfig,
    ModelConfig,
    Trainer,
    TrainingConfig,
    data,
    load_prepared_cache,
)
from mahjong_bc._device import resolve_device
from mahjong_bc._source import collect_training_source
from mahjong_bc.config import collect_dependency_versions
from mahjong_bc.errors import ConfigurationError, MahjongBCError
from torch.profiler import ProfilerActivity, profile, record_function, schedule

MIB = 1024**2
SPLITS = ("train", "validation")


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _checkpoint_payloads(run):
    return {
        name: torch.load(run / name, map_location="cpu", weights_only=True)
        for name in ("last.pt", "best.pt")
    }


def _assert_equal(expected, actual, path="checkpoint"):
    if type(expected) is type(actual):
        if isinstance(expected, torch.Tensor):
            if expected.dtype == actual.dtype and torch.equal(expected, actual):
                return
        elif isinstance(expected, dict):
            if expected.keys() == actual.keys():
                for key in expected:
                    _assert_equal(expected[key], actual[key], f"{path}.{key}")
                return
        elif isinstance(expected, (tuple, list)):
            if len(expected) == len(actual):
                for index, (left, right) in enumerate(zip(expected, actual)):
                    _assert_equal(left, right, f"{path}[{index}]")
                return
        elif expected == actual:
            return
    raise RuntimeError(f"{path} differs from the reference checkpoint")


def _run_epoch(inputs, output, device, model, config, transfer):
    trainer = Trainer(
        train_dataset=inputs.datasets["train"],
        validation_dataset=inputs.datasets["validation"],
        training_config=config,
        model_config=model,
        output_dir=output,
        device=device,
        dataset_provenance={"input_loading": inputs.loading_record},
        batch_transfer=transfer,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    summary = trainer.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    eligible = inputs.datasets["train"].eligible_count
    if (
        summary.completed_epochs != 1
        or summary.global_step != math.ceil(eligible / config.batch_size)
        or summary.samples_seen != eligible
    ):
        raise RuntimeError("trainer did not complete the planned epoch")
    return {
        "seconds": seconds,
        "training_decisions_per_second": summary.samples_seen / seconds,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "summary": summary.to_dict(),
    }


class _HostProbe:
    """Collect inclusive spans on the main thread and input producer."""

    def __init__(self, *, traced):
        self.traced = traced
        self.phase = "train"
        self.samples = defaultdict(list)

    def instrument(self, owner, attribute, name):
        original = getattr(owner, attribute)

        @wraps(original)
        def wrapped(*args, **kwargs):
            phase = self.phase
            if name == "validation":
                # The training producer has joined before validation starts.
                self.phase = "validation"
            start = time.perf_counter_ns()
            try:
                with record_function(f"bc.{name}") if self.traced else nullcontext():
                    return original(*args, **kwargs)
            finally:
                self.samples[f"{phase}.{name}"].append(time.perf_counter_ns() - start)
                self.phase = phase

        return patch.object(owner, attribute, wrapped)

    def summarize(self):
        return {
            name: {
                "calls": len(samples),
                "total_ms": sum(samples) / 1e6,
                "median_per_call_ms": statistics.median(samples) / 1e6,
                "p95_per_call_ms": sorted(samples)[math.ceil(0.95 * len(samples)) - 1]
                / 1e6,
            }
            for name, samples in self.samples.items()
        }


def _busy_us(events, start, end):
    intervals = sorted(
        (max(start, event["ts"]), min(end, event["ts"] + event["dur"]))
        for event in events
    )
    covered_end = start
    total = 0
    for left, right in intervals:
        if right > max(left, covered_end):
            total += right - max(left, covered_end)
            covered_end = right
    return total


def _analyze_trace(path, active_steps):
    events = [
        event
        for event in json.loads(path.read_text(encoding="utf-8"))["traceEvents"]
        if event.get("ph") == "X"
    ]
    steps = [
        event
        for event in events
        if event.get("cat") == "user_annotation"
        and event["name"].startswith("ProfilerStep#")
    ]
    if len(steps) != active_steps:
        raise RuntimeError(f"expected {active_steps} trace steps, got {len(steps)}")
    start = min(event["ts"] for event in steps)
    end = max(event["ts"] + event["dur"] for event in steps)
    window = [
        event
        for event in events
        if event["ts"] < end and event["ts"] + event["dur"] > start
    ]
    kernels = [event for event in window if event.get("cat") == "kernel"]
    copies = [event for event in window if event.get("cat") == "gpu_memcpy"]
    h2d = [event for event in copies if "HtoD" in event["name"]]
    d2h = [event for event in copies if "DtoH" in event["name"]]
    sync = Counter(
        event["name"]
        for event in window
        if event.get("cat") == "cuda_runtime" and "Synchronize" in event["name"]
    )
    kernel_us = _busy_us(kernels, start, end)
    h2d_us = _busy_us(h2d, start, end)
    return {
        "trace_steps": len(steps),
        "observed_cpu_window_ms": (end - start) / 1e3,
        "kernel_busy_ms": kernel_us / 1e3,
        "kernel_busy_fraction": kernel_us / (end - start),
        "kernel_or_copy_busy_fraction": _busy_us(kernels + copies, start, end)
        / (end - start),
        "h2d_busy_ms": h2d_us / 1e3,
        "h2d_overlap_with_kernel_ms": max(
            0, kernel_us + h2d_us - _busy_us(kernels + h2d, start, end)
        )
        / 1e3,
        "h2d_calls": len(h2d),
        "d2h_calls": len(d2h),
        "cuda_synchronization_calls": dict(sync),
    }


def _diagnose(run, output, trace):
    probe = _HostProbe(traced=trace is not None)
    hooks = (
        (data, "_gather_pruned_segments", "gather"),
        (data, "_gather_base_columns", "gather_states_targets"),
        (data, "_flat_candidate_rows", "gather_candidates"),
        (pipeline._PrefetchPipeline, "_acquire", "producer_slot_acquire"),
        (pipeline._PrefetchPipeline, "_stage", "producer_stage"),
        (pipeline._PrefetchPipeline, "_next_item", "consumer_queue_take"),
        (pipeline._PrefetchPipeline, "_commit", "submit_h2d"),
        (pipeline._PrefetchPipeline, "_deliver", "make_compute_wait"),
        (training, "_batch_logits", "forward"),
        (training, "_require_finite_valid", "finite_logits"),
        (training, "_decision_nll_validated", "nll"),
        (torch.Tensor, "backward", "backward"),
        (training._EpochLossTotals, "_flush", "loss_flush"),
        (training, "evaluate_dataset", "validation"),
        (training, "save_checkpoint", "save_checkpoint"),
        (training.Trainer, "_write_run_record", "write_run_record"),
    )
    with ExitStack() as stack:
        for owner, attribute, name in hooks:
            stack.enter_context(probe.instrument(owner, attribute, name))
        if trace is not None:
            # The trace is exported before checkpoint publication creates the directory.
            output.mkdir()
            trace_path = output / "cuda.trace.json"
            capture = stack.enter_context(
                profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=schedule(**trace, repeat=1),
                    on_trace_ready=lambda result: result.export_chrome_trace(
                        str(trace_path)
                    ),
                )
            )
            add = training._EpochLossTotals.add

            @wraps(add)
            def step(*args, **kwargs):
                add(*args, **kwargs)
                capture.step()

            stack.enter_context(patch.object(training._EpochLossTotals, "add", step))
        trial = run(output)
    trial["host_spans"] = probe.summarize()
    if trace is not None:
        trial["cuda_trace"] = _analyze_trace(trace_path, trace["active"])
    return trial


def _profile_training(args):
    destination = args.output.resolve()
    cache_root = args.dataset.resolve()
    if destination == cache_root or cache_root in destination.parents:
        raise ConfigurationError("output must be outside the prepared cache")
    device = resolve_device(args.device)
    if device.type != "cuda" and (
        args.mode == "cuda-trace" or args.batch_transfer == "cuda-prefetch"
    ):
        raise ConfigurationError("CUDA trace and prefetch require a CUDA device")
    if device.type == "cuda":
        if device.index is not None:
            torch.cuda.set_device(device)
        device = torch.device("cuda", torch.cuda.current_device())
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    model = ModelConfig()
    if args.model_config is not None:
        model = ModelConfig.from_dict(
            json.loads(args.model_config.read_text(encoding="utf-8"))
        )
    config = TrainingConfig(
        seed=args.seed,
        epochs=1,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    transfer = BatchTransferConfig(
        mode=args.batch_transfer,
        pinned_limit_bytes=args.pinned_limit_mib * MIB
        if args.pinned_limit_mib
        else None,
    )
    ram_limit_bytes = args.ram_limit_mib * MIB if args.ram_limit_mib else None
    cache = load_prepared_cache(cache_root)
    trace = None
    if args.mode == "cuda-trace":
        trace = {
            "wait": args.trace_wait,
            "warmup": args.trace_warmup,
            "active": args.trace_steps,
        }
        steps = math.ceil(
            cache.splits["train"].counts.eligible_decisions / config.batch_size
        )
        if sum(trace.values()) > steps:
            raise ConfigurationError(
                f"trace needs {sum(trace.values())} training steps, cache has {steps}"
            )
    reference = _checkpoint_payloads(args.reference_run) if args.reference_run else None
    plan = {
        "mode": args.mode,
        "source": collect_training_source(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependencies": collect_dependency_versions().to_dict(),
        "platform": platform.platform(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        "cache_id": cache.cache_id,
        "corpus_manifest_sha256": cache.corpus_manifest_sha256,
        "training_config": config.to_dict(),
        "model_config": model.to_dict(),
        "storage": args.storage,
        "ram_limit_bytes": ram_limit_bytes,
        "batch_transfer": transfer.to_dict(),
        "torch_threads": args.threads,
        "torch_interop_threads": args.interop_threads,
        "warmup_runs": args.warmup_runs,
        "repeats": args.repeats,
        "trace_schedule": trace,
        "reference_run": (
            str(args.reference_run.resolve()) if args.reference_run else None
        ),
    }
    destination.mkdir(parents=True, exist_ok=False)
    _write_json(destination / "plan.json", plan)
    trials = []
    load_start = time.perf_counter()
    with cache.open_splits(
        SPLITS, storage=args.storage, ram_limit_bytes=ram_limit_bytes
    ) as inputs:
        load_seconds = time.perf_counter() - load_start

        def run(output):
            return _run_epoch(inputs, output, device, model, config, transfer)

        for index in range(args.warmup_runs):
            run(destination / f"warmup-{index + 1}")
        for index in range(args.repeats):
            output = destination / f"trial-{index + 1}"
            trial = (
                run(output) if args.mode == "timing" else _diagnose(run, output, trace)
            )
            payloads = _checkpoint_payloads(output)
            if reference is None:
                reference = payloads
            else:
                _assert_equal(reference, payloads)
            trials.append({"run": output.name, **trial})
        loading = inputs.loading_record
    return {
        "plan": plan,
        "input_loading": loading,
        "open_splits_seconds": load_seconds,
        "median_epoch_seconds": statistics.median(trial["seconds"] for trial in trials),
        "exact_checkpoint_comparisons": (
            len(trials) if args.reference_run else len(trials) - 1
        ),
        "trials": trials,
    }


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, type=Path, help="prepared cache directory"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="new output directory"
    )
    parser.add_argument(
        "--mode", choices=("timing", "host", "cuda-trace"), default="timing"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--storage", choices=("mmap", "ram"), default="mmap")
    parser.add_argument("--ram-limit-mib", type=_positive_int)
    parser.add_argument(
        "--batch-transfer", choices=("sync", "cuda-prefetch"), default="sync"
    )
    parser.add_argument("--pinned-limit-mib", type=_positive_int)
    parser.add_argument("--model-config", type=Path, help="serialized ModelConfig JSON")
    parser.add_argument("--seed", type=_nonnegative_int, default=7)
    parser.add_argument("--batch-size", type=_positive_int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--threads", type=_positive_int, default=4)
    parser.add_argument("--interop-threads", type=_positive_int, default=2)
    parser.add_argument("--warmup-runs", type=_nonnegative_int, default=1)
    parser.add_argument("--repeats", type=_positive_int, default=1)
    parser.add_argument(
        "--reference-run", type=Path, help="compare both checkpoints with this trial"
    )
    parser.add_argument("--trace-wait", type=_nonnegative_int, default=0)
    parser.add_argument("--trace-warmup", type=_nonnegative_int, default=1)
    parser.add_argument("--trace-steps", type=_positive_int, default=8)
    args = parser.parse_args()
    try:
        result = _profile_training(args)
    except (MahjongBCError, OSError, ValueError) as error:
        parser.error(str(error))
    _write_json(args.output / "profile.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
