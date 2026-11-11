"""Plan, raw terminal records, and seed-block summaries for online evaluation.

One run directory holds four artifacts:

``state.json``
    The run's persistent completion state. :func:`create_run` publishes it as
    ``"incomplete"`` before any other artifact. Only a clean close of the
    record stream changes it to ``"complete"``; a failed append or close leaves
    it incomplete even when every record already reached the disk.
``plan.json``
    The fixed six-placement plan and its reproducibility metadata, published
    before the first game.
``games.jsonl``
    One compact JSON object per completed game, appended and flushed as each
    game reaches a terminal result.
``summary.json``
    Published only after every persisted record validates, the complete
    summary has been recomputed from the files above, and the run state says
    the record stream closed cleanly.

The summary never trusts an in-memory game list: :func:`summarize_run` reads
``plan.json`` and ``games.jsonl`` again, and :func:`finalize_run` is the only
writer of ``summary.json``. A game, write, or record-consistency failure stops
the run while leaving every completed record in place.

The comparison is a free-for-all over the six neural-seat pairs
``[0,1] [0,2] [0,3] [1,2] [1,3] [2,3]``; the other two seats use the heuristic
policy. The primary strength observation is ``neural_mean``, the mean terminal
net score per neural seat. ``delta`` (neural mean minus heuristic mean) and both
seed-block standard errors are reported for context, not for checkpoint
selection.

This module deliberately does not import a native extension. Checkpoint
loading and Python source hashing reuse the offline ``mahjong_bc`` contract; a
Rust source/build record is accepted as an opaque JSON value so the native
layer can define its own structure.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import statistics
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: Neural-seat pairs, in the order games are played within each seed block.
PLACEMENTS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
)

SEAT_COUNT = 4
NEURAL_SEATS_PER_GAME = 2
END_REASONS: tuple[str, ...] = ("three_winners", "wall_exhausted")

PLAN_FORMAT = "mahjong-python-eval-plan"
PLAN_FORMAT_VERSION = 1
SUMMARY_FORMAT = "mahjong-python-eval-summary"
SUMMARY_FORMAT_VERSION = 1

STATE_FILENAME = "state.json"
PLAN_FILENAME = "plan.json"
RECORDS_FILENAME = "games.jsonl"
SUMMARY_FILENAME = "summary.json"

STATE_FORMAT = "mahjong-python-eval-run-state"
STATE_FORMAT_VERSION = 1
STATE_INCOMPLETE = "incomplete"
STATE_COMPLETE = "complete"

_U64_MAX = 2**64 - 1
_I32_MIN = -(2**31)
_I32_MAX = 2**31 - 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_STATE_KEYS = frozenset({"format", "format_version", "status"})
_PLAN_KEYS = frozenset(
    {
        "format",
        "format_version",
        "seeds",
        "placements",
        "neural_policy",
        "heuristic_policy",
        "decision_limit",
        "metadata",
    }
)
_RECORD_KEYS = frozenset(
    {
        "seed",
        "neural_seats",
        "scores",
        "winners",
        "end_reason",
        "decisions",
    }
)
_REQUIRED_METADATA_KEYS = frozenset(
    {
        "checkpoint",
        "python_source",
        "rust_runtime",
        "training_provenance",
        "encoding_version",
        "model_format_version",
        "split_algorithm",
        "selection",
        "teacher_cache",
        "dependencies",
    }
)
_CHECKPOINT_KEYS = frozenset(
    {
        "sha256",
        "size_bytes",
        "role",
        "package_version",
        "checkpoint_format_version",
        "model_format_version",
        "encoding_version",
        "model_config",
        "has_training_state",
        "metrics",
    }
)
_SOURCE_RECORD_KEYS = frozenset(
    {"package", "package_version", "hash", "revision", "dirty"}
)
_SELECTION_KEYS = frozenset({"mode", "device", "batch_size"})
_SOURCE_ROLES = ("online_evaluator", "offline_policy")

__all__ = [
    "END_REASONS",
    "EvaluationPlan",
    "NEURAL_SEATS_PER_GAME",
    "PLACEMENTS",
    "PLAN_FILENAME",
    "PLAN_FORMAT",
    "PLAN_FORMAT_VERSION",
    "RECORDS_FILENAME",
    "RecordError",
    "RecordWriter",
    "RunPaths",
    "SEAT_COUNT",
    "STATE_COMPLETE",
    "STATE_FILENAME",
    "STATE_FORMAT",
    "STATE_FORMAT_VERSION",
    "STATE_INCOMPLETE",
    "SUMMARY_FILENAME",
    "SUMMARY_FORMAT",
    "SUMMARY_FORMAT_VERSION",
    "TerminalRecord",
    "build_plan_metadata",
    "checkpoint_identity_record",
    "collect_python_source",
    "compute_summary",
    "create_run",
    "finalize_run",
    "load_checkpoint_with_identity",
    "read_plan",
    "read_records",
    "run_paths",
    "summarize_run",
    "training_provenance",
    "validate_record",
    "write_plan",
    "write_summary",
]


class RecordError(ValueError):
    """An invalid plan, terminal record, or inconsistent record set."""


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordError(f"{label} must be a JSON object")
    return value


def _require_int(
    value: Any,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RecordError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise RecordError(f"{label} must be at least {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise RecordError(f"{label} must be at most {maximum}, got {value}")
    return value


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RecordError(f"{label} must be a nonempty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RecordError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details = []
    if missing:
        details.append(f"missing {missing}")
    if extra:
        details.append(f"unknown {extra}")
    raise RecordError(f"{label} has invalid keys: {', '.join(details)}")


def _reject_json_constant(value: str) -> None:
    raise RecordError(f"non-finite JSON number {value!r} is not allowed")


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RecordError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _loads_json(text: str, label: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_pairs,
        )
    except json.JSONDecodeError as exc:
        raise RecordError(
            f"{label} is not valid JSON: {exc.msg} "
            f"(line {exc.lineno} column {exc.colno})"
        ) from exc


def _placement_text(placement: Sequence[int]) -> str:
    return f"[{placement[0]}, {placement[1]}]"


@dataclass(frozen=True)
class TerminalRecord:
    """One completed game's raw result.

    ``winners`` contains winning seat indices; ``decisions`` counts every
    accepted decision, including forced choices. Scores are seat-ordered and
    must sum to zero. The shape and end-reason/winner-count consistency are
    checked when the record is constructed; membership in the fixed plan and
    the decision limit are checked by :func:`validate_record`.
    """

    seed: int
    neural_seats: tuple[int, int]
    scores: tuple[int, int, int, int]
    winners: tuple[int, ...]
    end_reason: str
    decisions: int

    def __post_init__(self) -> None:
        try:
            neural = tuple(self.neural_seats)
            scores = tuple(self.scores)
            winners = tuple(self.winners)
        except TypeError as exc:
            raise RecordError(
                "record seats, scores, and winners must be sequences"
            ) from exc
        object.__setattr__(self, "neural_seats", neural)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "winners", winners)

        _require_int(self.seed, "seed", minimum=0, maximum=_U64_MAX)
        if len(neural) != NEURAL_SEATS_PER_GAME:
            raise RecordError(
                f"neural_seats must contain {NEURAL_SEATS_PER_GAME} seats"
            )
        for seat in neural:
            _require_int(seat, "neural seat", minimum=0, maximum=SEAT_COUNT - 1)
        if neural[0] == neural[1]:
            raise RecordError("neural_seats must name two distinct seats")
        if neural[0] > neural[1]:
            neural = (neural[1], neural[0])
            object.__setattr__(self, "neural_seats", neural)

        if len(scores) != SEAT_COUNT:
            raise RecordError(f"scores must contain {SEAT_COUNT} entries")
        for score in scores:
            _require_int(score, "score", minimum=_I32_MIN, maximum=_I32_MAX)
        score_sum = sum(scores)
        if score_sum != 0:
            raise RecordError(f"scores must sum to zero, got {score_sum}")

        seen_winners: set[int] = set()
        for seat in winners:
            _require_int(seat, "winner", minimum=0, maximum=SEAT_COUNT - 1)
            if seat in seen_winners:
                raise RecordError(f"winner seat {seat} is listed more than once")
            seen_winners.add(seat)
        winners = tuple(sorted(seen_winners))
        object.__setattr__(self, "winners", winners)

        _require_str(self.end_reason, "end_reason")
        if self.end_reason not in END_REASONS:
            raise RecordError(
                f"unsupported end_reason {self.end_reason!r}; "
                f"expected one of {list(END_REASONS)}"
            )
        if self.end_reason == "three_winners" and len(winners) != 3:
            raise RecordError(
                "three_winners requires exactly three winners, "
                f"got {len(winners)}"
            )
        if self.end_reason == "wall_exhausted" and len(winners) >= 3:
            raise RecordError(
                "wall_exhausted requires fewer than three winners, "
                f"got {len(winners)}"
            )
        _require_int(
            self.decisions, "decisions", minimum=1, maximum=_U64_MAX
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "neural_seats": list(self.neural_seats),
            "scores": list(self.scores),
            "winners": list(self.winners),
            "end_reason": self.end_reason,
            "decisions": self.decisions,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "TerminalRecord":
        record = _require_mapping(value, "terminal record")
        _require_exact_keys(record, _RECORD_KEYS, "terminal record")
        return cls(
            seed=record["seed"],
            neural_seats=record["neural_seats"],
            scores=record["scores"],
            winners=record["winners"],
            end_reason=record["end_reason"],
            decisions=record["decisions"],
        )


def validate_record(plan: "EvaluationPlan", record: TerminalRecord) -> None:
    """Reject a record that does not belong to the fixed plan."""

    if not isinstance(record, TerminalRecord):
        raise RecordError("record must be a TerminalRecord")
    if record.seed not in plan.seeds:
        raise RecordError(f"seed {record.seed} is not a planned game seed")
    if record.neural_seats not in plan.placements:
        raise RecordError(
            f"neural seats {_placement_text(record.neural_seats)} are not one "
            "of the six planned placements"
        )
    if record.decisions > plan.decision_limit:
        raise RecordError(
            f"decision count {record.decisions} exceeds the configured limit "
            f"{plan.decision_limit}"
        )


@dataclass(frozen=True)
class EvaluationPlan:
    """The fixed seed/placement plan and pre-run reproducibility metadata."""

    seeds: tuple[int, ...]
    metadata: Mapping[str, Any]
    decision_limit: int
    neural_policy: str = "neural-checkpoint"
    heuristic_policy: str = "heuristic"
    placements: tuple[tuple[int, int], ...] = PLACEMENTS

    def __post_init__(self) -> None:
        try:
            seeds = tuple(self.seeds)
            placements = tuple(tuple(pair) for pair in self.placements)
        except TypeError as exc:
            raise RecordError("seeds and placements must be sequences") from exc
        if not seeds:
            raise RecordError("the comparison requires at least one game seed")
        for seed in seeds:
            _require_int(seed, "seed", minimum=0, maximum=_U64_MAX)
        if len(set(seeds)) != len(seeds):
            raise RecordError("seeds must be unique")
        if placements != PLACEMENTS:
            raise RecordError(
                f"placements must be the fixed six-pair table {list(PLACEMENTS)}"
            )
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "placements", placements)

        _require_int(
            self.decision_limit,
            "decision_limit",
            minimum=1,
            maximum=_U64_MAX,
        )
        _require_str(self.neural_policy, "neural_policy")
        _require_str(self.heuristic_policy, "heuristic_policy")
        metadata = _require_mapping(self.metadata, "plan metadata")
        object.__setattr__(self, "metadata", dict(metadata))
        _validate_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": PLAN_FORMAT,
            "format_version": PLAN_FORMAT_VERSION,
            "seeds": list(self.seeds),
            "placements": [list(pair) for pair in self.placements],
            "neural_policy": self.neural_policy,
            "heuristic_policy": self.heuristic_policy,
            "decision_limit": self.decision_limit,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationPlan":
        plan = _require_mapping(value, "plan")
        _require_exact_keys(plan, _PLAN_KEYS, "plan")
        if plan["format"] != PLAN_FORMAT:
            raise RecordError(
                f"unsupported plan format {plan['format']!r}; expected {PLAN_FORMAT!r}"
            )
        format_version = _require_int(
            plan["format_version"], "plan format_version", minimum=1
        )
        if format_version != PLAN_FORMAT_VERSION:
            raise RecordError(
                f"unsupported plan format version {format_version!r}; "
                f"expected {PLAN_FORMAT_VERSION}"
            )
        return cls(
            seeds=plan["seeds"],
            metadata=plan["metadata"],
            decision_limit=plan["decision_limit"],
            neural_policy=plan["neural_policy"],
            heuristic_policy=plan["heuristic_policy"],
            placements=plan["placements"],
        )


def _revalidate_plan(plan: EvaluationPlan) -> None:
    """Re-run full plan validation before a plan is persisted.

    ``EvaluationPlan`` copies its metadata only shallowly, so a caller can
    mutate nested values after construction. Persisting such a plan would
    publish a document that :func:`read_plan` rejects, so every writer repeats
    validation on the current object before producing output.
    """

    if not isinstance(plan, EvaluationPlan):
        raise RecordError("plan must be an EvaluationPlan")
    EvaluationPlan.from_dict(plan.to_dict())


def _validate_checkpoint_identity(value: Any) -> Mapping[str, Any]:
    checkpoint = _require_mapping(value, "metadata.checkpoint")
    missing = sorted(_CHECKPOINT_KEYS - set(checkpoint))
    if missing:
        raise RecordError(f"metadata.checkpoint is missing keys: {missing}")
    _require_sha256(checkpoint["sha256"], "metadata.checkpoint.sha256")
    _require_int(
        checkpoint["size_bytes"], "metadata.checkpoint.size_bytes", minimum=1
    )
    _require_str(checkpoint["role"], "metadata.checkpoint.role")
    _require_str(
        checkpoint["package_version"], "metadata.checkpoint.package_version"
    )
    _require_int(
        checkpoint["checkpoint_format_version"],
        "metadata.checkpoint.checkpoint_format_version",
        minimum=1,
    )
    _require_int(
        checkpoint["model_format_version"],
        "metadata.checkpoint.model_format_version",
        minimum=1,
    )
    _require_int(
        checkpoint["encoding_version"],
        "metadata.checkpoint.encoding_version",
        minimum=1,
    )
    model_config = _require_mapping(
        checkpoint["model_config"], "metadata.checkpoint.model_config"
    )
    if "encoding_version" in model_config and (
        model_config["encoding_version"] != checkpoint["encoding_version"]
    ):
        raise RecordError(
            "metadata.checkpoint.model_config encoding_version differs from "
            "the checkpoint encoding_version"
        )
    if not isinstance(checkpoint["has_training_state"], bool):
        raise RecordError("metadata.checkpoint.has_training_state must be a boolean")
    metrics = checkpoint["metrics"]
    if metrics is not None:
        _require_mapping(metrics, "metadata.checkpoint.metrics")
    return checkpoint


def _validate_source_record(value: Any, label: str) -> Mapping[str, Any]:
    source = _require_mapping(value, label)
    missing = sorted(_SOURCE_RECORD_KEYS - set(source))
    if missing:
        raise RecordError(f"{label} is missing keys: {missing}")
    _require_str(source["package"], f"{label}.package")
    version = source["package_version"]
    if version is not None:
        _require_str(version, f"{label}.package_version")
    _require_sha256(source["hash"], f"{label}.hash")
    revision = source["revision"]
    if revision is not None:
        _require_str(revision, f"{label}.revision")
    dirty = source["dirty"]
    if dirty is not None and not isinstance(dirty, bool):
        raise RecordError(f"{label}.dirty must be a boolean or null")
    return source


def _validate_metadata(value: Any) -> Mapping[str, Any]:
    metadata = _require_mapping(value, "plan metadata")
    missing = sorted(_REQUIRED_METADATA_KEYS - set(metadata))
    if missing:
        raise RecordError(f"plan metadata is missing keys: {missing}")

    checkpoint = _validate_checkpoint_identity(metadata["checkpoint"])
    python_source = _require_mapping(
        metadata["python_source"], "metadata.python_source"
    )
    for role in _SOURCE_ROLES:
        if role not in python_source:
            raise RecordError(f"metadata.python_source is missing {role!r}")
        _validate_source_record(
            python_source[role], f"metadata.python_source.{role}"
        )

    encoding_version = _require_int(
        metadata["encoding_version"], "metadata.encoding_version", minimum=1
    )
    model_format_version = _require_int(
        metadata["model_format_version"],
        "metadata.model_format_version",
        minimum=1,
    )
    if encoding_version != checkpoint["encoding_version"]:
        raise RecordError(
            "metadata.encoding_version differs from metadata.checkpoint"
        )
    if model_format_version != checkpoint["model_format_version"]:
        raise RecordError(
            "metadata.model_format_version differs from metadata.checkpoint"
        )

    rust_runtime = _require_mapping(
        metadata["rust_runtime"], "metadata.rust_runtime"
    )
    if not rust_runtime:
        raise RecordError("metadata.rust_runtime must not be empty")

    _require_str(metadata["split_algorithm"], "metadata.split_algorithm")
    selection = _require_mapping(metadata["selection"], "metadata.selection")
    missing_selection = sorted(_SELECTION_KEYS - set(selection))
    if missing_selection:
        raise RecordError(
            f"metadata.selection is missing keys: {missing_selection}"
        )
    _require_str(selection["mode"], "metadata.selection.mode")
    _require_str(selection["device"], "metadata.selection.device")
    _require_int(selection["batch_size"], "metadata.selection.batch_size", minimum=1)
    teacher_cache = metadata["teacher_cache"]
    if not (
        isinstance(teacher_cache, Mapping)
        or (
            isinstance(teacher_cache, int)
            and not isinstance(teacher_cache, bool)
            and teacher_cache >= 0
        )
    ):
        raise RecordError(
            "metadata.teacher_cache must be a nonnegative integer or a JSON object"
        )
    _require_mapping(metadata["dependencies"], "metadata.dependencies")

    training_provenance = metadata["training_provenance"]
    if training_provenance is not None:
        _require_mapping(training_provenance, "metadata.training_provenance")
    return metadata


def build_plan_metadata(
    *,
    checkpoint: Mapping[str, Any],
    rust_runtime: Any,
    split_algorithm: str,
    selection: Mapping[str, Any],
    teacher_cache: Any,
    dependencies: Mapping[str, Any],
    training_provenance: Mapping[str, Any] | None = None,
    python_source: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the pre-run metadata for :class:`EvaluationPlan`.

    ``checkpoint`` is normally the identity returned by
    :func:`load_checkpoint_with_identity`. ``training_provenance`` stays a
    separate field and is never part of checkpoint/evaluation identity.
    ``rust_runtime`` is a nonempty mapping stored verbatim: the native layer
    owns its structure. ``teacher_cache`` is either ``0`` for no cache or a
    JSON object describing the configured cache.
    ``extra`` may add further top-level metadata keys but cannot replace a
    required one.
    """

    checkpoint_value = copy.deepcopy(
        dict(_require_mapping(checkpoint, "checkpoint"))
    )
    sources = copy.deepcopy(
        collect_python_source()
        if python_source is None
        else dict(_require_mapping(python_source, "python_source"))
    )
    metadata: dict[str, Any] = {
        "checkpoint": checkpoint_value,
        "python_source": sources,
        "rust_runtime": rust_runtime,
        "training_provenance": (
            None
            if training_provenance is None
            else copy.deepcopy(
                dict(
                    _require_mapping(
                        training_provenance, "training_provenance"
                    )
                )
            )
        ),
        "encoding_version": checkpoint_value.get("encoding_version"),
        "model_format_version": checkpoint_value.get("model_format_version"),
        "split_algorithm": split_algorithm,
        "selection": copy.deepcopy(dict(_require_mapping(selection, "selection"))),
        "teacher_cache": teacher_cache,
        "dependencies": copy.deepcopy(
            dict(_require_mapping(dependencies, "dependencies"))
        ),
    }
    if extra is not None:
        extra_value = dict(_require_mapping(extra, "extra metadata"))
        collisions = sorted(set(extra_value) & set(metadata))
        if collisions:
            raise RecordError(
                f"extra metadata cannot replace required keys: {collisions}"
            )
        metadata.update(extra_value)
    _validate_metadata(metadata)
    return metadata


# ---------------------------------------------------------------------------
# Checkpoint identity and executing-source records


def checkpoint_identity_record(
    loaded: Any,
    *,
    sha256: str,
    size_bytes: int,
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Build identity metadata for an already-loaded policy checkpoint.

    The caller must supply the SHA-256 and byte length of the exact checkpoint
    content used to produce ``loaded``. Keeping this explicit is what lets a
    caller avoid recording the identity of a path that was replaced after the
    model was loaded. :func:`load_checkpoint_with_identity` does this itself.
    """

    loaded_metadata = loaded.metadata()
    if not isinstance(sha256, str):
        raise RecordError("checkpoint sha256 must be a string")
    record: dict[str, Any] = {
        "sha256": _require_sha256(sha256.lower(), "checkpoint sha256"),
        "size_bytes": _require_int(size_bytes, "checkpoint size_bytes", minimum=1),
        "role": loaded.role,
        "package_version": loaded.package_version,
        "checkpoint_format_version": loaded.checkpoint_format_version,
        "model_format_version": loaded_metadata["model_format_version"],
        "encoding_version": loaded.encoding_version,
        "model_config": loaded.model_config.to_dict(),
        "has_training_state": loaded.training is not None,
        "metrics": loaded.metrics,
    }
    if path is not None:
        record["path"] = str(Path(path).resolve())
    _validate_checkpoint_identity(record)
    return record


def load_checkpoint_with_identity(
    path: Path | str,
    *,
    device: Any = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Load a policy checkpoint and identify the exact bytes that were loaded.

    The file is read once, hashed, and loaded from a temporary copy of those
    bytes. Replacing the original path afterwards cannot change the recorded
    identity. Unsupported checkpoint/model/encoding versions still fail in the
    offline loader before evaluation starts.
    """

    from mahjong_bc.checkpoint import load_policy_checkpoint

    source = Path(path).resolve()
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise RecordError(f"cannot read checkpoint {source}: {exc}") from exc
    digest = hashlib.sha256(data).hexdigest()

    descriptor, temporary = tempfile.mkstemp(
        prefix=".mahjong-eval-checkpoint-", suffix=".pt"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
        loaded = load_policy_checkpoint(temporary, device=device)
    finally:
        if descriptor != -1:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            os.unlink(temporary)
        except OSError:
            pass

    loaded = replace(loaded, path=source)
    identity = checkpoint_identity_record(
        loaded, sha256=digest, size_bytes=len(data), path=source
    )
    return loaded, identity


def training_provenance(loaded: Any) -> dict[str, Any] | None:
    """Return optional training provenance separately from evaluation identity.

    A policy-only checkpoint has no training state and returns ``None``.
    Optimizer, RNG, and embedded best-weight state are deliberately omitted:
    evaluating a frozen checkpoint never needs them.
    """

    state = loaded.training
    if state is None:
        return None
    return {
        "training_config": state.training_config.to_dict(),
        "train_dataset_identity": state.train_dataset_identity.to_dict(),
        "validation_dataset_identity": state.validation_dataset_identity.to_dict(),
        "dependencies": state.dependencies.to_dict(),
        "device_type": state.device_type,
        "completed_epochs": state.completed_epochs,
        "global_step": state.global_step,
        "samples_seen": state.samples_seen,
        "best_validation_nll": state.best_validation_nll,
        "best_epoch": state.best_epoch,
    }


def _git_output(
    directory: Path, arguments: Sequence[str], *, git_executable: str
) -> str | None:
    try:
        completed = subprocess.run(
            [git_executable, "-C", str(directory), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _git_source_identity(
    directory: Path,
    *,
    repository_package: Path,
    git_executable: str,
) -> tuple[str | None, bool | None]:
    root_text = _git_output(
        directory, ["rev-parse", "--show-toplevel"], git_executable=git_executable
    )
    if root_text is None:
        return None, None
    try:
        root = Path(root_text.strip()).resolve()
    except OSError:
        return None, None
    if (root / repository_package).resolve() != directory:
        return None, None
    revision_text = _git_output(
        root, ["rev-parse", "HEAD"], git_executable=git_executable
    )
    status_text = _git_output(
        root, ["status", "--porcelain"], git_executable=git_executable
    )
    revision = None
    if revision_text is not None:
        candidate = revision_text.strip()
        if candidate:
            revision = candidate
    dirty = None if status_text is None else bool(status_text.strip())
    return revision, dirty


def _package_source_record(
    directory: Path | str,
    *,
    package: str,
    package_version: str | None,
    repository_package: Path,
    git_executable: str,
) -> dict[str, Any]:
    from mahjong_bc._source import hash_source_directory

    resolved = Path(directory).resolve()
    if not resolved.is_dir():
        raise RecordError(f"Python package directory does not exist: {resolved}")
    try:
        digest = hash_source_directory(resolved)
    except OSError as exc:
        raise RecordError(f"cannot hash Python package {resolved}: {exc}") from exc
    revision, dirty = _git_source_identity(
        resolved,
        repository_package=repository_package,
        git_executable=git_executable,
    )
    return {
        "package": package,
        "package_version": package_version,
        "hash": digest,
        "revision": revision,
        "dirty": dirty,
    }


def _online_package_version() -> str | None:
    """Read installed metadata without importing the native-backed package."""

    try:
        from importlib import metadata

        return metadata.version("mahjong-eval")
    except Exception:
        return None


def collect_python_source(
    *,
    offline_directory: Path | str | None = None,
    online_directory: Path | str | None = None,
    offline_package_version: str | None = None,
    online_package_version: str | None = None,
    git_executable: str = "git",
) -> dict[str, Any]:
    """Identify the executing online evaluator and offline policy sources.

    Both hashes cover the Python files actually present in the imported package
    directories, not the process working directory. Repository revision/dirty
    state is attached only when the package sits at its expected repository
    path, matching the offline package's source-record behavior.

    The optional directories and versions exist for fixtures; normal runs use
    the imported package locations.
    """

    from mahjong_bc._source import collect_source_record

    if offline_directory is None:
        import mahjong_bc

        offline_directory = Path(mahjong_bc.__file__).resolve().parent
        offline = dict(
            collect_source_record(
                offline_directory, git_executable=git_executable
            )
        )
        if offline_package_version is not None:
            offline["package_version"] = _require_str(
                offline_package_version, "offline_package_version"
            )
    else:
        offline = _package_source_record(
            offline_directory,
            package="mahjong-bc",
            package_version=offline_package_version,
            repository_package=Path("python") / "mahjong_bc",
            git_executable=git_executable,
        )

    explicit_online_directory = online_directory is not None
    if online_directory is None:
        online_directory = Path(__file__).resolve().parent
    if online_package_version is None and not explicit_online_directory:
        online_package_version = _online_package_version()
    online = _package_source_record(
        online_directory,
        package="mahjong-eval",
        package_version=online_package_version,
        repository_package=Path("online") / "mahjong_eval",
        git_executable=git_executable,
    )
    return {"online_evaluator": online, "offline_policy": offline}


# ---------------------------------------------------------------------------
# Plan persistence and raw-record streaming


def _atomic_write_json(path: Path | str, value: Any) -> None:
    from mahjong_bc.checkpoint import atomic_write_json

    atomic_write_json(path, value)


def run_paths(directory: Path | str) -> "RunPaths":
    """Return the artifact paths inside one run directory."""

    root = Path(directory)
    return RunPaths(
        directory=root,
        plan=root / PLAN_FILENAME,
        records=root / RECORDS_FILENAME,
        summary=root / SUMMARY_FILENAME,
    )


@dataclass(frozen=True)
class RunPaths:
    directory: Path
    plan: Path
    records: Path
    summary: Path

    @property
    def state(self) -> Path:
        """The persistent run-state path; not an optional artifact."""

        return self.directory / STATE_FILENAME


def _state_document(status: str) -> dict[str, Any]:
    return {
        "format": STATE_FORMAT,
        "format_version": STATE_FORMAT_VERSION,
        "status": status,
    }


def _write_run_state(path: Path | str, status: str) -> None:
    try:
        _atomic_write_json(path, _state_document(status))
    except (OSError, TypeError, ValueError) as exc:
        raise RecordError(f"failed to write run state {path}: {exc}") from exc


def _read_run_state(path: Path | str) -> str:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise RecordError(f"cannot read run state {source}: {exc}") from exc
    state = _require_mapping(
        _loads_json(text, f"run state {source}"), "run state"
    )
    _require_exact_keys(state, _STATE_KEYS, "run state")
    if state["format"] != STATE_FORMAT:
        raise RecordError(
            f"unsupported run state format {state['format']!r}; "
            f"expected {STATE_FORMAT!r}"
        )
    format_version = _require_int(
        state["format_version"], "run state format_version", minimum=1
    )
    if format_version != STATE_FORMAT_VERSION:
        raise RecordError(
            f"unsupported run state format version {format_version!r}; "
            f"expected {STATE_FORMAT_VERSION}"
        )
    status = _require_str(state["status"], "run state status")
    if status not in (STATE_INCOMPLETE, STATE_COMPLETE):
        raise RecordError(f"unsupported run state status {status!r}")
    return status


def _require_complete_run_state(path: Path | str) -> None:
    source = Path(path)
    try:
        status = _read_run_state(source)
    except RecordError as exc:
        raise RecordError(
            f"run is incomplete: cannot verify run state {source}: {exc}"
        ) from exc
    if status != STATE_COMPLETE:
        raise RecordError(
            f"run is incomplete (state {status!r}); refusing to publish a summary"
        )


def write_plan(plan: EvaluationPlan, path: Path | str) -> None:
    """Atomically publish a validated plan before play starts."""

    _revalidate_plan(plan)
    document = plan.to_dict()
    try:
        _atomic_write_json(path, document)
    except (OSError, TypeError, ValueError) as exc:
        raise RecordError(f"failed to write plan {path}: {exc}") from exc


def read_plan(path: Path | str) -> EvaluationPlan:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise RecordError(f"cannot read plan {source}: {exc}") from exc
    return EvaluationPlan.from_dict(_loads_json(text, f"plan {source}"))


def _encode_record(record: TerminalRecord) -> str:
    return json.dumps(
        record.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _append_line(handle: Any, line: str) -> None:
    handle.write(line)
    handle.flush()
    os.fsync(handle.fileno())


def _open_records_file(path: Path) -> Any:
    return path.open("x", encoding="utf-8", newline="\n")


class RecordWriter:
    """Stream completed games to ``games.jsonl``.

    Use :func:`create_run` rather than constructing this class directly. The run
    state starts as ``"incomplete"`` and changes to ``"complete"`` only after a
    clean close with no failed append. Each ``append`` validates the record,
    writes one line, flushes, and fsyncs before returning. Any append or close
    failure marks the writer failed; completed lines remain on disk, the
    persisted state stays incomplete, and :meth:`finalize` and
    :func:`finalize_run` refuse to publish a summary.
    """

    def __init__(self, directory: Path, plan: EvaluationPlan, handle: Any) -> None:
        self._paths = run_paths(directory)
        self._plan = plan
        self._handle = handle
        self._seen: set[tuple[int, tuple[int, int]]] = set()
        self._failed: str | None = None
        self._closed = False
        self._state_complete = False

    @property
    def plan(self) -> EvaluationPlan:
        return self._plan

    @property
    def paths(self) -> RunPaths:
        return self._paths

    @property
    def failed(self) -> bool:
        return self._failed is not None

    def __enter__(self) -> "RecordWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        try:
            self._close(mark_complete=exc_type is None)
        except Exception:
            if exc_type is None:
                raise
        return False

    def append(self, record: TerminalRecord | Mapping[str, Any]) -> None:
        if self._failed is not None:
            raise RecordError(
                "record writer cannot continue after a failure: "
                f"{self._failed}"
            )
        if self._closed or self._handle is None:
            raise RecordError("record writer is closed")
        try:
            if isinstance(record, Mapping):
                record = TerminalRecord.from_dict(record)
            validate_record(self._plan, record)
            key = (record.seed, record.neural_seats)
            if key in self._seen:
                raise RecordError(
                    f"duplicate record for seed {record.seed} placement "
                    f"{_placement_text(record.neural_seats)}"
                )
            line = _encode_record(record) + "\n"
            _append_line(self._handle, line)
        except RecordError as exc:
            self._failed = f"{type(exc).__name__}: {exc}"
            raise
        except Exception as exc:
            self._failed = f"{type(exc).__name__}: {exc}"
            raise RecordError(
                f"failed to write record for seed {record.seed} placement "
                f"{_placement_text(record.neural_seats)}: {exc}"
            ) from exc
        self._seen.add(key)

    def close(self) -> None:
        """Close the record stream and mark the run complete when clean."""

        self._close(mark_complete=True)

    def _close(self, *, mark_complete: bool) -> None:
        if self._handle is not None:
            handle = self._handle
            self._handle = None
            try:
                handle.close()
            except Exception as exc:
                self._failed = f"{type(exc).__name__}: {exc}"
                self._closed = True
                raise RecordError(
                    f"failed to close {self._paths.records}: {exc}"
                ) from exc
        self._closed = True
        if self._failed is not None or self._state_complete:
            return
        if not mark_complete:
            self._failed = "run exited with an exception"
            return
        try:
            _write_run_state(self._paths.state, STATE_COMPLETE)
        except RecordError as exc:
            self._failed = f"{type(exc).__name__}: {exc}"
            raise RecordError(
                f"failed to mark run complete: {exc}"
            ) from exc
        self._state_complete = True

    def finalize(self) -> dict[str, Any]:
        """Recompute and publish ``summary.json`` from the persisted files."""

        failure = self._failed
        try:
            self.close()
        except RecordError as exc:
            if failure is None:
                failure = str(exc)
        if failure is not None:
            raise RecordError(
                "record writer failed; refusing to publish a summary: "
                f"{failure}"
            )
        return finalize_run(self._paths.directory)


def create_run(directory: Path | str, plan: EvaluationPlan) -> RecordWriter:
    """Create a fresh run directory, write the plan, and open the record file.

    The plan is revalidated before anything is created. The run state is
    published as ``"incomplete"`` first, then the plan, then the record file.
    Existing state/plan/record/summary artifacts are refused so a failed or
    completed run cannot be silently mixed with a new one. A later open failure
    still leaves the run state and plan.
    """

    _revalidate_plan(plan)
    paths = run_paths(directory)
    try:
        paths.directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RecordError(
            f"cannot create run directory {paths.directory}: {exc}"
        ) from exc
    existing = [
        path.name
        for path in (paths.state, paths.plan, paths.records, paths.summary)
        if path.exists()
    ]
    if existing:
        raise RecordError(
            f"run directory {paths.directory} already contains {existing}; "
            "choose a fresh directory"
        )
    _write_run_state(paths.state, STATE_INCOMPLETE)
    write_plan(plan, paths.plan)
    try:
        handle = _open_records_file(paths.records)
    except OSError as exc:
        raise RecordError(f"failed to open {paths.records}: {exc}") from exc
    return RecordWriter(paths.directory, plan, handle)


def read_records(
    path: Path | str,
    plan: EvaluationPlan,
    *,
    require_complete: bool = False,
) -> list[TerminalRecord]:
    """Read, shape-check, and plan-check every persisted terminal record."""

    source = Path(path)
    records: list[TerminalRecord] = []
    try:
        handle = source.open("r", encoding="utf-8")
    except OSError as exc:
        raise RecordError(f"cannot read records {source}: {exc}") from exc
    with handle:
        for line_number, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            try:
                record = TerminalRecord.from_dict(
                    _loads_json(text, f"records line {line_number}")
                )
            except RecordError as exc:
                raise RecordError(
                    f"{source}: line {line_number}: {exc}"
                ) from exc
            records.append(record)
    _index_records(plan, records, require_complete=require_complete)
    return records


def _index_records(
    plan: EvaluationPlan,
    records: Iterable[TerminalRecord],
    *,
    require_complete: bool,
) -> dict[int, list[TerminalRecord]]:
    expected = {
        (seed, placement) for seed in plan.seeds for placement in plan.placements
    }
    seen: dict[tuple[int, tuple[int, int]], int] = {}
    by_seed: dict[int, list[TerminalRecord]] = {seed: [] for seed in plan.seeds}
    for position, record in enumerate(records, 1):
        validate_record(plan, record)
        key = (record.seed, record.neural_seats)
        if key in seen:
            raise RecordError(
                f"duplicate record for seed {record.seed} placement "
                f"{_placement_text(record.neural_seats)} "
                f"(positions {seen[key]} and {position})"
            )
        seen[key] = position
        by_seed[record.seed].append(record)

    if require_complete:
        missing = [key for key in sorted(expected) if key not in seen]
        if missing:
            first = missing[0]
            raise RecordError(
                f"record set is incomplete: {len(missing)} of "
                f"{len(expected)} planned games are missing "
                f"(first missing: seed {first[0]} placement "
                f"{_placement_text(first[1])})"
            )
    return by_seed


def _finish_counts(records: Sequence[TerminalRecord]) -> dict[str, int]:
    counts = {reason: 0 for reason in END_REASONS}
    for record in records:
        counts[record.end_reason] += 1
    return counts


def _decision_counts(records: Sequence[TerminalRecord]) -> dict[str, Any]:
    values = [record.decisions for record in records]
    return {
        "total": sum(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def _sample_sd(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values)


@dataclass(frozen=True)
class _BlockScores:
    """Per-seat and per-game scores in deterministic record order."""

    neural_scores: tuple[int, ...]
    heuristic_scores: tuple[int, ...]
    neural_game_means: tuple[float, ...]
    game_deltas: tuple[float, ...]


def _score_block(records: Sequence[TerminalRecord]) -> _BlockScores:
    """Compute the score values shared by block and overall summaries.

    Keep this comprehension order stable: the published summary depends on the
    resulting floating-point aggregation order.
    """

    return _BlockScores(
        neural_scores=tuple(
            record.scores[seat]
            for record in records
            for seat in record.neural_seats
        ),
        heuristic_scores=tuple(
            record.scores[seat]
            for record in records
            for seat in range(SEAT_COUNT)
            if seat not in record.neural_seats
        ),
        neural_game_means=tuple(
            statistics.fmean(record.scores[seat] for seat in record.neural_seats)
            for record in records
        ),
        game_deltas=tuple(
            statistics.fmean(record.scores[seat] for seat in record.neural_seats)
            - statistics.fmean(
                record.scores[seat]
                for seat in range(SEAT_COUNT)
                if seat not in record.neural_seats
            )
            for record in records
        ),
    )


def _seed_block_summary(
    seed: int,
    records: Sequence[TerminalRecord],
    scores: _BlockScores,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "games": len(records),
        "neural_mean": statistics.fmean(scores.neural_scores),
        "delta": statistics.fmean(scores.game_deltas),
        "spread": {
            "neural_seat_score_sd": _sample_sd(scores.neural_scores),
            "neural_game_mean_sd": _sample_sd(scores.neural_game_means),
        },
        "finish_reasons": _finish_counts(records),
        "decision_counts": _decision_counts(records),
    }


def compute_summary(
    plan: EvaluationPlan, records: Sequence[TerminalRecord]
) -> dict[str, Any]:
    """Recompute the complete seed-block summary from validated records.

    ``neural_mean_se`` and ``delta_se`` use the sample standard deviation of the
    per-seed means divided by ``sqrt(seed count)`` and are null with fewer than
    two seed blocks. They describe variation among the fixed planned seed
    blocks, not population-level precision.
    """

    if not isinstance(plan, EvaluationPlan):
        raise RecordError("plan must be an EvaluationPlan")
    records = list(records)
    by_seed = _index_records(plan, records, require_complete=True)

    neural_scores: list[int] = []
    heuristic_scores: list[int] = []
    game_neural_means: list[float] = []
    game_deltas: list[float] = []
    seed_neural_means: list[float] = []
    seed_deltas: list[float] = []
    seed_blocks: list[dict[str, Any]] = []

    for seed in plan.seeds:
        block = by_seed[seed]
        block_scores = _score_block(block)
        neural_scores.extend(block_scores.neural_scores)
        heuristic_scores.extend(block_scores.heuristic_scores)
        game_neural_means.extend(block_scores.neural_game_means)
        game_deltas.extend(block_scores.game_deltas)
        seed_neural_means.append(statistics.fmean(block_scores.neural_scores))
        seed_deltas.append(statistics.fmean(block_scores.game_deltas))
        seed_blocks.append(_seed_block_summary(seed, block, block_scores))

    neural_mean = statistics.fmean(neural_scores)
    heuristic_mean = statistics.fmean(heuristic_scores)
    delta = neural_mean - heuristic_mean
    seed_count = len(plan.seeds)
    neural_sd = _sample_sd(seed_neural_means)
    delta_sd = _sample_sd(seed_deltas)

    return {
        "format": SUMMARY_FORMAT,
        "format_version": SUMMARY_FORMAT_VERSION,
        "primary_metric": "neural_mean",
        "seeds": seed_count,
        "games": len(records),
        "placements_per_seed": len(plan.placements),
        "neural_seats_per_game": NEURAL_SEATS_PER_GAME,
        "neural_mean": neural_mean,
        "heuristic_mean": heuristic_mean,
        "delta": delta,
        "neural_mean_se": (
            None if neural_sd is None else neural_sd / seed_count**0.5
        ),
        "delta_se": None if delta_sd is None else delta_sd / seed_count**0.5,
        "se_method": (
            "sample standard deviation of per-seed means divided by "
            "sqrt(seed blocks); null with fewer than two seed blocks"
        ),
        "spread": {
            "neural_seat_score_sd": _sample_sd(neural_scores),
            "neural_game_mean_sd": _sample_sd(game_neural_means),
        },
        "finish_reasons": _finish_counts(
            [record for seed in plan.seeds for record in by_seed[seed]]
        ),
        "decision_counts": _decision_counts(
            [record for seed in plan.seeds for record in by_seed[seed]]
        ),
        "seed_blocks": seed_blocks,
    }


def summarize_run(directory: Path | str) -> dict[str, Any]:
    """Read the persisted plan and records and recompute the summary.

    This is an independent, read-only recomputation. It does not publish
    ``summary.json`` and does not consult the run's completion state;
    :func:`finalize_run` is the publishing gate that also requires a cleanly
    closed record stream.
    """

    paths = run_paths(directory)
    plan = read_plan(paths.plan)
    records = read_records(paths.records, plan, require_complete=True)
    return compute_summary(plan, records)


def write_summary(summary: Mapping[str, Any], path: Path | str) -> None:
    """Atomically publish a complete summary computed from persisted records."""

    value = _require_mapping(summary, "summary")
    try:
        _atomic_write_json(path, dict(value))
    except (OSError, TypeError, ValueError) as exc:
        raise RecordError(f"failed to write summary {path}: {exc}") from exc


def finalize_run(directory: Path | str) -> dict[str, Any]:
    """Validate the run state and records, then publish ``summary.json``.

    A missing, malformed, or incomplete run state means an append or close
    failed (or the writer never closed cleanly), so the public finalizer refuses
    even when every raw record is already on disk. Missing, duplicate,
    malformed, or inconsistent records also raise before the summary path is
    touched. Completed records remain available for diagnosis.
    """

    paths = run_paths(directory)
    _require_complete_run_state(paths.state)
    summary = summarize_run(directory)
    write_summary(summary, paths.summary)
    return summary
