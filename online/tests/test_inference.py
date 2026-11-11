from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mahjong_bc.checkpoint import (
    TrainingCheckpointState,
    atomic_torch_save,
    build_payload,
    save_checkpoint,
)
from mahjong_bc.config import (
    DatasetIdentity,
    ModelConfig,
    TrainingConfig,
    collect_dependency_versions,
)
from mahjong_bc.constants import (
    CANDIDATE_WIDTH,
    CHECKPOINT_FORMAT_VERSION,
    ENCODING_VERSION,
    MODEL_FORMAT_VERSION,
    ROLE_BEST,
    ROLE_LAST,
    ROLE_POLICY,
    STATE_WIDTH,
)
from mahjong_bc.data import pad_decision_inputs
from mahjong_bc.errors import CheckpointError, DataError
from mahjong_bc.policy import build_policy, greedy_indices

from mahjong_eval.inference import FrozenPolicy


def _model_config() -> ModelConfig:
    return ModelConfig(state_hidden=(8,), scorer_hidden=(8,))


def _training_state(model) -> TrainingCheckpointState:
    train = DatasetIdentity(
        cache_id="inference-fixture",
        split="train",
        encoding_version=ENCODING_VERSION,
        preparation_format_version=1,
        source_revision="fixture-revision",
        decisions=4,
        eligible_decisions=2,
    )
    validation = DatasetIdentity(
        cache_id="inference-fixture",
        split="validation",
        encoding_version=ENCODING_VERSION,
        preparation_format_version=1,
        source_revision="fixture-revision",
        decisions=2,
        eligible_decisions=1,
    )
    return TrainingCheckpointState(
        training_config=TrainingConfig(
            seed=3, epochs=1, batch_size=1, learning_rate=0.01
        ),
        train_dataset_identity=train,
        validation_dataset_identity=validation,
        dependencies=collect_dependency_versions(),
        device_type="cpu",
        completed_epochs=1,
        global_step=4,
        samples_seen=6,
        best_validation_nll=0.25,
        best_epoch=0,
        history=[{"epoch": 0, "steps": 4, "samples": 6}],
        optimizer_state={"state": {}, "param_groups": [{"lr": 0.01}]},
        torch_rng_state=torch.get_rng_state(),
        torch_cuda_rng_states=None,
        order_rng_state=torch.Generator().manual_seed(7).get_state(),
        best_model_state={
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        },
        best_metrics={"nll": 0.25},
    )


def _write_checkpoint(path, *, role: str = ROLE_BEST):
    model = build_policy(_model_config(), seed=5)
    if role == ROLE_POLICY:
        save_checkpoint(path, model, role=role, metrics={"nll": 0.25})
    else:
        save_checkpoint(
            path,
            model,
            role=role,
            training=_training_state(model),
            metrics={"nll": 0.25},
        )
    return path


def _decision(rng: np.random.Generator, candidates: int):
    return SimpleNamespace(
        state=rng.normal(size=STATE_WIDTH).astype(np.float32),
        candidate_features=rng.normal(
            size=(candidates, CANDIDATE_WIDTH)
        ).astype(np.float32),
    )


class _RecordingPolicy:
    """Deterministic scorer that records the model boundary it observes."""

    def __init__(self, logits) -> None:
        self._logits = torch.tensor(logits, dtype=torch.float32)
        self.calls: list[dict[str, object]] = []

    def __call__(self, states, candidates, mask):
        self.calls.append(
            {
                "batch_size": int(states.shape[0]),
                "candidates": int(candidates.shape[1]),
                "states_dtype": states.dtype,
                "candidates_dtype": candidates.dtype,
                "mask_dtype": mask.dtype,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        columns = int(candidates.shape[1])
        return self._logits[: states.shape[0], :columns].clone()


class _FirstFeatureScorer:
    """Score each candidate by its first feature so row order is observable."""

    def __call__(self, states, candidates, mask):
        return candidates[:, :, 0].clone()


def test_select_indices_matches_direct_scores_and_keeps_input_order(tmp_path):
    policy = FrozenPolicy(_write_checkpoint(tmp_path / "best.pt"))
    assert policy.role == ROLE_BEST
    assert policy.encoding_version == ENCODING_VERSION
    assert policy.metrics == {"nll": 0.25}
    metadata = policy.metadata()
    assert metadata["role"] == ROLE_BEST
    assert metadata["has_training_state"] is True
    assert policy._model.training is False
    assert all(
        not parameter.requires_grad
        for parameter in policy._model.parameters()
    )

    rng = np.random.default_rng(11)
    counts = [3, 1, 2, 4, 1, 2, 3]
    decisions = [_decision(rng, count) for count in counts]
    selected = policy.select_indices(decisions)
    assert len(selected) == len(counts)
    assert all(isinstance(index, int) for index in selected)
    for decision, count, index in zip(decisions, counts, selected):
        if count == 1:
            assert index == 0
            continue
        inputs = pad_decision_inputs([decision])
        with torch.no_grad():
            logits = policy._model(
                inputs.states, inputs.candidates, inputs.mask
            )
        expected = int(greedy_indices(logits, inputs.mask)[0])
        assert index == expected


def test_larger_padding_does_not_change_shorter_decision(tmp_path):
    policy = FrozenPolicy(_write_checkpoint(tmp_path / "best.pt"))
    rng = np.random.default_rng(12)
    short = _decision(rng, 2)
    long = _decision(rng, 5)
    alone = policy.select_indices([short])
    padded = policy.select_indices([short, long])
    assert padded[0] == alone[0]


def test_forced_decisions_bypass_inference_and_keep_alignment(tmp_path):
    policy = FrozenPolicy(_write_checkpoint(tmp_path / "best.pt"))
    recording = _RecordingPolicy([[-3.0, 4.0, 1.0]])
    policy._model = recording
    rng = np.random.default_rng(13)
    decisions = [_decision(rng, 1), _decision(rng, 3), _decision(rng, 1)]
    assert policy.select_indices(decisions) == [0, 1, 0]
    assert len(recording.calls) == 1
    call = recording.calls[0]
    assert call["batch_size"] == 1
    assert call["candidates"] == 3
    assert call["states_dtype"] == torch.float32
    assert call["candidates_dtype"] == torch.float32
    assert call["mask_dtype"] == torch.bool
    assert call["grad_enabled"] is False

    recording.calls.clear()
    assert policy.select_indices(
        [_decision(rng, 1), _decision(rng, 1)]
    ) == [0, 0]
    assert recording.calls == []


def test_ties_choose_the_lowest_canonical_candidate_index(tmp_path):
    policy = FrozenPolicy(_write_checkpoint(tmp_path / "best.pt"))
    policy._model = _RecordingPolicy(
        [
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 2.0],
            [5.0, 5.0, 9.0],
        ]
    )
    rng = np.random.default_rng(14)
    decisions = [_decision(rng, 3), _decision(rng, 3), _decision(rng, 2)]
    assert policy.select_indices(decisions) == [0, 1, 0]


def test_nonfinite_valid_logits_fail_before_selection(tmp_path):
    policy = FrozenPolicy(_write_checkpoint(tmp_path / "best.pt"))
    rng = np.random.default_rng(15)
    decision = _decision(rng, 3)
    for invalid in (float("nan"), float("inf"), float("-inf")):
        policy._model = _RecordingPolicy([[0.0, invalid, 1.0]])
        with pytest.raises(DataError, match="finite"):
            policy.select_indices([decision])


def test_nonfinite_padded_logits_do_not_affect_valid_selection(tmp_path):
    policy = FrozenPolicy(_write_checkpoint(tmp_path / "best.pt"))
    policy._model = _RecordingPolicy(
        [[1.0, 0.0, 0.0], [0.0, 2.0, float("nan")]]
    )
    rng = np.random.default_rng(16)
    decisions = [_decision(rng, 3), _decision(rng, 2)]
    assert policy.select_indices(decisions) == [0, 1]


def test_candidate_order_controls_the_selected_index(tmp_path):
    policy = FrozenPolicy(_write_checkpoint(tmp_path / "best.pt"))
    policy._model = _FirstFeatureScorer()
    state = np.zeros(STATE_WIDTH, dtype=np.float32)
    features = np.zeros((2, CANDIDATE_WIDTH), dtype=np.float32)
    features[0, 0] = 1.0
    features[1, 0] = 2.0
    forward = SimpleNamespace(state=state, candidate_features=features)
    backward = SimpleNamespace(
        state=state,
        candidate_features=np.ascontiguousarray(features[::-1]),
    )
    assert policy.select_indices([forward]) == [1]
    assert policy.select_indices([backward]) == [0]


def test_empty_and_invalid_decisions_are_rejected(tmp_path):
    policy = FrozenPolicy(_write_checkpoint(tmp_path / "best.pt"))
    assert policy.select_indices([]) == []
    with pytest.raises(DataError, match="iterable"):
        policy.select_indices(42)  # type: ignore[arg-type]
    with pytest.raises(DataError, match="state and candidate_features"):
        policy.select_indices([object()])
    rng = np.random.default_rng(18)
    good = _decision(rng, 2)
    with pytest.raises(DataError, match="state must have shape"):
        policy.select_indices(
            [
                SimpleNamespace(
                    state=good.state[:-1],
                    candidate_features=good.candidate_features,
                )
            ]
        )
    with pytest.raises(DataError, match="candidates must have shape"):
        policy.select_indices(
            [
                SimpleNamespace(
                    state=good.state,
                    candidate_features=good.candidate_features[:, :-1],
                )
            ]
        )
    with pytest.raises(DataError, match="at least one candidate"):
        policy.select_indices(
            [
                SimpleNamespace(
                    state=good.state,
                    candidate_features=good.candidate_features[:0],
                )
            ]
        )


def test_checkpoint_roles_are_limited_to_frozen_policies(tmp_path):
    best = _write_checkpoint(tmp_path / "best.pt", role=ROLE_BEST)
    loaded = FrozenPolicy(best, "cpu")
    assert loaded.role == ROLE_BEST
    assert loaded.device == torch.device("cpu")
    frozen = _write_checkpoint(tmp_path / "policy.pt", role=ROLE_POLICY)
    assert FrozenPolicy(frozen).role == ROLE_POLICY
    last = _write_checkpoint(tmp_path / "last.pt", role=ROLE_LAST)
    with pytest.raises(CheckpointError, match="not 'last'"):
        FrozenPolicy(last)


def test_legacy_format2_best_checkpoint_is_readable(tmp_path):
    model = build_policy(_model_config(), seed=5)
    payload = build_payload(model, role=ROLE_POLICY, metrics={"nll": 0.25})
    payload["role"] = ROLE_BEST
    payload["checkpoint_format_version"] = 2
    payload["training"] = None
    path = tmp_path / "legacy-best.pt"
    atomic_torch_save(payload, path)

    policy = FrozenPolicy(path)
    assert policy.role == ROLE_BEST
    metadata = policy.metadata()
    assert metadata["checkpoint_format_version"] == 2
    assert metadata["has_training_state"] is False
    rng = np.random.default_rng(20)
    assert policy.select_indices([_decision(rng, 2)])[0] in (0, 1)


def test_unsupported_checkpoint_versions_are_rejected(tmp_path):
    model = build_policy(_model_config(), seed=5)
    payload = build_payload(model, role=ROLE_POLICY, metrics={"nll": 0.25})

    encoding = dict(payload)
    encoding["encoding_version"] = ENCODING_VERSION + 1
    encoding_path = tmp_path / "encoding.pt"
    atomic_torch_save(encoding, encoding_path)
    with pytest.raises(CheckpointError, match="encoding version"):
        FrozenPolicy(encoding_path)

    container = dict(payload)
    container["checkpoint_format_version"] = CHECKPOINT_FORMAT_VERSION + 1
    container_path = tmp_path / "container.pt"
    atomic_torch_save(container, container_path)
    with pytest.raises(CheckpointError, match="checkpoint format version"):
        FrozenPolicy(container_path)

    model_version = dict(payload)
    model_version["model_format_version"] = MODEL_FORMAT_VERSION + 1
    model_path = tmp_path / "model.pt"
    atomic_torch_save(model_version, model_path)
    with pytest.raises(CheckpointError, match="model format version"):
        FrozenPolicy(model_path)

    with pytest.raises(CheckpointError):
        FrozenPolicy(tmp_path / "missing.pt")


def test_inference_uses_only_the_checkpoint_not_training_artifacts(
    tmp_path, monkeypatch
):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    checkpoint = _write_checkpoint(artifacts / "best.pt")
    assert sorted(path.name for path in artifacts.iterdir()) == ["best.pt"]

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    policy = FrozenPolicy(checkpoint)
    rng = np.random.default_rng(19)
    selected = policy.select_indices([_decision(rng, 2), _decision(rng, 1)])
    assert selected[1] == 0
    assert selected[0] in (0, 1)
    assert not (elsewhere / "run.json").exists()
