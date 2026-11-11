from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mahjong_bc.checkpoint import (
    TrainingCheckpointState,
    atomic_torch_save,
    build_payload,
    load_policy_checkpoint,
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
from mahjong_bc.errors import CheckpointError, ConfigurationError, DataError
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


def test_from_loaded_keeps_weights_after_checkpoint_path_removed(tmp_path):
    path = _write_checkpoint(tmp_path / "best.pt")
    loaded = load_policy_checkpoint(path)
    weights = {
        name: value.detach().clone()
        for name, value in loaded.model.state_dict().items()
    }
    rng = np.random.default_rng(21)
    decisions = [_decision(rng, count) for count in (3, 1, 2, 4)]
    expected = FrozenPolicy(path).select_indices(decisions)

    path.unlink()
    assert not path.exists()
    policy = FrozenPolicy.from_loaded(loaded)
    assert policy.checkpoint == path
    assert policy.role == ROLE_BEST
    assert policy.device == torch.device("cpu")
    assert policy.metadata()["has_training_state"] is True
    assert policy.metadata()["path"] == str(path)
    assert policy._model is not loaded.model
    assert all(
        not parameter.requires_grad
        for parameter in policy._model.parameters()
    )
    for name, value in policy._model.state_dict().items():
        assert torch.equal(value, weights[name])
    assert policy.select_indices(decisions) == expected


def test_from_loaded_never_rereads_checkpoint_path(tmp_path, monkeypatch):
    path = _write_checkpoint(tmp_path / "best.pt")
    loaded = load_policy_checkpoint(path)
    rng = np.random.default_rng(22)
    decisions = [_decision(rng, count) for count in (2, 3, 1)]
    expected = FrozenPolicy(path).select_indices(decisions)

    path.write_bytes(b"replaced checkpoint bytes")

    def _must_not_load(*args, **kwargs):
        raise AssertionError("from_loaded must not reload the checkpoint")

    monkeypatch.setattr(
        "mahjong_eval.inference.load_policy_checkpoint", _must_not_load
    )
    policy = FrozenPolicy.from_loaded(loaded)
    assert policy.checkpoint == path
    assert policy.select_indices(decisions) == expected


def test_from_loaded_reuses_role_validation_without_file_access(tmp_path):
    last_path = _write_checkpoint(tmp_path / "last.pt", role=ROLE_LAST)
    loaded = load_policy_checkpoint(last_path)
    last_path.unlink()
    with pytest.raises(CheckpointError, match="not 'last'"):
        FrozenPolicy.from_loaded(loaded)
    with pytest.raises(ConfigurationError):
        FrozenPolicy.from_loaded(object())  # type: ignore[arg-type]


def test_from_loaded_freezes_independent_copies_on_requested_device(tmp_path):
    path = _write_checkpoint(tmp_path / "best.pt")
    loaded = load_policy_checkpoint(path)
    first = FrozenPolicy.from_loaded(loaded, torch.device("cpu"))
    second = FrozenPolicy.from_loaded(loaded)
    assert first.device == torch.device("cpu")
    assert second.device == torch.device("cpu")
    assert first._model is not second._model
    assert first._model is not loaded.model
    for policy in (first, second):
        assert all(
            parameter.device == torch.device("cpu")
            for parameter in policy._model.parameters()
        )
        assert all(
            not parameter.requires_grad
            for parameter in policy._model.parameters()
        )
    assert any(
        parameter.requires_grad for parameter in loaded.model.parameters()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_from_loaded_moves_independent_copy_to_cuda(tmp_path):
    path = _write_checkpoint(tmp_path / "best.pt")
    loaded = load_policy_checkpoint(path)
    policy = FrozenPolicy.from_loaded(loaded, "cuda")
    assert policy.device.type == "cuda"
    assert all(
        parameter.device == policy.device
        for parameter in policy._model.parameters()
    )
    assert all(
        parameter.device.type == "cpu"
        for parameter in loaded.model.parameters()
    )
    rng = np.random.default_rng(23)
    assert policy.select_indices([_decision(rng, 2)])[0] in (0, 1)

    loaded_cuda = load_policy_checkpoint(path, device="cuda")
    cpu_policy = FrozenPolicy.from_loaded(loaded_cuda, "cpu")
    assert cpu_policy.device == torch.device("cpu")
    assert all(
        parameter.device == torch.device("cpu")
        for parameter in cpu_policy._model.parameters()
    )
    assert all(
        parameter.device.type == "cuda"
        for parameter in loaded_cuda.model.parameters()
    )
    assert cpu_policy.select_indices([_decision(rng, 2)])[0] in (0, 1)


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
