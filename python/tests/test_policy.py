from __future__ import annotations

import math

import pytest
import torch
from torch import nn

import mahjong_bc.policy as policy_module
from mahjong_bc.config import ACTIVATION_NAMES, ModelConfig
from mahjong_bc.errors import ConfigurationError, DataError
from mahjong_bc.policy import (
    CandidatePolicy,
    _decision_nll_validated,
    _greedy_indices_validated,
    _masked_cross_entropy_validated,
    _require_finite_valid,
    _suit_rank_features,
    build_policy,
    decision_nll,
    greedy_indices,
    masked_cross_entropy,
    masked_log_probs,
    sample_indices,
)

from .helpers import count_cuda_syncs, count_host_reads, host_read_count


def small_policy(seed: int = 1) -> CandidatePolicy:
    return build_policy(
        ModelConfig(state_hidden=(16, 8), scorer_hidden=(12,)),
        seed=seed,
    )


def inputs(batch: int = 4, candidates: int = 3):
    generator = torch.Generator().manual_seed(7)
    states = torch.randn(batch, 1013, generator=generator)
    features = torch.randn(batch, candidates, 39, generator=generator)
    mask = torch.ones(batch, candidates, dtype=torch.bool)
    if batch >= 2 and candidates >= 3:
        mask[1, 2:] = False
    if batch >= 4:
        mask[3, 1:] = False
    targets = torch.arange(batch) % candidates
    if batch >= 4:
        targets[3] = 0
    return states, features, mask, targets


def test_build_policy_rejects_invalid_seed():
    with pytest.raises(ConfigurationError):
        build_policy(seed=-1)
    with pytest.raises(ConfigurationError):
        build_policy(seed=2**64)


def test_model_emits_float32_logits_without_dropout_or_normalization():
    model = small_policy()
    states, features, mask, _ = inputs()
    logits = model(states, features, mask)
    assert logits.shape == (4, 3)
    assert logits.dtype == torch.float32
    assert model.training is True

    forbidden = (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)
    assert not any(isinstance(module, forbidden) for module in model.modules())

    names = dict(model.named_modules())
    assert names["candidate_scorer"][-1].__class__ is nn.Linear
    assert names["state_encoder"][-1].__class__ is nn.ReLU
    assert isinstance(model.candidate_scorer[-2], nn.ReLU)


def test_suit_rank_features_preserve_v1_layout():
    state = torch.zeros(1, 1013)
    state[0, 9] = 0.75  # Own hand, first Tong.
    player = 27 + 155  # Next clockwise.
    state[0, player + 2] = 1  # Missing Tong.
    state[0, player + 5] = -0.5
    meld = player + 7
    state[0, meld] = 1
    state[0, meld + 1 + 20] = 1  # Tiao 3.
    state[0, meld + 28 + 1] = 1  # Open kong.
    state[0, meld + 32 + 3] = 1  # Source is previous clockwise.
    state[0, 647 + (2 * 3 + 1) * 27 + 8] = 0.25
    state[0, 971] = 0.5  # Wall.
    state[0, 980] = 1  # Response present.
    state[0, 981 + 17] = 1  # Tong 9.
    state[0, 1008 + 2] = 1  # Response source is opposite.
    planes, global_features = _suit_rank_features(state)

    expected = torch.zeros(1, 54, 3, 9)
    expected[0, 0, 1, 0] = 0.75
    expected[0, 8, 0, 8] = 0.25
    expected[0, 23, 2, 2] = 1
    expected[0, 29, 2, 2] = 1
    expected[0, 50, 1] = 1
    expected[0, 53, 1, 8] = 1
    torch.testing.assert_close(planes, expected)
    expected_global = torch.zeros(1, 31)
    expected_global[0, 8] = -0.5
    expected_global[0, 16] = 0.5
    expected_global[0, 25] = 1
    expected_global[0, 28] = 1
    torch.testing.assert_close(global_features, expected_global)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is not available"
            ),
        ),
    ],
)
def test_suit_rank_encoder_shares_suit_weights_and_scores_packed_candidates(device):
    model = build_policy(ModelConfig(architecture="suit-rank-resnet"), seed=1).to(device)
    states, features, mask, _ = (value.to(device) for value in inputs())
    planes, _ = _suit_rank_features(states)
    order = [2, 0, 1]
    torch.testing.assert_close(
        model.state_encoder.ranks(planes[:, :, order]),
        model.state_encoder.ranks(planes)[:, :, order],
    )
    positions = mask.flatten().nonzero().flatten()
    packed = model(
        states, features[mask], mask, packed=True, candidate_positions=positions
    )
    torch.testing.assert_close(packed[mask], model(states, features, mask)[mask])
    packed[mask].sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def _padded_batch_inputs():
    padded_mask = torch.tensor(
        [
            [False, False, True, True],
            [False, False, False, False],
            [False, True, True, True],
            [False, False, False, True],
            [False, False, True, True],
            [False, False, False, False],
        ]
    )
    generator = torch.Generator().manual_seed(31)
    logits = torch.randn(6, 4, generator=generator).masked_fill(padded_mask, 1.0e9)
    mask = ~padded_mask
    targets = torch.tensor([1, 3, 0, 2, 0, 1])
    return logits, mask, targets


def test_internal_validated_primitives_match_public_loss_selection_and_gradients():
    logits, mask, targets = _padded_batch_inputs()
    eligible = mask.sum(dim=1) > 1
    assert torch.equal(
        decision_nll(logits, mask, targets),
        _decision_nll_validated(logits, mask, targets),
    )
    assert torch.equal(
        greedy_indices(logits, mask), _greedy_indices_validated(logits, mask)
    )

    reference = logits.clone().requires_grad_(True)
    trusted = logits.clone().requires_grad_(True)
    public_loss = masked_cross_entropy(reference, mask, targets, eligible=eligible)
    trusted_loss = _masked_cross_entropy_validated(trusted, mask, targets, eligible)
    assert torch.equal(public_loss, trusted_loss)
    public_loss.backward()
    trusted_loss.backward()
    assert torch.equal(reference.grad, trusted.grad)
    assert int(torch.count_nonzero(reference.grad[~mask])) == 0


def test_internal_validated_primitives_skip_public_validation_round_trips():
    logits, mask, targets = _padded_batch_inputs()
    eligible = mask.sum(dim=1) > 1
    trusted = count_host_reads(
        lambda: (
            _decision_nll_validated(logits, mask, targets),
            _greedy_indices_validated(logits, mask),
            _masked_cross_entropy_validated(logits, mask, targets, eligible),
        )
    )
    assert host_read_count(trusted) == 0
    public = count_host_reads(
        lambda: masked_cross_entropy(logits, mask, targets, eligible=eligible)
    )
    # The public entry point still pays for its own value validation.
    assert public["__bool__"] > 0


def test_public_entry_points_still_reject_what_the_trusted_path_skips():
    logits = torch.zeros(2, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    targets = torch.tensor([0, 1])
    nonfinite = logits.clone()
    nonfinite[0, 0] = float("nan")
    for entry in (
        lambda: decision_nll(nonfinite, mask, targets),
        lambda: masked_cross_entropy(nonfinite, mask, targets),
        lambda: masked_log_probs(nonfinite, mask),
        lambda: greedy_indices(nonfinite, mask),
        lambda: sample_indices(nonfinite, mask),
    ):
        with pytest.raises(DataError, match="finite"):
            entry()
    with pytest.raises(DataError, match="float32"):
        decision_nll(logits.double(), mask, targets)
    with pytest.raises(DataError, match="boolean"):
        decision_nll(logits, mask.to(torch.int64), targets)
    with pytest.raises(DataError, match="does not reference a valid candidate"):
        decision_nll(
            logits,
            torch.tensor([[True, False, True], [True, True, True]]),
            torch.tensor([1, 0]),
        )
    with pytest.raises(DataError, match="at least one eligible"):
        masked_cross_entropy(
            logits, mask, targets, eligible=torch.zeros(2, dtype=torch.bool)
        )


def test_finite_check_ignores_nonfinite_padding():
    logits = torch.tensor([[1.0, float("nan")], [float("inf"), 2.0]])
    mask = torch.tensor([[True, False], [False, True]])
    # Padding values are irrelevant; only valid logits must be finite.
    assert _require_finite_valid(logits, mask) is None
    with pytest.raises(DataError, match="finite"):
        _require_finite_valid(
            torch.tensor([[float("nan")]]), torch.ones(1, 1, dtype=torch.bool)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_finite_check_needs_only_its_validation_sync():
    logits = torch.tensor([[1.0, float("nan")]], device="cuda")
    mask = torch.tensor([[True, False]], device="cuda")
    _require_finite_valid(logits, mask)
    assert count_cuda_syncs(lambda: _require_finite_valid(logits, mask)) <= 1


def test_logits_are_computed_once_per_state_representation():
    model = small_policy()
    seen: list[tuple[int, ...]] = []

    def hook(_module, _inputs, output):
        seen.append(tuple(output.shape))

    model.state_encoder.register_forward_hook(hook)
    states, features, mask, _ = inputs(batch=5, candidates=4)
    model(states, features, mask)
    assert seen == [(5, 8)]


def test_padding_does_not_change_valid_logits_loss_or_selection():
    model = small_policy().eval()
    states, features, mask, targets = inputs(batch=4, candidates=4)
    # Make padding deliberately hostile; masked candidate rows are replaced.
    features[1, 2] = float("nan")
    features[3, 1:] = 1.0e30

    logits = model(states, features, mask)
    loss = masked_cross_entropy(logits, mask, targets, eligible=mask.sum(1) > 1)
    selected = greedy_indices(logits, mask)

    for row in range(states.shape[0]):
        valid = int(mask[row].sum())
        row_logits = model(
            states[row : row + 1],
            features[row : row + 1, :valid],
            torch.ones(1, valid, dtype=torch.bool),
        )
        assert torch.allclose(
            logits[row, :valid], row_logits[0], atol=1e-6, rtol=0
        )
        row_loss = decision_nll(
            row_logits,
            torch.ones(1, valid, dtype=torch.bool),
            targets[row : row + 1],
        )
        assert torch.allclose(
            decision_nll(logits[row : row + 1], mask[row : row + 1], targets[row : row + 1]),
            row_loss,
            atol=1e-6,
            rtol=0,
        )
        assert int(selected[row]) < valid
    expected = torch.stack(
        [
            decision_nll(
                model(
                    states[row : row + 1],
                    features[row : row + 1, : int(mask[row].sum())],
                    torch.ones(1, int(mask[row].sum()), dtype=torch.bool),
                ),
                torch.ones(1, int(mask[row].sum()), dtype=torch.bool),
                targets[row : row + 1],
            )[0]
            for row in range(states.shape[0])
        ]
    )[mask.sum(1) > 1].mean()
    assert torch.allclose(loss, expected, atol=1e-6, rtol=0)


def test_candidate_permutation_permutes_logits_and_preserves_loss_and_choice():
    model = small_policy().eval()
    states, features, mask, targets = inputs(batch=6, candidates=3)
    mask[:] = True
    permutation = torch.tensor([2, 0, 1])
    permuted_features = features[:, permutation, :]
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    permuted_targets = inverse[targets]

    original_logits = model(states, features, mask)
    permuted_logits = model(states, permuted_features, mask)
    assert torch.allclose(
        permuted_logits, original_logits[:, permutation], atol=1e-6, rtol=0
    )
    original_loss = masked_cross_entropy(
        original_logits, mask, targets, eligible=mask.sum(1) > 1
    )
    permuted_loss = masked_cross_entropy(
        permuted_logits, mask, permuted_targets, eligible=mask.sum(1) > 1
    )
    assert torch.allclose(original_loss, permuted_loss, atol=1e-6, rtol=0)

    original_choice = greedy_indices(original_logits, mask)
    permuted_choice = greedy_indices(permuted_logits, mask)
    for row in range(states.shape[0]):
        assert int(permuted_choice[row]) == int(inverse[original_choice[row]])


def test_greedy_ties_choose_lowest_local_candidate_index():
    mask = torch.ones(1, 3, dtype=torch.bool)
    assert int(greedy_indices(torch.tensor([[1.0, 1.0, 1.0]]), mask)[0]) == 0
    assert int(greedy_indices(torch.tensor([[1.0, 1.0, 0.0]]), mask)[0]) == 0
    assert int(greedy_indices(torch.tensor([[0.0, 2.0, 2.0]]), mask)[0]) == 1
    masked = torch.tensor([[3.0, 100.0, 3.0]])
    partial = torch.tensor([[True, False, True]])
    assert int(greedy_indices(masked, partial)[0]) == 0
    assert int(
        greedy_indices(
            torch.tensor([[-1.0, -1.0, -1.0]]), torch.ones(1, 3, dtype=torch.bool)
        )[0]
    ) == 0
    only_last = torch.tensor([[True, False, False]])
    assert int(greedy_indices(torch.tensor([[5.0, 9.0, 9.0]]), only_last)[0]) == 0


def test_sample_respects_mask_and_reproduces_with_generator():
    logits = torch.tensor([[100.0, 0.0, 50.0], [0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False], [False, False, True]])
    generator = torch.Generator().manual_seed(1234)
    draws = sample_indices(
        logits.repeat_interleave(20, dim=0),
        mask.repeat_interleave(20, dim=0),
        generator=generator,
    )
    assert torch.equal(draws[:20].sort().values, torch.tensor([0] * 20))
    assert torch.equal(draws[20:], torch.full((20,), 2, dtype=torch.int64))

    first = sample_indices(logits, mask, generator=torch.Generator().manual_seed(9))
    second = sample_indices(logits, mask, generator=torch.Generator().manual_seed(9))
    assert torch.equal(first, second)
    assert int(first[0]) in (0, 1)
    assert int(first[1]) == 2


def test_masked_cross_entropy_weights_each_decision_equally():
    logits = torch.tensor(
        [[0.0, 0.0, -100.0], [-5.0, 5.0, 0.0], [100.0, 0.0, 0.0]]
    )
    mask = torch.tensor(
        [[True, True, False], [True, True, True], [True, False, False]]
    )
    targets = torch.tensor([0, 1, 0])
    eligible = mask.sum(dim=1) > 1
    nll = decision_nll(logits, mask, targets)
    expected_nll = torch.tensor(
        [
            math.log(2.0),
            math.log1p(math.exp(-5.0) + math.exp(-10.0)),
            0.0,
        ]
    )
    assert torch.allclose(nll, expected_nll, atol=1e-6, rtol=0)
    loss = masked_cross_entropy(logits, mask, targets, eligible=eligible)
    expected = nll[eligible].mean()
    assert torch.allclose(loss, expected, atol=1e-6, rtol=0)
    only_second = masked_cross_entropy(
        logits,
        mask,
        targets,
        eligible=torch.tensor([False, True, False]),
    )
    assert torch.allclose(only_second, nll[1], atol=1e-6, rtol=0)
    forced = decision_nll(
        torch.tensor([[7.0]]), torch.tensor([[True]]), torch.tensor([0])
    )
    assert torch.equal(forced, torch.zeros(1))


def test_invalid_shapes_targets_masks_and_dtypes_raise():
    model = small_policy()
    states, features, mask, targets = inputs()
    with pytest.raises(DataError):
        model(states.double(), features, mask)
    with pytest.raises(DataError):
        model(states, features.double(), mask)
    with pytest.raises(DataError):
        model(states, features, mask.to(torch.int64))
    with pytest.raises(DataError):
        model(states, features[:, :, :38], mask)
    with pytest.raises(DataError):
        model(states, features, torch.zeros_like(mask))
    with pytest.raises(DataError):
        model.half()(states.half(), features.half(), mask)

    logits = torch.zeros(2, 3)
    full_mask = torch.ones(2, 3, dtype=torch.bool)
    with pytest.raises(DataError):
        decision_nll(logits, full_mask, torch.tensor([0, 3]))
    with pytest.raises(DataError):
        decision_nll(logits, full_mask, torch.tensor([0, -1]))
    with pytest.raises(DataError):
        decision_nll(
            logits,
            torch.tensor([[True, False, False], [False, False, False]]),
            torch.tensor([0, 0]),
        )
    with pytest.raises(DataError):
        masked_cross_entropy(
            logits, full_mask, torch.tensor([0, 0]), eligible=torch.zeros(2, dtype=torch.bool)
        )
    with pytest.raises(DataError):
        greedy_indices(torch.tensor([[float("nan"), 0.0]]), torch.ones(1, 2, dtype=torch.bool))


def test_public_selection_entries_keep_their_own_validation():
    logits = torch.zeros(2, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    targets = torch.tensor([0, 1])
    with pytest.raises(DataError):
        masked_log_probs(logits.double(), mask)
    with pytest.raises(DataError):
        masked_log_probs(logits, torch.zeros(2, 3, dtype=torch.bool))
    with pytest.raises(DataError):
        decision_nll(
            logits,
            torch.tensor([[True, False, False], [False, True, True]]),
            torch.tensor([1, 0]),
            check_finite=False,
        )
    with pytest.raises(DataError):
        masked_cross_entropy(
            logits,
            mask,
            targets,
            eligible=torch.tensor([1, 0]),
            check_finite=False,
        )
    with pytest.raises(DataError):
        sample_indices(logits.double(), mask, generator=torch.Generator())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_sampling_rejects_generator_on_a_different_device():
    logits = torch.zeros(2, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    generator = torch.Generator(device="cuda")
    with pytest.raises(DataError, match="generator device.*does not match"):
        sample_indices(logits, mask, generator=generator)


def test_masked_cross_entropy_gradient_ignores_padding():
    logits = torch.tensor(
        [[0.2, -0.4, 50.0], [1.5, 0.1, -2.0]], requires_grad=True
    )
    mask = torch.tensor([[True, True, False], [True, True, False]])
    loss = masked_cross_entropy(logits, mask, torch.tensor([1, 0]))
    loss.backward()
    gradient = logits.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[~mask]) == 0
    assert torch.allclose(
        gradient[mask].view(2, 2).sum(dim=1),
        torch.zeros(2),
        atol=1e-6,
        rtol=0,
    )


def test_activation_names_match_factory_and_build_policies():
    assert set(policy_module._ACTIVATION_FACTORIES) == set(ACTIVATION_NAMES)
    expected = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
    for name in ACTIVATION_NAMES:
        model = build_policy(
            ModelConfig(
                state_hidden=(4,), scorer_hidden=(4,), activation=name
            ),
            seed=1,
        )
        assert isinstance(model.state_encoder[-1], expected[name])
        assert isinstance(model.candidate_scorer[-1], nn.Linear)
    with pytest.raises(ConfigurationError):
        ModelConfig(activation="unknown")
