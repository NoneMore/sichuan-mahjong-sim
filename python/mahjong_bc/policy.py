"""Shared float32 candidate-scoring policy and masked selection primitives.

The policy exposes logits for a batch of decisions. Offline behavior cloning
uses masked cross-entropy; closed-loop evaluation can use the same logits with
greedy or sampled selection. Candidate position is never an input, so permuting
candidate rows permutes logits but cannot change a semantic decision.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .data import _PackedCandidateBatch
from .errors import ConfigurationError, DataError

_ACTIVATION_FACTORIES: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


def _activation(name: str) -> nn.Module:
    try:
        return _ACTIVATION_FACTORIES[name]()
    except KeyError as exc:  # pragma: no cover - guarded by ModelConfig
        raise DataError(f"unsupported activation {name!r}") from exc


def _mlp(
    widths: tuple[int, ...], activation: str, *, activate_final: bool = True
) -> nn.Sequential:
    layers: list[nn.Module] = []
    pairs = list(zip(widths, widths[1:]))
    for index, (source, target) in enumerate(pairs):
        layers.append(nn.Linear(source, target))
        if activate_final or index + 1 < len(pairs):
            layers.append(_activation(activation))
    return nn.Sequential(*layers)


_SUITS = 3
_RANKS = 9
_TILES = _SUITS * _RANKS
_SEATS = 4
_MELD_SLOTS = 4
_MELD_FIELDS = 1 + _TILES + 4 + 5
_PLAYER_FIELDS = 7 + _MELD_SLOTS * _MELD_FIELDS
_DISCARD_PLANES = _SEATS * 3
_CONTEXT_FIELDS = 9
_RESPONSE_FIELDS = 1 + _TILES + 5
_TILE_PLANES = 1 + _DISCARD_PLANES + _SEATS * (4 + 5) + _SEATS + 1
_GLOBAL_FIELDS = _SEATS * 4 + _CONTEXT_FIELDS + 6


def _suit_rank_features(states: Tensor) -> tuple[Tensor, Tensor]:
    """Rearrange the Rust v1 state into tile planes and global scalars."""
    hand, players, discards, context, response = states.split(
        (
            _TILES,
            _SEATS * _PLAYER_FIELDS,
            _DISCARD_PLANES * _TILES,
            _CONTEXT_FIELDS,
            _RESPONSE_FIELDS,
        ),
        dim=1,
    )
    batch = states.shape[0]
    players = players.reshape(batch, _SEATS, _PLAYER_FIELDS)
    melds = players[..., 7:].reshape(batch, _SEATS, _MELD_SLOTS, _MELD_FIELDS)
    _, meld_tiles, meld_attributes = melds.split((1, _TILES, 4 + 5), dim=-1)
    # Within each owner's melds, a tile kind identifies at most one occupied
    # slot. This attaches kind and source to the tile without encoding slot order.
    meld_planes = (meld_tiles.transpose(-2, -1) @ meld_attributes).transpose(-2, -1)
    missing_suit = players[..., 1:4].unsqueeze(-1).expand(-1, -1, -1, _RANKS)
    response_present, response_tile, response_context = response.split(
        (1, _TILES, 5), dim=1
    )
    planes = torch.cat(
        (
            hand.unsqueeze(1),
            discards.reshape(batch, _DISCARD_PLANES, _TILES),
            meld_planes.reshape(batch, _SEATS * (4 + 5), _TILES),
            missing_suit.reshape(batch, _SEATS, _TILES),
            response_tile.unsqueeze(1),
        ),
        dim=1,
    )
    global_features = torch.cat(
        (
            players[..., :1].flatten(1),
            players[..., 4:7].reshape(batch, 3 * _SEATS),
            context,
            response_present,
            response_context,
        ),
        dim=1,
    )
    return planes.reshape(batch, _TILE_PLANES, _SUITS, _RANKS), global_features


class _RankResidualBlock(nn.Module):
    def __init__(self, channels: int, activation: str) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), padding=(0, 1)),
            _activation(activation),
            nn.Conv2d(channels, channels, (1, 3), padding=(0, 1)),
        )
        self.activation = _activation(activation)

    def forward(self, tiles: Tensor) -> Tensor:
        return self.activation(tiles + self.residual(tiles))


class _SuitRankEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        channels = 32
        # Kernel height one applies the same rank convolution to every suit.
        self.ranks = nn.Sequential(
            nn.Conv2d(_TILE_PLANES, channels, (1, 3), padding=(0, 1)),
            _activation(config.activation),
            _RankResidualBlock(channels, config.activation),
            _RankResidualBlock(channels, config.activation),
        ).to(memory_format=torch.channels_last)
        self.fuse = _mlp(
            (channels * _TILES + _GLOBAL_FIELDS, *config.state_hidden),
            config.activation,
        )

    def forward(self, states: Tensor) -> Tensor:
        planes, global_features = _suit_rank_features(states)
        local = self.ranks(planes.contiguous(memory_format=torch.channels_last))
        # Fuse the full rank maps in canonical suit order. The teacher can use
        # suit identity for ties, while neighboring ranks share the local weights.
        local = local.permute(0, 2, 3, 1).flatten(1)
        return self.fuse(torch.cat((local, global_features), dim=1))


class CandidatePolicy(nn.Module):
    """Pointwise candidate scorer with one shared state representation.

    Architecture::

        state -> configured state encoder -> h
        concat(h, candidate_features) -> scorer MLP -> single logit

    The scorer is shared across all candidates and stages, and has at least one
    nonlinear hidden layer. There is no dropout, normalization, candidate
    position feature, or other batch-dependent behavior.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else ModelConfig()
        widths = (
            self.config.state_width,
            *self.config.state_hidden,
        )
        if self.config.architecture == "suit-rank-resnet":
            self.state_encoder = _SuitRankEncoder(self.config)
        else:
            self.state_encoder = _mlp(widths, self.config.activation)
        state_output = self.config.state_hidden[-1]
        scorer_widths = (
            state_output + self.config.candidate_width,
            *self.config.scorer_hidden,
            1,
        )
        self.candidate_scorer = _mlp(
            scorer_widths, self.config.activation, activate_final=False
        )
        self.float()

    @property
    def encoding_version(self) -> int:
        return self.config.encoding_version

    def state_representation(self, states: Tensor) -> Tensor:
        """Encode each decision state once."""

        if states.dim() != 2 or states.shape[1] != self.config.state_width:
            raise DataError(
                f"states must have shape [B, {self.config.state_width}], "
                f"got {tuple(states.shape)}"
            )
        return self.state_encoder(states)

    def score(
        self,
        states: Tensor,
        candidates: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Return one logit per padded candidate.

        ``candidates`` is ``[B, C, candidate_width]`` and ``mask`` is a boolean
        ``[B, C]`` valid-candidate mask with at least one valid entry per row.
        Padded candidate rows are replaced with zeros before the pointwise
        scorer, so arbitrary padding values cannot affect valid logits.
        """

        _validate_policy_inputs(self, states, candidates, mask)
        hidden = self.state_representation(states)
        expanded = hidden.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        padded_candidates = masked_candidate_inputs(candidates, mask)
        scorer_input = torch.cat((expanded, padded_candidates), dim=-1)
        return self.candidate_scorer(scorer_input).squeeze(-1)

    def forward(
        self,
        states: Tensor,
        candidates: Tensor,
        mask: Tensor,
        *,
        packed: bool = False,
        candidate_positions: Tensor | None = None,
    ) -> Tensor:
        """Score padded or packed candidates.

        ``candidate_positions`` is trusted internal packed-layout metadata:
        strictly increasing, in-range padded positions matching ``mask`` in
        row-major order. Internal batching establishes these value invariants
        on the host; direct packed callers can omit it to use the mask layout.
        """

        if packed:
            return _packed_candidate_logits(
                self, states, candidates, mask, candidate_positions
            )
        return self.score(states, candidates, mask)


def _supports_packed_candidate_scoring(model: object) -> bool:
    """Custom policy classes retain the padded forward interface."""

    return type(model) is CandidatePolicy


class _ExpandDecisionRows(torch.autograd.Function):
    """Repeat each decision's encoded state once per flat valid candidate.

    CPU-derived positions avoid device-side mask expansion. A plain
    ``index_select`` backward uses atomic ``index_add_`` for duplicate decision
    rows, which is nondeterministic on CUDA. Instead, backward copies gradients
    to unique padded positions and sums the candidate axis in a fixed order.
    """

    @staticmethod
    def forward(
        ctx, hidden: Tensor, candidate_positions: Tensor, candidate_columns: int
    ) -> Tensor:
        ctx.rows = hidden.shape[0]
        ctx.candidate_columns = candidate_columns
        ctx.save_for_backward(candidate_positions)
        decisions = candidate_positions.div(
            candidate_columns, rounding_mode="floor"
        )
        return hidden.index_select(0, decisions)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None, None]:
        (candidate_positions,) = ctx.saved_tensors
        padded = grad_output.new_zeros(
            (ctx.rows * ctx.candidate_columns, grad_output.shape[1])
        )
        padded.index_copy_(0, candidate_positions, grad_output)
        return padded.view(ctx.rows, ctx.candidate_columns, -1).sum(dim=1), None, None


def _packed_candidate_logits(
    model: CandidatePolicy,
    states: Tensor,
    candidates: Tensor,
    mask: Tensor,
    candidate_positions: Tensor | None = None,
) -> Tensor:
    """Score flat valid candidates and rebuild the padded logit layout.

    Flat rows match the ``[B, C]`` mask's true entries in row-major order.
    Internal batches supply unique padded positions for fixed-shape gathering
    and scattering, avoiding CUDA synchronization from boolean indexing.
    Padded logits are zero and must be masked by consumers.

    Direct packed callers without positions retain the mask-driven layout.
    """

    hidden = model.state_representation(states)
    if candidate_positions is None:
        expanded = hidden.unsqueeze(1).expand(-1, mask.shape[1], -1)[mask]
        flat = model.candidate_scorer(
            torch.cat((expanded, candidates), dim=-1)
        ).squeeze(-1)
        logits = flat.new_zeros(mask.shape)
        logits[mask] = flat
        return logits
    _validate_candidate_positions(candidate_positions, candidates)
    expanded = _ExpandDecisionRows.apply(
        hidden, candidate_positions, mask.shape[1]
    )
    flat = model.candidate_scorer(
        torch.cat((expanded, candidates), dim=-1)
    ).squeeze(-1)
    logits = flat.new_zeros(mask.shape)
    logits.view(-1).index_copy_(0, candidate_positions, flat)
    return logits


def _validate_candidate_positions(
    candidate_positions: Tensor, candidates: Tensor
) -> None:
    """Structural flat-row layout checks that never read tensor values."""

    if (
        not isinstance(candidate_positions, Tensor)
        or candidate_positions.dtype != torch.int64
    ):
        raise DataError("candidate positions must be an int64 tensor")
    if candidate_positions.shape != (candidates.shape[0],):
        raise DataError(
            f"candidate positions must have shape ({candidates.shape[0]},), "
            f"got {tuple(candidate_positions.shape)}"
        )
    if candidate_positions.device != candidates.device:
        raise DataError(
            "candidate positions and candidates must be on the same device"
        )


def _batch_logits(model: CandidatePolicy, batch: object) -> Tensor:
    """Padded logits for a public, pruned or flat-candidate internal batch."""

    if not isinstance(batch, _PackedCandidateBatch):
        return model(batch.states, batch.candidates, batch.mask)
    return model(
        batch.states,
        batch.candidates,
        batch.mask,
        packed=True,
        candidate_positions=batch.candidate_positions,
    )


def build_policy(config: ModelConfig | None = None, *, seed: int | None = None) -> CandidatePolicy:
    """Create a float32 policy, optionally reseeding global torch RNG first."""

    if seed is not None:
        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or seed < 0
            or seed > 2**64 - 1
        ):
            raise ConfigurationError("seed must fit in an unsigned 64-bit integer")
        torch.manual_seed(seed)
    return CandidatePolicy(config).float()


def _validate_policy_inputs(
    model: CandidatePolicy,
    states: Tensor,
    candidates: Tensor,
    mask: Tensor,
) -> None:
    config = model.config
    if not isinstance(states, Tensor) or not isinstance(candidates, Tensor):
        raise DataError("states and candidates must be torch tensors")
    if not isinstance(mask, Tensor):
        raise DataError("mask must be a torch tensor")
    if states.dtype != torch.float32:
        raise DataError(f"states must be float32, got {states.dtype}")
    if candidates.dtype != torch.float32:
        raise DataError(f"candidates must be float32, got {candidates.dtype}")
    if mask.dtype != torch.bool:
        raise DataError(f"mask must be bool, got {mask.dtype}")
    if states.dim() != 2 or states.shape[1] != config.state_width:
        raise DataError(
            f"states must have shape [B, {config.state_width}], "
            f"got {tuple(states.shape)}"
        )
    if candidates.dim() != 3 or candidates.shape[2] != config.candidate_width:
        raise DataError(
            f"candidates must have shape [B, C, {config.candidate_width}], "
            f"got {tuple(candidates.shape)}"
        )
    if candidates.shape[0] != states.shape[0]:
        raise DataError("states and candidates must have the same batch size")
    if candidates.shape[1] == 0:
        raise DataError("each batch must have at least one candidate column")
    if mask.shape != candidates.shape[:2]:
        raise DataError(
            f"mask must have shape {tuple(candidates.shape[:2])}, "
            f"got {tuple(mask.shape)}"
        )
    if not bool(mask.any(dim=1).all()):
        raise DataError("every decision must have at least one valid candidate")
    if states.device != candidates.device or states.device != mask.device:
        raise DataError("states, candidates, and mask must be on the same device")
    parameter = next(model.parameters(), None)
    if parameter is not None and parameter.dtype != torch.float32:
        raise DataError(
            f"policy parameters must be float32, got {parameter.dtype}"
        )


def masked_candidate_inputs(candidates: Tensor, mask: Tensor) -> Tensor:
    """Replace padded candidate rows with zeros before pointwise scoring."""

    return candidates.masked_fill(~mask.unsqueeze(-1), 0.0)


def _validate_mask_structure(
    mask: Tensor, shape: tuple[int, int], device: torch.device
) -> None:
    """Check mask dtype/shape/device/device agreement without host reads."""

    if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
        raise DataError("mask must be a boolean tensor")
    if mask.shape != shape:
        raise DataError(f"mask must have shape {shape}, got {tuple(mask.shape)}")
    if shape[1] == 0:
        raise DataError("each decision must have at least one candidate")
    if mask.device != device:
        raise DataError("logits and mask must be on the same device")


def _validate_logits_mask_structure(logits: Tensor, mask: Tensor) -> None:
    """Structural logits/mask checks without reading either value to the host.

    Public entry points also check that every decision has a valid candidate;
    internal batch paths validate that predicate once per batch on the host.
    """

    if not isinstance(logits, Tensor) or logits.dtype != torch.float32:
        raise DataError("logits must be a float32 tensor")
    if logits.dim() != 2:
        raise DataError(f"logits must have shape [B, C], got {tuple(logits.shape)}")
    _validate_mask_structure(mask, tuple(logits.shape), logits.device)


def _validate_logits_mask(logits: Tensor, mask: Tensor) -> None:
    _validate_logits_mask_structure(logits, mask)
    if not bool(mask.any(dim=1).all()):
        raise DataError("every decision must have at least one valid candidate")


def _require_finite_valid(logits: Tensor, mask: Tensor) -> None:
    """Reject non-finite valid logits with one fixed-shape device check.

    ``masked_select`` sizes its result on the device and synchronizes to build
    it. Padding values are irrelevant, so a fixed-shape logical reduction over
    the mask has the same meaning and still raises before any update.
    """

    if not bool((torch.isfinite(logits) | ~mask).all()):
        raise DataError("valid candidate logits must be finite")


def _validate_targets_structure(
    targets: Tensor, batch_size: int, device: torch.device
) -> None:
    """Check target dtype/shape/device/device agreement without host reads."""

    if not isinstance(targets, Tensor):
        raise DataError("targets must be a torch tensor")
    if targets.dim() != 1 or targets.shape[0] != batch_size:
        raise DataError(
            f"targets must have shape [{batch_size}], got {tuple(targets.shape)}"
        )
    if targets.dtype not in (torch.int64, torch.int32):
        raise DataError(f"targets must be an integer tensor, got {targets.dtype}")
    if targets.device != device:
        raise DataError("targets and logits must be on the same device")


def _validate_targets(logits: Tensor, mask: Tensor, targets: Tensor) -> None:
    _validate_targets_structure(targets, logits.shape[0], logits.device)
    if targets.numel() == 0:
        return
    if int(targets.min()) < 0 or int(targets.max()) >= logits.shape[1]:
        raise DataError("target index is outside the padded candidate range")
    valid_target = mask.gather(1, targets.unsqueeze(1).long()).squeeze(1)
    if not bool(valid_target.all()):
        raise DataError("target index does not reference a valid candidate")


def _masked_log_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    """Log-softmax over already validated logits and mask."""

    return torch.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


def masked_log_probs(
    logits: Tensor,
    mask: Tensor,
    *,
    check_finite: bool = True,
) -> Tensor:
    """Log-softmax over valid candidates; padded entries become ``-inf``."""

    _validate_logits_mask(logits, mask)
    if check_finite:
        _require_finite_valid(logits, mask)
    return _masked_log_softmax(logits, mask)


def _decision_nll_validated(
    logits: Tensor,
    mask: Tensor,
    targets: Tensor,
) -> Tensor:
    """Per-decision NLL for tensors a caller has already fully validated.

    Internal batch paths check a padded batch once, including value checks, and
    then reuse this primitive for the NLL instead of repeating every public
    check. Public entry points keep their strict validation.
    """

    log_probs = _masked_log_softmax(logits, mask)
    return -log_probs.gather(1, targets.unsqueeze(1).long()).squeeze(1)


def decision_nll(
    logits: Tensor,
    mask: Tensor,
    targets: Tensor,
    *,
    check_finite: bool = True,
) -> Tensor:
    """Negative log-likelihood of each decision target, shaped ``[B]``."""

    _validate_logits_mask(logits, mask)
    _validate_targets(logits, mask, targets)
    if check_finite:
        _require_finite_valid(logits, mask)
    return _decision_nll_validated(logits, mask, targets)


def masked_cross_entropy(
    logits: Tensor,
    mask: Tensor,
    targets: Tensor,
    *,
    eligible: Tensor | None = None,
    check_finite: bool = True,
) -> Tensor:
    """Mean cross-entropy weighting each eligible decision equally.

    ``eligible`` selects decisions with more than one candidate. Forced
    decisions have zero candidate cross-entropy but are excluded by callers
    from training and primary metrics.
    """

    nll = decision_nll(logits, mask, targets, check_finite=check_finite)
    if eligible is None:
        eligible = torch.ones_like(nll, dtype=torch.bool)
    if not isinstance(eligible, Tensor) or eligible.dtype != torch.bool:
        raise DataError("eligible must be a boolean tensor")
    if eligible.shape != nll.shape:
        raise DataError(
            f"eligible must have shape {tuple(nll.shape)}, got {tuple(eligible.shape)}"
        )
    if eligible.device != nll.device:
        raise DataError("eligible and logits must be on the same device")
    if not bool(eligible.any()):
        raise DataError("at least one eligible decision is required")
    return nll.masked_select(eligible).mean()


def _masked_cross_entropy_validated(
    logits: Tensor,
    mask: Tensor,
    targets: Tensor,
    eligible: Tensor,
) -> Tensor:
    """Eligible-mean NLL for tensors a caller has already fully validated.

    ``eligible`` must already be a boolean decision mask on the same device
    with at least one selected decision, and the logits must be finite.
    """

    nll = _decision_nll_validated(logits, mask, targets)
    return nll.masked_select(eligible).mean()


def greedy_indices(logits: Tensor, mask: Tensor, *, check_finite: bool = True) -> Tensor:
    """Return the highest-logit valid candidate per decision.

    Ties are resolved by the lowest local candidate index, which is the
    canonical candidate order. Implementation avoids relying on the documented
    but platform-sensitive first-maximum behavior of ``argmax``.
    """

    _validate_logits_mask(logits, mask)
    if check_finite:
        _require_finite_valid(logits, mask)
    return _greedy_indices_validated(logits, mask)


def _greedy_indices_validated(logits: Tensor, mask: Tensor) -> Tensor:
    """Greedy selection for logits and mask a caller has already validated."""

    masked = logits.masked_fill(~mask, float("-inf"))
    maxima = masked.max(dim=1, keepdim=True).values
    positions = torch.arange(logits.shape[1], device=logits.device)
    candidates = positions.unsqueeze(0).expand_as(logits)
    return torch.where(mask & (logits == maxima), candidates, logits.shape[1]).min(
        dim=1
    ).values


def sample_indices(
    logits: Tensor,
    mask: Tensor,
    *,
    generator: torch.Generator | None = None,
    check_finite: bool = True,
) -> Tensor:
    """Sample one valid candidate per decision from the masked softmax."""

    log_probs = masked_log_probs(logits, mask, check_finite=check_finite)
    probabilities = log_probs.exp()
    if generator is not None and generator.device != logits.device:
        raise DataError(
            f"generator device {generator.device} does not match logits device "
            f"{logits.device}"
        )
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
