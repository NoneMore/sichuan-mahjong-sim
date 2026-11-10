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


class CandidatePolicy(nn.Module):
    """Pointwise candidate scorer with one shared state representation.

    Architecture::

        state -> state MLP -> h
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

    def forward(self, states: Tensor, candidates: Tensor, mask: Tensor) -> Tensor:
        return self.score(states, candidates, mask)


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


def _validate_logits_mask(logits: Tensor, mask: Tensor) -> None:
    if not isinstance(logits, Tensor) or logits.dtype != torch.float32:
        raise DataError("logits must be a float32 tensor")
    if logits.dim() != 2:
        raise DataError(f"logits must have shape [B, C], got {tuple(logits.shape)}")
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
        raise DataError("mask must be a boolean tensor")
    if mask.shape != logits.shape:
        raise DataError(
            f"mask must have shape {tuple(logits.shape)}, got {tuple(mask.shape)}"
        )
    if mask.shape[1] == 0:
        raise DataError("each decision must have at least one candidate")
    if not bool(mask.any(dim=1).all()):
        raise DataError("every decision must have at least one valid candidate")
    if logits.device != mask.device:
        raise DataError("logits and mask must be on the same device")


def _require_finite_valid(logits: Tensor, mask: Tensor) -> None:
    if not bool(torch.isfinite(logits.masked_select(mask)).all()):
        raise DataError("valid candidate logits must be finite")


def _validate_targets(logits: Tensor, mask: Tensor, targets: Tensor) -> None:
    if not isinstance(targets, Tensor):
        raise DataError("targets must be a torch tensor")
    if targets.dim() != 1 or targets.shape[0] != logits.shape[0]:
        raise DataError(
            f"targets must have shape [{logits.shape[0]}], "
            f"got {tuple(targets.shape)}"
        )
    if targets.dtype not in (torch.int64, torch.int32):
        raise DataError(f"targets must be an integer tensor, got {targets.dtype}")
    if targets.device != logits.device:
        raise DataError("targets and logits must be on the same device")
    if targets.numel() == 0:
        return
    if int(targets.min()) < 0 or int(targets.max()) >= logits.shape[1]:
        raise DataError("target index is outside the padded candidate range")
    valid_target = mask.gather(1, targets.unsqueeze(1).long()).squeeze(1)
    if not bool(valid_target.all()):
        raise DataError("target index does not reference a valid candidate")


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
    masked = logits.masked_fill(~mask, float("-inf"))
    return torch.log_softmax(masked, dim=-1)


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
    log_probs = masked_log_probs(logits, mask, check_finite=check_finite)
    return -log_probs.gather(1, targets.unsqueeze(1).long()).squeeze(1)


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


def greedy_indices(logits: Tensor, mask: Tensor, *, check_finite: bool = True) -> Tensor:
    """Return the highest-logit valid candidate per decision.

    Ties are resolved by the lowest local candidate index, which is the
    canonical candidate order. Implementation avoids relying on the documented
    but platform-sensitive first-maximum behavior of ``argmax``.
    """

    _validate_logits_mask(logits, mask)
    if check_finite:
        _require_finite_valid(logits, mask)
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
