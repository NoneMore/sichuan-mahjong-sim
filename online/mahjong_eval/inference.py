"""Frozen neural policy inference for online game evaluation.

A :class:`FrozenPolicy` owns one eval-mode candidate policy loaded from an
offline checkpoint, or adopted from an already loaded ``LoadedCheckpoint``.
Pending decisions need only expose ``state`` and ``candidate_features`` arrays;
batching, padding, masking and greedy selection reuse the offline package. No
native extension, training data, or run directory is required.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor

from mahjong_bc.checkpoint import LoadedCheckpoint, load_policy_checkpoint
from mahjong_bc.config import ModelConfig
from mahjong_bc.constants import (
    FORCED_CANDIDATE_COUNT,
    ROLE_BEST,
    ROLE_POLICY,
)
from mahjong_bc.data import DecisionInputs, pad_decision_inputs
from mahjong_bc.errors import CheckpointError, ConfigurationError, DataError
from mahjong_bc.policy import greedy_indices
from mahjong_bc.train import resolve_device

#: Frozen inference accepts selected-best and plain frozen policy artifacts.
#: ``last`` remains a training-recovery artifact and is deliberately rejected.
_SUPPORTED_ROLES = (ROLE_BEST, ROLE_POLICY)

__all__ = ["FrozenPolicy"]


class FrozenPolicy:
    """Selected policy weights exposed as a greedy decision selector.

    The checkpoint loader validates container, model, and encoding versions
    before model construction. :meth:`from_loaded` reuses an already validated
    ``LoadedCheckpoint`` without reading its path again. The policy is
    evaluated in eval mode without gradients; single-candidate decisions are
    submitted as index zero without model inference.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device = "cpu",
    ) -> None:
        target = resolve_device(device)
        loaded = load_policy_checkpoint(checkpoint, device=target)
        self._adopt(loaded, target, clone_model=False)

    @classmethod
    def from_loaded(
        cls,
        loaded: LoadedCheckpoint,
        device: str | torch.device = "cpu",
    ) -> FrozenPolicy:
        """Build a frozen policy from an already loaded checkpoint.

        The checkpoint file is not read again. The policy owns a frozen copy
        of ``loaded.model`` placed on ``device``, so removing or replacing the
        original path, or creating another policy from the same loaded value,
        cannot alter this policy's weights or device.
        """

        if not isinstance(loaded, LoadedCheckpoint):
            raise ConfigurationError("loaded must be a LoadedCheckpoint")
        target = resolve_device(device)
        policy = cls.__new__(cls)
        policy._adopt(loaded, target, clone_model=True)
        return policy

    def _adopt(
        self,
        loaded: LoadedCheckpoint,
        target: torch.device,
        *,
        clone_model: bool,
    ) -> None:
        if loaded.role not in _SUPPORTED_ROLES:
            raise CheckpointError(
                f"{loaded.path} has checkpoint role {loaded.role!r}; frozen "
                "inference accepts selected 'best' (or 'policy') checkpoints, "
                "not 'last'"
            )
        model = copy.deepcopy(loaded.model) if clone_model else loaded.model
        model.to(target)
        model.eval().requires_grad_(False)
        devices = {parameter.device for parameter in model.parameters()}
        devices.update(buffer.device for buffer in model.buffers())
        if not devices:
            devices = {target}
        if len(devices) != 1:
            raise CheckpointError(
                "loaded policy model spans multiple devices: "
                + ", ".join(sorted(str(device) for device in devices))
            )
        actual = next(iter(devices))
        if actual.type != target.type or (
            target.index is not None and actual.index != target.index
        ):
            raise CheckpointError(
                f"loaded policy model is on {actual}, requested {target}"
            )
        self._device = actual
        self._loaded = replace(loaded, model=model) if clone_model else loaded
        self._model = model

    @property
    def checkpoint(self) -> Path:
        return self._loaded.path

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def role(self) -> str:
        return self._loaded.role

    @property
    def encoding_version(self) -> int:
        return self._loaded.encoding_version

    @property
    def model_config(self) -> ModelConfig:
        return self._loaded.model_config

    @property
    def metrics(self) -> dict[str, object] | None:
        return None if self._loaded.metrics is None else dict(self._loaded.metrics)

    def metadata(self) -> dict[str, object]:
        """Return checkpoint metadata without training or dataset access."""

        return self._loaded.metadata()

    def select_indices(self, decisions: Iterable[object]) -> list[int]:
        """Return one greedy local candidate index per decision, in input order.

        Every decision must expose ``state`` and ``candidate_features`` arrays.
        Multi-candidate decisions are padded to the batch maximum and scored
        together; ties choose the lowest local index, which is canonical order.
        Valid non-finite logits raise :class:`~mahjong_bc.errors.DataError`
        before any index is returned. A single-candidate decision bypasses
        inference and contributes index zero, preserving input counts and
        alignment.
        """

        try:
            items = list(decisions)
        except TypeError as exc:
            raise DataError(
                "decisions must be an iterable of decision objects"
            ) from exc
        if not items:
            return []

        inputs = pad_decision_inputs(items)
        selected = [0] * len(items)
        active = [
            row
            for row, count in enumerate(inputs.mask.sum(dim=1).tolist())
            if count > FORCED_CANDIDATE_COUNT
        ]
        if not active:
            return selected
        rows = torch.tensor(active, dtype=torch.int64)
        active_inputs = DecisionInputs(
            states=inputs.states.index_select(0, rows),
            candidates=inputs.candidates.index_select(0, rows),
            mask=inputs.mask.index_select(0, rows),
        )
        for row, index in zip(active, self._select(active_inputs)):
            selected[row] = index
        return selected

    def _score(self, inputs: DecisionInputs) -> Tensor:
        """Score one batch already placed on the policy device."""

        with torch.no_grad():
            return self._model(inputs.states, inputs.candidates, inputs.mask)

    def _select(self, inputs: DecisionInputs) -> list[int]:
        moved = inputs.to(self._device)
        logits = self._score(moved)
        indices = greedy_indices(logits, moved.mask)
        return [int(value) for value in indices.tolist()]
