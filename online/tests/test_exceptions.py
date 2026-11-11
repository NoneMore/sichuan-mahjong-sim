"""Public native exception identities must survive ordinary serialization."""

import pickle

import pytest

from mahjong_eval import _native


@pytest.mark.parametrize("name", [
    "MahjongEvalError", "EnvironmentStateError", "StaleDecisionError",
    "CandidateIndexError", "DecisionLimitError", "GameRuleError",
])
def test_exception_has_importable_identity(name):
    exception = getattr(_native, name)
    assert exception.__module__ == "mahjong_eval._native"
    original = exception("seed 7, decision 9")
    restored = pickle.loads(pickle.dumps(original))
    assert type(restored) is exception
    assert restored.args == original.args
