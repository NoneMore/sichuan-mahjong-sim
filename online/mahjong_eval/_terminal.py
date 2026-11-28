"""Settlement invariants shared by execution and persisted comparison records."""

from collections.abc import Sequence

SEAT_COUNT = 4
END_REASONS = ("three_winners", "wall_exhausted")
U64_MAX = 2**64 - 1


def is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_settlement(
    scores: Sequence[int], winners: Sequence[int], end_reason: str
) -> tuple[int, ...]:
    if len(scores) != SEAT_COUNT:
        raise ValueError(f"scores must contain {SEAT_COUNT} entries")
    for score in scores:
        _require_int_range(score, "score", -(2**31), 2**31 - 1)
    if sum(scores) != 0:
        raise ValueError(f"scores must sum to zero, got {sum(scores)}")

    seen: set[int] = set()
    for seat in winners:
        _require_int_range(seat, "winner", 0, SEAT_COUNT - 1)
        if seat in seen:
            raise ValueError(f"winner seat {seat} is listed more than once")
        seen.add(seat)

    count = len(seen)
    if not isinstance(end_reason, str) or not end_reason:
        raise ValueError("end_reason must be a nonempty string")
    if end_reason not in END_REASONS:
        raise ValueError(
            f"unsupported end_reason {end_reason!r}; expected one of {list(END_REASONS)}"
        )
    if end_reason == "three_winners" and count != 3:
        raise ValueError(f"three_winners requires exactly three winners, got {count}")
    if end_reason == "wall_exhausted" and count >= 3:
        raise ValueError(
            f"wall_exhausted requires fewer than three winners, got {count}"
        )
    return tuple(sorted(seen))


def _require_int_range(value: int, label: str, minimum: int, maximum: int) -> None:
    if not is_int(value):
        raise ValueError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}, got {value}")
    if value > maximum:
        raise ValueError(f"{label} must be at most {maximum}, got {value}")
