from __future__ import annotations


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile of empty values")
    if pct < 0 or pct > 1:
        raise ValueError("percentile out of range")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("median of empty values")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
