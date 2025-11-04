"""Split utilities for the dataset pipeline."""

from collections import Counter
from typing import Dict, Iterable, List, Sequence

import numpy as np


SplitRatios = Dict[str, float]


def _normalize_ratios(ratios: SplitRatios) -> SplitRatios:
    if not ratios:
        raise ValueError("Split ratios dictionary cannot be empty")
    positive = {k: float(v) for k, v in ratios.items() if float(v) > 0}
    if not positive:
        raise ValueError("At least one split ratio must be positive")
    total = sum(positive.values())
    return {k: v / total for k, v in positive.items()}


def plan_bucket(
    count: int,
    ratios: SplitRatios | None = None,
    rng: np.random.Generator | None = None,
) -> List[str]:
    """Return a list of split names for `count` items respecting ratios."""

    if count <= 0:
        return []

    ratios = _normalize_ratios(ratios or {"train": 0.7, "val": 0.15, "test": 0.15})
    rng = rng or np.random.default_rng()

    raw_counts = {name: ratio * count for name, ratio in ratios.items()}
    floor_counts = {name: int(np.floor(val)) for name, val in raw_counts.items()}
    assigned = sum(floor_counts.values())
    remainder = count - assigned
    if remainder:
        fractional = sorted(
            ((raw_counts[name] - floor_counts[name], name) for name in ratios),
            reverse=True,
        )
        for _, split in fractional[:remainder]:
            floor_counts[split] += 1

    assignments: List[str] = []
    for split, qty in floor_counts.items():
        assignments.extend([split] * qty)

    # In case rounding produced too many items, truncate and ensure length.
    rng.shuffle(assignments)
    return assignments[:count]


def iter_bucket_assignments(
    labels: Sequence[str],
    ratios: SplitRatios | None = None,
    seed: int = 0,
) -> Iterable[str]:
    """Yield split names for each label in the provided order."""

    rng = np.random.default_rng(seed)
    per_label_map: Dict[str, List[str]] = {}
    counters: Dict[str, int] = {}
    counts = Counter(labels)

    for label in labels:
        plan = per_label_map.get(label)
        if plan is None:
            plan = plan_bucket(counts[label], ratios, rng)
            per_label_map[label] = plan
            counters[label] = 0
        idx = counters[label]
        counters[label] = idx + 1
        yield plan[idx]


__all__ = ["plan_bucket", "iter_bucket_assignments"]
