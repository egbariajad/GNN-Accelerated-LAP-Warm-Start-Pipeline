"""Data utilities for GNN training and evaluation."""

from .generators import (
    SYNTHETIC_FAMILIES,
    SyntheticInstance,
    generate_synthetic_instance,
)
__all__ = [
    "SYNTHETIC_FAMILIES",
    "SyntheticInstance",
    "generate_synthetic_instance",
]


def make_split_plan(*args, **kwargs):  # type: ignore[override]
    """Lazily import split planner to avoid heavy dependencies."""
    from .splits import make_split_plan as _make_split_plan

    return _make_split_plan(*args, **kwargs)


def generate_dataset(*args, **kwargs):  # type: ignore[override]
    """Lazily import dataset generator to avoid mandatory h5py on import."""
    from .generate_dataset import generate_dataset as _generate

    return _generate(*args, **kwargs)
