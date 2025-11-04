"""GNN components for dual prediction."""

from .dual_gnn import DualGNN
from .one_gnn import OneGNN
from .features import (
    compute_features,
    compute_row_features,
    compute_row_features_torch,
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    ROW_FEATURE_DIM,
)

__all__ = [
    "DualGNN",
    "OneGNN",
    "compute_features",
    "compute_row_features",
    "compute_row_features_torch",
    "EDGE_FEATURE_DIM",
    "NODE_FEATURE_DIM",
    "ROW_FEATURE_DIM",
]
