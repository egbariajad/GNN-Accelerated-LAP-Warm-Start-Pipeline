"""Feature engineering utilities for the DualGNN/OneGNN warm-start pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


POS_FREQS = (1, 2, 4, 8)
TAU = 1e-3
EPS = 1e-9


def _positional_encodings(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, len(POS_FREQS) * 2), dtype=np.float32)
    positions = np.arange(n, dtype=np.float64)
    scale = max(1, n - 1)
    encs = []
    for freq in POS_FREQS:
        angle = 2.0 * np.pi * positions * freq / scale
        encs.append(np.sin(angle))
        encs.append(np.cos(angle))
    return np.stack(encs, axis=-1).astype(np.float32)


def _normalized_ranks(values: np.ndarray, axis: int) -> np.ndarray:
    ranks = np.argsort(np.argsort(values, axis=axis), axis=axis)
    denom = values.shape[axis] - 1
    if denom <= 0:
        return np.zeros_like(values, dtype=np.float32)
    return ranks / denom


@dataclass
class GraphFeatures:
    row_feat: np.ndarray
    col_feat: np.ndarray
    edge_feat: np.ndarray


def compute_features(
    C: np.ndarray,
    *,
    include_reduced_cost: bool = False,
    u: Optional[np.ndarray] = None,
) -> GraphFeatures:
    C = np.asarray(C, dtype=np.float64)
    n = C.shape[0]

    row_min = C.min(axis=1)
    row_max = C.max(axis=1)
    row_mean = C.mean(axis=1)
    row_std = C.std(axis=1)
    row_med = np.median(C, axis=1)
    row_mad = np.median(np.abs(C - row_med[:, None]), axis=1)
    row_mad = np.where(row_mad < EPS, EPS, row_mad)

    row_softmax = np.exp(-C)
    row_softmax /= row_softmax.sum(axis=1, keepdims=True) + EPS
    row_entropy = -(row_softmax * np.log(row_softmax + EPS)).sum(axis=1)

    col_min = C.min(axis=0)
    col_max = C.max(axis=0)
    col_mean = C.mean(axis=0)
    col_std = C.std(axis=0)
    col_med = np.median(C, axis=0)
    col_mad = np.median(np.abs(C - col_med[None, :]), axis=0)
    col_mad = np.where(col_mad < EPS, EPS, col_mad)

    col_softmax = np.exp(-C)
    col_softmax /= col_softmax.sum(axis=0, keepdims=True) + EPS
    col_entropy = -(col_softmax * np.log(col_softmax + EPS)).sum(axis=0)

    scaled_cost = (C - row_med[:, None]) / row_mad[:, None]
    row_rank = _normalized_ranks(C, axis=1)
    col_rank = _normalized_ranks(C, axis=0)

    row_gap = C - row_min[:, None]
    col_gap = C - col_min[None, :]

    row_tie = (row_gap <= TAU).sum(axis=1) / max(1, n)
    col_tie = (col_gap <= TAU).sum(axis=0) / max(1, n)

    row_entropy_broadcast = row_entropy[:, None]
    col_entropy_broadcast = col_entropy[None, :]

    row_tie_matrix = np.repeat(row_tie[:, None], n, axis=1)
    col_tie_matrix = np.repeat(col_tie[None, :], n, axis=0)
    row_entropy_matrix = np.repeat(row_entropy[:, None], n, axis=1)
    col_entropy_matrix = np.repeat(col_entropy[None, :], n, axis=0)

    edge_components = [
        scaled_cost,
        row_rank,
        col_rank,
        row_gap,
        col_gap,
        row_tie_matrix,
        col_tie_matrix,
        row_entropy_matrix,
        col_entropy_matrix,
    ]

    if include_reduced_cost and u is not None:
        u = np.asarray(u, dtype=np.float64)
        v = np.min(C - u[:, None], axis=0)
        reduced = C - u[:, None] - v[None, :]
        edge_components.append(reduced)
    else:
        zero_reduced = np.zeros_like(C)
        edge_components.append(zero_reduced)

    edge_feat = np.stack(edge_components, axis=-1).astype(np.float32)

    pos_row = _positional_encodings(n)
    pos_col = _positional_encodings(n)

    row_feat = np.concatenate(
        [
            row_min[:, None],
            row_max[:, None],
            row_mean[:, None],
            row_std[:, None],
            row_mad[:, None],
            row_entropy[:, None],
            pos_row,
        ],
        axis=1,
    ).astype(np.float32)

    col_stack = np.concatenate(
        [
            col_min[None, :],
            col_max[None, :],
            col_mean[None, :],
            col_std[None, :],
            col_mad[None, :],
            col_entropy[None, :],
            pos_col.T,
        ],
        axis=0,
    )
    col_feat = col_stack.T.astype(np.float32)

    return GraphFeatures(row_feat=row_feat, col_feat=col_feat, edge_feat=edge_feat)


_DUMMY = compute_features(np.zeros((1, 1)), include_reduced_cost=True, u=np.zeros(1))
EDGE_FEATURE_DIM = _DUMMY.edge_feat.shape[-1]
NODE_FEATURE_DIM = _DUMMY.row_feat.shape[1]


def compute_row_features(C: np.ndarray) -> np.ndarray:
    """Compute the 21D row features used by OneGNN."""

    C = np.asarray(C, dtype=np.float64)
    n = C.shape[0]

    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    row_min = C.min(axis=1)
    row_max = C.max(axis=1)
    row_mean = C.mean(axis=1)
    row_std = C.std(axis=1)
    row_med = np.median(C, axis=1)
    row_mad = np.median(np.abs(C - row_med[:, None]), axis=1)
    row_mad = np.where(row_mad < EPS, EPS, row_mad)

    # Stabilized softmax entropy
    Z = C.min(axis=1, keepdims=True)
    exp_neg_costs = np.exp(-(C - Z))
    row_softmax = exp_neg_costs / (exp_neg_costs.sum(axis=1, keepdims=True) + EPS)
    row_entropy = -(row_softmax * np.log(row_softmax + EPS)).sum(axis=1)

    # Second-best gap & normalized competition
    if C.shape[1] >= 2:
        row_two_smallest = np.partition(C, 1, axis=1)[:, :2]
        second_best_gap = row_two_smallest[:, 1] - row_two_smallest[:, 0]

        sorted_costs = np.sort(C, axis=1)
        cost_span = sorted_costs[:, -1] - sorted_costs[:, 0]
        competition = second_best_gap / (cost_span + EPS)
    else:
        second_best_gap = np.zeros(n)
        competition = np.zeros(n)

    # k-nearest statistics
    k = min(10, C.shape[1])
    if k >= 1:
        k_smallest = np.partition(C, k - 1, axis=1)[:, :k]
        k_mean = k_smallest.mean(axis=1)
        k_std = k_smallest.std(axis=1)
    else:
        k_mean = np.zeros(n)
        k_std = np.zeros(n)

    # Assignment difficulty (mean gap inverse)
    if C.shape[1] >= 2:
        sorted_costs = np.sort(C, axis=1)
        diffs = np.diff(sorted_costs, axis=1)
        difficulty = 1.0 / (diffs.mean(axis=1) + EPS)
    else:
        difficulty = np.zeros(n)

    # Near-best density
    near_best = (C <= row_min[:, None] * 1.1).sum(axis=1) / max(1, C.shape[1])

    # Column preference count
    col_min = C.min(axis=0)
    is_col_best = (C == col_min).sum(axis=1) / max(1, C.shape[1])

    pos_row = _positional_encodings(n)

    row_feat = np.concatenate(
        [
            row_min[:, None],
            row_max[:, None],
            row_mean[:, None],
            row_std[:, None],
            row_mad[:, None],
            row_entropy[:, None],
            second_best_gap[:, None],
            competition[:, None],
            k_mean[:, None],
            k_std[:, None],
            difficulty[:, None],
            near_best[:, None],
            is_col_best[:, None],
            pos_row,
        ],
        axis=1,
    ).astype(np.float32)

    return row_feat


def compute_row_features_torch(cost: 'torch.Tensor') -> 'torch.Tensor':
    """CUDA-optimized 21D row feature computation."""

    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available for compute_row_features_torch")
        
    C = cost.float()  # Ensure float32
    device = C.device
    n, m = C.shape
    
    if n == 0:
        return torch.zeros((0, 0), dtype=torch.float32, device=device)
    
    eps = 1e-9
    
    # Basic row statistics (all vectorized on GPU)
    row_min = C.min(dim=1)[0]  # [n]
    row_max = C.max(dim=1)[0]  # [n]
    row_mean = C.mean(dim=1)   # [n]
    row_std = C.std(dim=1)     # [n]
    
    # Median and MAD using sorting (efficient on GPU)
    sorted_C = torch.sort(C, dim=1)[0]  # [n, m]
    mid_idx = m // 2
    if m % 2 == 1:
        row_med = sorted_C[:, mid_idx]
    else:
        row_med = (sorted_C[:, mid_idx-1] + sorted_C[:, mid_idx]) / 2
    
    # MAD computation
    abs_dev = torch.abs(C - row_med.unsqueeze(1))  # [n, m]
    sorted_abs_dev = torch.sort(abs_dev, dim=1)[0]
    if m % 2 == 1:
        row_mad = sorted_abs_dev[:, mid_idx]
    else:
        row_mad = (sorted_abs_dev[:, mid_idx-1] + sorted_abs_dev[:, mid_idx]) / 2
    row_mad = torch.clamp(row_mad, min=eps)
    
    # Stabilized softmax entropy
    Z = row_min.unsqueeze(1)  # [n, 1] - use min for stability
    exp_neg_costs = torch.exp(-(C - Z))  # [n, m]
    row_softmax = exp_neg_costs / (exp_neg_costs.sum(dim=1, keepdim=True) + eps)
    row_entropy = -(row_softmax * torch.log(row_softmax + eps)).sum(dim=1)  # [n]
    
    # Second-best gap using topk (fastest on GPU)
    if m >= 2:
        topk_vals = torch.topk(C, k=2, dim=1, largest=False)[0]  # [n, 2]
        second_best_gap = topk_vals[:, 1] - topk_vals[:, 0]  # [n]
        
        # Competition: normalized gap
        cost_span = row_max - row_min + eps
        competition = second_best_gap / cost_span  # [n]
    else:
        second_best_gap = torch.zeros(n, dtype=torch.float32, device=device)
        competition = torch.zeros(n, dtype=torch.float32, device=device)
    
    # k-nearest statistics (using topk for efficiency)
    k = min(10, m)
    if k >= 1:
        k_smallest = torch.topk(C, k=k, dim=1, largest=False)[0]  # [n, k]
        k_mean = k_smallest.mean(dim=1)  # [n]
        k_std = k_smallest.std(dim=1)    # [n]
    else:
        k_mean = torch.zeros(n, dtype=torch.float32, device=device)
        k_std = torch.zeros(n, dtype=torch.float32, device=device)
    
    # Assignment difficulty (mean gap inverse)
    if m >= 2:
        diffs = sorted_C[:, 1:] - sorted_C[:, :-1]  # [n, m-1]
        difficulty = 1.0 / (diffs.mean(dim=1) + eps)  # [n]
    else:
        difficulty = torch.zeros(n, dtype=torch.float32, device=device)
    
    # Near-best density (broadcast compare on GPU)
    threshold = row_min.unsqueeze(1) * 1.1  # [n, 1]
    near_best = (C <= threshold).float().mean(dim=1)  # [n]
    
    # Column preference count (argmin + bincount)
    col_min_idx = C.argmin(dim=0)  # [m] - row indices of column minimums
    is_col_best = torch.bincount(col_min_idx, minlength=n).float() / m  # [n]
    
    # Positional encodings (convert to torch)
    pos_row_np = _positional_encodings(n)
    pos_row = torch.from_numpy(pos_row_np).to(device=device, dtype=torch.float32)  # [n, 8]
    
    # Concatenate all features
    row_feat = torch.stack([
        row_min,
        row_max, 
        row_mean,
        row_std,
        row_mad,
        row_entropy,
        second_best_gap,
        competition,
        k_mean,
        k_std,
        difficulty,
        near_best,
        is_col_best,
    ], dim=1)  # [n, 13]
    
    # Add positional encodings
    row_feat = torch.cat([row_feat, pos_row], dim=1)  # [n, 21]
    
    return row_feat


_DUMMY_ROW = compute_row_features(np.zeros((1, 1)))
ROW_FEATURE_DIM = _DUMMY_ROW.shape[1] if _DUMMY_ROW.ndim == 2 else 0
