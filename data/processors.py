"""Processing helpers for real-world datasets (OR-Library, SuiteSparse, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional
import gzip
import urllib.request

import numpy as np


@dataclass
class RealInstance:
    name: str
    cost: np.ndarray
    source: str


def parse_or_library_assign(path: str | Path) -> List[np.ndarray]:
    """Parse OR-Library assignment files into dense cost matrices."""

    path = Path(path)
    raw = path.read_text().split()
    if not raw:
        raise ValueError(f"File '{path}' is empty")

    n = int(raw[0])
    values = list(map(float, raw[1:]))
    block = n * n
    if block == 0 or len(values) % block != 0:
        raise ValueError(f"Unexpected OR-Library layout in '{path}'")

    matrices: List[np.ndarray] = []
    offset = 0
    idx = 0
    while offset + block <= len(values):
        chunk = values[offset : offset + block]
        matrices.append(np.array(chunk, dtype=np.float64).reshape(n, n))
        offset += block
        idx += 1
    return matrices


def fetch_suite_sparse(matrix_name: str, *, to_dense: bool = True) -> np.ndarray:
    """Retrieve SuiteSparse matrix by name (requires ssgetpy)."""

    try:
        import ssgetpy  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install ssgetpy to download SuiteSparse matrices") from exc

    result = ssgetpy.fetch(matrix_name, format="coo")
    matrix = result.toarray() if to_dense else result
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Matrix '{matrix_name}' is not square: {mat.shape}")
    return mat


def fetch_string_protein_links(tax_id: str = "9606", version: str = "12.0") -> dict:
    """Download and parse STRING protein links for given taxonomy ID.

    Returns a dict with 'proteins': list of protein IDs, 'edges': list of (i,j,score) tuples.
    Scores are normalized to [0,1].
    """
    url = f"https://stringdb-static.org/download/protein.links.v{version}/{tax_id}.protein.links.v{version}.txt.gz"
    print(f"Downloading STRING protein links from {url}")

    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as f:
            lines = f.read().decode("utf-8").strip().split("\n")

    # Skip header if any
    if lines[0].startswith("#"):
        lines = lines[1:]

    protein_set = set()
    edges = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            p1, p2, score_str = parts[0], parts[1], parts[2]
            score = int(score_str) / 1000.0  # Normalize to [0,1]
            protein_set.add(p1)
            protein_set.add(p2)
            edges.append((p1, p2, score))

    proteins = sorted(protein_set)
    protein_to_idx = {p: i for i, p in enumerate(proteins)}

    # Convert edges to indices
    edge_list = [(protein_to_idx[p1], protein_to_idx[p2], score) for p1, p2, score in edges]

    return {"proteins": proteins, "edges": edge_list}


def create_string_cost_matrices(string_data: dict, subset_size: int = 1000, num_matrices: int = 10) -> List[np.ndarray]:
    """Create bipartite cost matrices from STRING data by partitioning subsets.

    For each matrix, take a random subset of proteins, partition into left/right,
    and set cost[i][j] = 1 - score for edges, 1.0 for non-edges.
    """
    proteins = string_data["proteins"]
    edges = string_data["edges"]

    # Create adjacency dict for quick lookup
    adj = {}
    for i, j, score in edges:
        if i not in adj:
            adj[i] = {}
        if j not in adj:
            adj[j] = {}
        adj[i][j] = score
        adj[j][i] = score  # Undirected

    matrices = []
    np.random.seed(42)  # For reproducibility

    for _ in range(num_matrices):
        # Random subset
        subset = np.random.choice(len(proteins), size=min(subset_size, len(proteins)), replace=False)
        subset_proteins = [proteins[i] for i in subset]

        # Partition into left and right
        n = len(subset_proteins) // 2
        left = subset_proteins[:n]
        right = subset_proteins[n : 2 * n]

        # Map to local indices
        left_idx = {p: i for i, p in enumerate(left)}
        right_idx = {p: i for i, p in enumerate(right)}

        # Create cost matrix
        cost = np.ones((n, n), dtype=np.float64)
        for i, p1 in enumerate(left):
            for j, p2 in enumerate(right):
                if p1 in adj and p2 in adj[p1]:
                    cost[i, j] = 1.0 - adj[p1][p2]

        matrices.append(cost)

    return matrices


def iter_real_instances(
    or_library_paths: Iterable[str | Path] | None = None,
    suite_sparse_names: Iterable[str] | None = None,
    string_tax_id: str | None = None,
    string_subset_size: int = 1000,
    string_num_matrices: int = 10,
) -> Iterator[RealInstance]:
    """Yield real-world cost matrices using available sources."""

    if or_library_paths:
        for path in or_library_paths:
            for idx, matrix in enumerate(parse_or_library_assign(path)):
                yield RealInstance(
                    name=f"{Path(path).stem}_{idx}",
                    cost=matrix,
                    source="or_library",
                )

    if suite_sparse_names:
        for name in suite_sparse_names:
            yield RealInstance(
                name=name,
                cost=fetch_suite_sparse(name),
                source="suite_sparse",
            )
    
    if string_tax_id:
        string_data = fetch_string_protein_links(string_tax_id)
        matrices = create_string_cost_matrices(string_data, string_subset_size, string_num_matrices)
        for idx, matrix in enumerate(matrices):
            yield RealInstance(
                name=f"string_{string_tax_id}_{idx}",
                cost=matrix,
                source="string",
            )


def normalize_cost_matrix(cost: np.ndarray) -> np.ndarray:
    """Normalize matrix to [0,1] range (without touching infinities)."""

    finite_mask = np.isfinite(cost)
    if not finite_mask.any():
        return cost.astype(np.float64)
    finite_values = cost[finite_mask]
    mn = float(finite_values.min())
    mx = float(finite_values.max())
    if mx == mn:
        return np.where(finite_mask, 0.0, cost).astype(np.float64)
    scaled = (cost - mn) / (mx - mn)
    return np.where(finite_mask, scaled, cost).astype(np.float64)


__all__ = [
    "RealInstance",
    "parse_or_library_assign",
    "fetch_suite_sparse",
    "fetch_string_protein_links",
    "create_string_cost_matrices",
    "iter_real_instances",
    "normalize_cost_matrix",
]
