#!/usr/bin/env python3
"""
Prepare STRING Dataset for LAP Benchmarking

Downloads and processes the STRING protein-protein interaction database
to create LAP test instances based on real biological networks.

The STRING database provides confidence scores for protein-protein interactions.
We convert these into cost matrices for linear assignment problems.
"""

import argparse
import gzip
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import urllib.request
import numpy as np
import h5py
from tqdm import tqdm


def download_string_data(organism: str, version: str, output_dir: Path) -> Path:
    """Download STRING database file if not present.
    
    Args:
        organism: NCBI taxonomy ID (e.g., '9606' for human)
        version: STRING version (e.g., 'v12.0')
        output_dir: Directory to save raw data
        
    Returns:
        Path to downloaded .txt.gz file
    """
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{organism}.protein.links.{version}.txt.gz"
    filepath = raw_dir / filename
    
    if filepath.exists():
        print(f"✅ STRING data already exists: {filepath}")
        return filepath
    
    # Construct download URL
    url = f"https://stringdb-downloads.org/download/protein.links.{version}/{filename}"
    
    print(f"Downloading STRING data from: {url}")
    print("This may take several minutes...")
    
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"✅ Downloaded: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\nPlease download manually:")
        print(f"  URL: {url}")
        print(f"  Save to: {filepath}")
        raise


def extract_string_data(gz_path: Path) -> Path:
    """Extract .txt file from .gz archive.
    
    Args:
        gz_path: Path to .txt.gz file
        
    Returns:
        Path to extracted .txt file
    """
    txt_path = gz_path.with_suffix('')  # Remove .gz
    
    if txt_path.exists():
        print(f"✅ Extracted file already exists: {txt_path}")
        return txt_path
    
    print(f"Extracting {gz_path.name}...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(txt_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    print(f"✅ Extracted: {txt_path}")
    return txt_path


def load_string_network(
    txt_path: Path,
    score_threshold: int = 400,
    max_edges: Optional[int] = None
) -> Tuple[np.ndarray, List[str]]:
    """Load STRING network edges.
    
    Args:
        txt_path: Path to STRING .txt file
        score_threshold: Minimum combined score (0-999)
        max_edges: Maximum number of edges to load (for testing)
        
    Returns:
        edges: (n_edges, 3) array of [protein1_idx, protein2_idx, score]
        proteins: List of unique protein IDs
    """
    print(f"\nLoading STRING network from: {txt_path.name}")
    print(f"  Score threshold: {score_threshold}")
    
    # First pass: collect unique proteins
    protein_set = set()
    n_edges_total = 0
    
    with open(txt_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                protein1, protein2, score = parts[0], parts[1], int(parts[2])
                if score >= score_threshold:
                    protein_set.add(protein1)
                    protein_set.add(protein2)
                    n_edges_total += 1
    
    proteins = sorted(protein_set)
    protein_to_idx = {p: i for i, p in enumerate(proteins)}
    
    print(f"  Total proteins: {len(proteins)}")
    print(f"  Total edges (≥{score_threshold}): {n_edges_total}")
    
    # Second pass: load edges
    edges = []
    with open(txt_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                protein1, protein2, score = parts[0], parts[1], int(parts[2])
                if score >= score_threshold:
                    idx1 = protein_to_idx[protein1]
                    idx2 = protein_to_idx[protein2]
                    edges.append([idx1, idx2, score])
                    
                    if max_edges and len(edges) >= max_edges:
                        break
    
    edges = np.array(edges, dtype=np.int32)
    print(f"  Loaded edges: {len(edges)}")
    
    return edges, proteins


def sample_subnetwork(
    edges: np.ndarray,
    proteins: List[str],
    target_size: int,
    density: float,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, List[int]]:
    """Sample a subnetwork for a LAP instance.
    
    Strategy: Start with a random seed protein, grow by adding neighbors
    until we reach the target size.
    
    Args:
        edges: (n_edges, 3) array of [protein1_idx, protein2_idx, score]
        proteins: List of all protein IDs
        target_size: Desired number of proteins
        density: Target edge density (as fraction)
        seed: Random seed
        
    Returns:
        cost_matrix: (n×n) cost matrix for LAP
        protein_indices: List of protein indices in subnetwork
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_proteins = len(proteins)
    
    # Build adjacency structure
    neighbors = {i: [] for i in range(n_proteins)}
    edge_scores = {}
    
    for p1, p2, score in edges:
        neighbors[p1].append(p2)
        neighbors[p2].append(p1)
        edge_scores[(min(p1, p2), max(p1, p2))] = score
    
    # Start with random seed protein
    seed_protein = np.random.randint(0, n_proteins)
    selected = {seed_protein}
    frontier = set(neighbors[seed_protein])
    
    # Grow network by adding neighbors
    while len(selected) < target_size and frontier:
        # Add random protein from frontier
        new_protein = np.random.choice(list(frontier))
        selected.add(new_protein)
        
        # Update frontier
        frontier.remove(new_protein)
        for neighbor in neighbors[new_protein]:
            if neighbor not in selected:
                frontier.add(neighbor)
    
    # If we couldn't reach target size, add random proteins
    while len(selected) < target_size:
        remaining = set(range(n_proteins)) - selected
        if not remaining:
            break
        selected.add(np.random.choice(list(remaining)))
    
    protein_indices = sorted(selected)
    n = len(protein_indices)
    idx_map = {p: i for i, p in enumerate(protein_indices)}
    
    # Build cost matrix (lower score = higher cost)
    # Normalize scores to [0, 1] range where 0 = low cost (high confidence)
    cost_matrix = np.ones((n, n), dtype=np.float32)
    
    for p1, p2, score in edges:
        if p1 in selected and p2 in selected:
            i1, i2 = idx_map[p1], idx_map[p2]
            # Convert score (0-999) to cost (0-1): high score = low cost
            cost = 1.0 - (score / 1000.0)
            cost_matrix[i1, i2] = cost
            cost_matrix[i2, i1] = cost
    
    # Set diagonal to zero (self-assignment has no cost)
    np.fill_diagonal(cost_matrix, 0.0)
    
    return cost_matrix, protein_indices


def generate_lap_instances(
    edges: np.ndarray,
    proteins: List[str],
    num_instances: int,
    size_min: int,
    size_max: int,
    density_min: float,
    density_max: float,
    seed: int = 42
) -> List[Tuple[np.ndarray, List[int]]]:
    """Generate multiple LAP instances from STRING network.
    
    Args:
        edges: Network edges
        proteins: Protein IDs
        num_instances: Number of instances to generate
        size_min: Minimum problem size
        size_max: Maximum problem size
        density_min: Minimum edge density
        density_max: Maximum edge density
        seed: Random seed
        
    Returns:
        List of (cost_matrix, protein_indices) tuples
    """
    print(f"\nGenerating {num_instances} LAP instances...")
    print(f"  Size range: [{size_min}, {size_max}]")
    print(f"  Density range: [{density_min:.2%}, {density_max:.2%}]")
    
    np.random.seed(seed)
    instances = []
    
    for i in tqdm(range(num_instances)):
        # Random size and density
        size = np.random.randint(size_min, size_max + 1)
        density = np.random.uniform(density_min, density_max)
        
        # Sample subnetwork
        cost_matrix, protein_inds = sample_subnetwork(
            edges, proteins, size, density, seed=seed + i
        )
        
        instances.append((cost_matrix, protein_inds))
    
    print(f"✅ Generated {len(instances)} instances")
    return instances


def save_to_h5(
    instances: List[Tuple[np.ndarray, List[int]]],
    output_path: Path
):
    """Save LAP instances to H5 file.
    
    Args:
        instances: List of (cost_matrix, protein_indices)
        output_path: Path to output .h5 file
    """
    print(f"\nSaving to: {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        for i, (cost_matrix, protein_inds) in enumerate(instances):
            grp = f.create_group(f'instance_{i}')
            grp.create_dataset('cost_matrix', data=cost_matrix, compression='gzip')
            grp.create_dataset('protein_indices', data=np.array(protein_inds))
    
    print(f"✅ Saved {len(instances)} instances")
    
    # Print summary
    sizes = [c.shape[0] for c, _ in instances]
    densities = [(c > 0).sum() / (c.shape[0] ** 2) for c, _ in instances]
    
    print(f"\nDataset summary:")
    print(f"  Size: {min(sizes)} - {max(sizes)} (avg: {np.mean(sizes):.0f})")
    print(f"  Density: {min(densities):.2%} - {max(densities):.2%} (avg: {np.mean(densities):.2%})")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare STRING dataset for LAP benchmarking"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/string_dataset"),
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--organism",
        default="9606",
        help="NCBI taxonomy ID (9606=human, 10090=mouse, etc.)",
    )
    parser.add_argument(
        "--version",
        default="v12.0",
        help="STRING database version",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=100,
        help="Number of LAP instances to generate",
    )
    parser.add_argument(
        "--size-min",
        type=int,
        default=512,
        help="Minimum problem size",
    )
    parser.add_argument(
        "--size-max",
        type=int,
        default=2048,
        help="Maximum problem size",
    )
    parser.add_argument(
        "--density-min",
        type=float,
        default=0.05,
        help="Minimum edge density",
    )
    parser.add_argument(
        "--density-max",
        type=float,
        default=0.20,
        help="Maximum edge density",
    )
    parser.add_argument(
        "--score-threshold",
        type=int,
        default=400,
        help="Minimum STRING confidence score (0-999)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("STRING DATASET PREPARATION")
    print("="*80)
    
    # Download/locate STRING data
    try:
        gz_path = download_string_data(args.organism, args.version, args.output_dir)
    except Exception as e:
        print(f"\n❌ Failed to download STRING data: {e}")
        print("\nPlease download manually and place at:")
        print(f"  {args.output_dir}/raw/{args.organism}.protein.links.{args.version}.txt.gz")
        return
    
    # Extract if needed
    txt_path = extract_string_data(gz_path)
    
    # Load network
    edges, proteins = load_string_network(
        txt_path,
        score_threshold=args.score_threshold
    )
    
    # Generate LAP instances
    instances = generate_lap_instances(
        edges,
        proteins,
        num_instances=args.num_instances,
        size_min=args.size_min,
        size_max=args.size_max,
        density_min=args.density_min,
        density_max=args.density_max,
        seed=args.seed,
    )
    
    # Save to H5
    output_path = args.output_dir / "processed" / "test.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_to_h5(instances, output_path)
    
    print("\n" + "="*80)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("="*80)
    print(f"\nTest file ready: {output_path}")
    print("\nNext steps:")
    print("  1. Run benchmarks: python scripts/benchmark_string.py")
    print("  2. Check results in: logs/string_benchmark/")


if __name__ == "__main__":
    main()
