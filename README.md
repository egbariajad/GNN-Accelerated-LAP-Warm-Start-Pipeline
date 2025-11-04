# GNN-Accelerated LAP Warm-Start Pipeline

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[![arXiv](https://img.shields.io/badge/arXiv-2201.02057-b31b1b)](https://arxiv.org/abs/2201.02057)

[![arXiv](https://img.shields.io/badge/arXiv-2505.24067-b31b1b)](https://arxiv.org/abs/2505.24067)

End-to-end system for accelerating the Linear Assignment Problem (LAP) by integrating learned dual potentials with a custom seeded Jonker-Volgenant (JV) solver. The primary pipeline employs a lightweight **OneGNN** model to predict row duals (û), reconstructs column duals via the min-trick (v̂ = min(C - û)), and completes the solve using a warm-started C++ extension. Benchmarks on synthetic and real-world instances demonstrate 1.3x–2x average wall-clock speedups (up to 5x on large problems) while maintaining exact optimality. This builds on concepts from GLAN (arXiv:2201.02057) and Primal-Dual NAR (arXiv:2505.24067), emphasizing scalability for cost matrices with n up to 16K+.

```

Cost matrix C  →  Feature engineering  →  OneGNN (û)  →  v̂ = min(C - û)

                                    ↘  lapjv_seeded(C, û, v̂)  →  Optimal assignment

```

## Highlights

- **OneGNN Row-Only Model** (`gnn/one_gnn.py`): Residual MLP with sparse top-k refinement (k=16 default) for O(n) scaling on large n (>4K). Predicts row duals using 21D features from `compute_row_features` (stats like min/max/mean/std, entropy, competition, positional encodings).

- **Seeded LAP Solver Extension** (`LAP/_lapjv_cpp/lapjv_seeded.cpp`): Custom JV variant with feasibility projection, greedy matching on tight edges, micro-augmenting row reduction (ARR), quality checks (fallback to full JV if tight edge density <1.2n), and partial augmentation for unmatched rows.

- **Dual Utilities** (`solvers/advanced_dual.py`, `solvers/dual_computation.py`): Feasibility projection, reduced-cost computation, difference-constraints reconstruction for oracle duals, and validation tools.

- **Dataset Generation** (`data/generate_dataset.py`): Supports synthetic families (uniform, sparse, metric, clustered) and real sources (OR-Library, SuiteSparse, STRING); outputs HDF5 with C, u/v, assignments, and metadata.

- **Benchmark Suite** (`scripts/gnn_benchmark.py`, `scripts/comprehensive_gnn_benchmark.py`): Compares SciPy, unseeded LAP, and full pipeline; generates speedup distributions, means by size/dataset, and visualizations. Rigorous timing with repeats and outlier handling.

## Repository Layout

- `data/` – Synthetic generators, real-data processors, HDF5 pipelines (`generate_dataset.py`, `processors.py`, `splits.py`).

- `gnn/` – OneGNN and DualGNN models, feature engineering, training loops, evaluation (`one_gnn.py`, `features.py`, `train_one_gnn.py`, `train_progressive_clean.py`, `eval.py`).

- `solvers/` – Unified solver wrappers, seeded LAP integration, warm-start helpers, timing, verification, and seeding baselines (`lap_solver.py`, `advanced_dual.py`, `verification.py`).

- `scripts/` – Reproducible benchmarks, visualization utilities, SLURM drivers, and analysis notebooks (`gnn_benchmark.py`, `comprehensive_gnn_benchmark.py`).

- `LAP/` – Fork of the `lap` package with `lapjv_seeded` implementation and build artifacts.

- `docs/` – Architecture overviews, pipeline flows, quick references, and guides (`ARCHITECTURE_OVERVIEW.md`, `PIPELINE_FLOW.md`, `CLEAN_DATASET_GUIDE.md`).

## Installation

1. **Python Environment**  

   Recommended Python ≥3.9 with virtualenv/conda:  

   ```bash

   conda create -n gnn-lap python=3.10

   conda activate gnn-lap

   ```

2. **Core Dependencies**  

   ```bash

   pip install numpy scipy h5py torch matplotlib

   ```

   Optional: `ssgetpy` (for SuiteSparse), `networkx`/`pandas` (for analysis), `biopython`/`pubchempy` (for STRING processing).

3. **Custom LAP Build**  

   Install the forked `lap` package to enable `lap.lapjv_seeded`:  

   ```bash

   pip install -e LAP

   ```

   This compiles the C++ extensions (requires a C++ compiler like g++). For older wheels, it may fall back to `_seeded_jv.lapjv_seeded`.

4. **Project Setup (Optional)**  

   Add the repo root to `PYTHONPATH` or install editable:  

   ```bash

   pip install -e .

   ```

GPU acceleration is auto-detected via PyTorch; CPU fallback is fully supported.

## Typical Workflow

### 1. Generate Datasets

Create stratified HDF5 datasets:  

```bash

python data/generate_dataset.py \

  --output-dir data/generated/mid2048_full \

  --sizes 2048 \

  --instances-per-family 1200 \

  --families uniform sparse metric clustered \

  --split-ratios train=0.8 val=0.1 test=0.1 \

  --seed 42 \

  --summary

```

Verify with `verify_clean_datasets.py`. See `CLEAN_DATASET_GUIDE.md` for large-scale generation.

### 2. Train Models

For OneGNN:  

```bash

python gnn/train_one_gnn.py \

  --train data/generated/mid2048_full/train.h5 \

  --val data/generated/mid2048_full/val.h5 \

  --epochs 100 \

  --batch-size 32 \

  --lr 0.001 \

  --hidden 192 \

  --layers 4 \

  --dropout 0.1 \

  --output checkpoints/one_gnn_mid2048.pt

```

Uses L1 loss on duals, primal gap metrics, AdamW optimizer, and WarmupCosineScheduler. For DualGNN or progressive size-bucket training, use `train_progressive_clean.py` (see `PROGRESSIVE_CLEAN_QUICK.md`).

### 3. Evaluate Dual Quality

```bash

python gnn/eval.py \

  --split data/generated/mid2048_full/test.h5 \

  --checkpoint checkpoints/one_gnn_mid2048.pt \

  --device cuda \

  --limit 100

```

Outputs feasibility rates, dual/primal gaps, and warmup assist metrics. Disable projection with `--no-project` for raw analysis.

### 4. Run Benchmarks

```bash

python scripts/gnn_benchmark.py \

  --models checkpoints/one_gnn_mid2048.pt checkpoints/dual_gnn_sparse.pt \

  --data-root data/generated \

  --datasets mid2048_full \

  --sizes 512 1024 2048 4096 \

  --max-instances 10 \

  --device auto \

  --repeats 20

```

Generates speedup reports vs. SciPy/unseeded LAP. Use `comprehensive_gnn_benchmark.py` for plots and filtered datasets.

For automated workflows, see `run_clean_workflow.sh` and SLURM templates (`run_*.slurm`).

## Using the Solver Interfaces

Solve LAP instances with seeding:  

```python

import numpy as np

import torch

from gnn.one_gnn import OneGNN

from gnn.features import compute_row_features

from solvers import SciPySolver, LAPSolver, SeededLAPSolver

# Baseline solvers

C = np.random.uniform(0, 1, (1024, 1024))

rows_s, cols_s, cost_s = SciPySolver().solve(C)

rows_l, cols_l, cost_l = LAPSolver().solve(C)

# Load OneGNN and predict duals

checkpoint = torch.load('checkpoints/one_gnn_mid2048.pt')

model = OneGNN(in_dim=checkpoint['row_feat_dim'], hidden=192, layers=4, dropout=0.1, topk=16)

model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

row_feat = compute_row_features(C)  # GPU: compute_row_features_torch

with torch.no_grad():

    outputs = model(torch.from_numpy(row_feat).float().unsqueeze(0), cost=torch.from_numpy(C).float())

    u_pred = outputs['u'].squeeze(0).numpy()

v_pred = np.min(C - u_pred[:, None], axis=0)  # Min-trick

rows_se, cols_se, cost_se = SeededLAPSolver().solve(C, u_pred, v_pred)

```

Additional tools: `time_solver_rigorous` for benchmarks, `project_feasible` for dual prep, `verify_solver_correctness` for validation.

## GNN Models at a Glance

- **OneGNN** (`gnn/one_gnn.py`): O(n) row encoder with residual blocks (Linear + GELU + Dropout + LayerNorm), sparse top-k refinement for global context, and mean-centered outputs. Pairs with min-trick for v.

- **DualGNN** (`gnn/dual_gnn.py`): GATv2-based graph attention over full edge features for joint u/v prediction—higher accuracy but O(n²) cost.

- **Features** (`gnn/features.py`): NumPy/Torch implementations; 21D rows (min/max/mean/std/mad/entropy/second_best_gap/competition/k_mean/k_std/difficulty/near_best/is_col_best + positional encodings).

## Data Sources & Families

- **Synthetic** (`solvers/generators.py`): Uniform, sparse, metric (geometric), clustered, low-rank + noise, adversarial (anti-diagonal), identity-like, hard random.

- **Real** (`data/processors.py`): OR-Library, SuiteSparse (via `ssgetpy`), STRING protein graphs as bipartite costs.

- **HDF5 Schema**: Per-instance keys for flattened C, u/v, rows/cols, n, family, noise level, tags.

## Results

- **Uniform Datasets**: Mean 1.84x vs. SciPy (median 1.95x); 1.62x vs. unseeded LAP.

- **Sparse Datasets**: Mean 1.37x vs. SciPy (median 1.36x).

- Speedups grow with size (1.90x at n=16K). See `ARCHITECTURE_SUMMARY.md` for charts.

## Documentation & Guides

- `ARCHITECTURE_OVERVIEW.md` & `ARCHITECTURE_SUMMARY.md`: Theory, designs, comparisons.

- `PIPELINE_FLOW.md`: Visual architecture.

- `CLEAN_DATASET_GUIDE.md`, `CLEAN_WORKFLOW_SUMMARY.md`: Dataset hygiene, SLURM automation.

- `FALLBACK_BENCHMARK_INFO.md`: Fallback analysis.

- Logs in `logs/` for reproducibility.

## Testing & Validation

- `verify_clean_datasets.py`: HDF5 integrity checks.

- `test_dual_noise_flag.py`, `test_cs_loss.py`, `test_string.py`: Regression tests.

- `solvers/verification.py`: Feasibility and assignment validators.

