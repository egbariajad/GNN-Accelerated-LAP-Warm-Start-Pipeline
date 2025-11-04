# Usage Guide: Training and Dataset Generation

This guide shows you how to generate datasets, train GNN models, and run benchmarks using this repository.

---

## 🗂️ Dataset Generation

### Basic Dataset Generation

Generate synthetic LAP datasets with the `data/generate_dataset.py` script:

```bash
python data/generate_dataset.py \
  --output-dir data/my_dataset \
  --sizes 512 1024 2048 \
  --instances-per-family 1000 \
  --families uniform sparse metric clustered \
  --split-ratios train=0.7 val=0.15 test=0.15 \
  --seed 42
```

**Parameters:**
- `--output-dir`: Where to save the generated HDF5 files
- `--sizes`: Matrix dimensions to generate (e.g., 512, 1024, 2048, 4096)
- `--instances-per-family`: Number of instances per problem family
- `--families`: Problem types to generate:
  - `uniform`: Uniform random costs
  - `sparse`: Sparse cost matrices
  - `metric`: Geometric/metric costs
  - `clustered`: Clustered structure
  - `low_rank`: Low-rank structure
  - `adversarial`: Hard instances
- `--split-ratios`: Train/validation/test split proportions
- `--seed`: Random seed for reproducibility

**Output:**
Creates `train.h5`, `val.h5`, and `test.h5` files in the output directory.

### Dataset Structure

Each HDF5 file contains instances with:
- `cost_matrix`: Flattened cost matrix C
- `optimal_u`, `optimal_v`: Oracle dual potentials
- `row_assignment`, `col_assignment`: Optimal assignment
- `n`: Matrix dimension
- `family`: Problem type
- Metadata (tags, noise levels, etc.)

### Generating Large-Scale Datasets

For production use:

```bash
# Small problems (512-1024) - fast training
python data/generate_dataset.py \
  --output-dir data/small \
  --sizes 512 1024 \
  --instances-per-family 2000 \
  --families uniform sparse metric clustered

# Medium problems (2048-3072) - balanced
python data/generate_dataset.py \
  --output-dir data/medium \
  --sizes 2048 3072 \
  --instances-per-family 500 \
  --families uniform sparse metric

# Large problems (4096+) - for evaluation
python data/generate_dataset.py \
  --output-dir data/large \
  --sizes 4096 8192 \
  --instances-per-family 100 \
  --families uniform sparse
```

---

## 🤖 Model Training

### Training OneGNN (Recommended)

OneGNN is the lightweight, scalable model that predicts row duals only:

```bash
python gnn/train_one_gnn.py \
  --train data/my_dataset/train.h5 \
  --val data/my_dataset/val.h5 \
  --output checkpoints/my_model.pt \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.001 \
  --hidden 192 \
  --layers 4 \
  --dropout 0.1 \
  --topk 16 \
  --device cuda
```

**Key Parameters:**
- `--train`, `--val`: Paths to HDF5 dataset files
- `--output`: Where to save the trained model checkpoint
- `--epochs`: Number of training epochs (50-200 typical)
- `--batch-size`: Batch size (adjust based on GPU memory)
  - 512×512: batch_size=32-64
  - 2048×2048: batch_size=8-16
  - 4096×4096: batch_size=2-4
- `--lr`: Learning rate (0.0005-0.001 typical)
- `--hidden`: Hidden dimension (128-256 typical)
- `--layers`: Number of MLP layers (3-5 typical)
- `--dropout`: Dropout rate (0.1-0.2 typical)
- `--topk`: Number of top-k elements for global context (8-32)
- `--device`: `cuda` for GPU, `cpu` for CPU

**Training Output:**
- Saves best checkpoint based on validation loss
- Logs training/validation metrics each epoch
- Final checkpoint includes model weights and configuration

### Training DualGNN (Alternative)

DualGNN predicts both row and column duals using graph attention:

```bash
python gnn/train.py \
  --train data/my_dataset/train.h5 \
  --val data/my_dataset/val.h5 \
  --output checkpoints/dual_gnn.pt \
  --epochs 100 \
  --batch-size 16 \
  --lr 0.001 \
  --hidden 128 \
  --layers 3 \
  --device cuda
```

**Note:** DualGNN has O(n²) complexity and is slower but potentially more accurate.

### Multi-Size Progressive Training

For training on datasets with multiple matrix sizes:

```bash
python gnn/train_progressive_clean.py \
  --config progressive_clean_config.yaml \
  --output checkpoints/progressive_model.pt \
  --device cuda
```

Edit `progressive_clean_config.yaml` to specify:
- Multiple dataset paths with different sizes
- Curriculum schedule (when to introduce each size)
- Size-specific batch sizes and sampling weights

**Example config structure:**
```yaml
training:
  datasets:
    - name: small_512
      path: data/small/train.h5
      size: 512
      batch_size: 32
      weight: 1.0
      curriculum_start_epoch: 1
    
    - name: medium_2048
      path: data/medium/train.h5
      size: 2048
      batch_size: 8
      weight: 1.0
      curriculum_start_epoch: 10

validation:
  datasets:
    - name: small_512_val
      path: data/small/val.h5
    - name: medium_2048_val
      path: data/medium/val.h5
```

---

## 📊 Model Evaluation

### Basic Evaluation

Evaluate dual prediction quality:

```bash
python gnn/eval.py \
  --split data/my_dataset/test.h5 \
  --checkpoint checkpoints/my_model.pt \
  --device cuda \
  --limit 100
```

**Metrics reported:**
- Dual feasibility rate
- Dual gap (L1 error vs optimal)
- Primal gap (solution quality)
- Warmup assist metrics

### Benchmark Against Baselines

Run comprehensive benchmarks comparing GNN+seeded solver vs baselines:

```bash
python scripts/gnn_benchmark.py \
  --models checkpoints/my_model.pt \
  --data-root data/my_dataset \
  --dataset test.h5 \
  --device cuda \
  --repeats 20 \
  --max-instances 50
```

**Baselines compared:**
- SciPy `linear_sum_assignment` (baseline)
- Unseeded LAP solver
- GNN + Seeded LAP solver (ours)

**Output:**
- Speedup statistics (mean, median, percentiles)
- Timing breakdown by matrix size
- Solution quality verification

### Comprehensive Benchmark with Visualizations

```bash
python scripts/comprehensive_gnn_benchmark.py \
  --models checkpoints/my_model.pt \
  --data-root data \
  --datasets small/test.h5 medium/test.h5 large/test.h5 \
  --device cuda \
  --output-dir results/benchmark_$(date +%Y%m%d)
```

**Generates:**
- Speedup distribution plots
- Performance by matrix size
- Performance by problem family
- CSV files with detailed results

---

## 🔍 Using Trained Models Programmatically

### Basic Usage

```python
import torch
import numpy as np
from gnn.one_gnn import OneGNN
from gnn.features import compute_row_features
from solvers import SeededLAPSolver

# Load trained model
checkpoint = torch.load('checkpoints/my_model.pt')
model = OneGNN(
    in_dim=checkpoint['row_feat_dim'],
    hidden=checkpoint['hidden_dim'],
    layers=checkpoint['num_layers'],
    dropout=0.0,  # No dropout during inference
    topk=checkpoint.get('topk', 16)
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare cost matrix
C = np.random.uniform(0, 1, (1024, 1024))

# Compute features
row_features = compute_row_features(C)

# Predict duals
with torch.no_grad():
    row_feat_tensor = torch.from_numpy(row_features).float().unsqueeze(0)
    cost_tensor = torch.from_numpy(C).float().unsqueeze(0)
    outputs = model(row_feat_tensor, cost=cost_tensor)
    u_pred = outputs['u'].squeeze(0).numpy()

# Reconstruct column duals using min-trick
v_pred = np.min(C - u_pred[:, None], axis=0)

# Solve with warm-start
solver = SeededLAPSolver()
row_assignment, col_assignment, total_cost = solver.solve(C, u_pred, v_pred)

print(f"Optimal cost: {total_cost}")
print(f"Assignment: {row_assignment[:10]}...")  # First 10 rows
```

### Batch Inference

```python
import h5py

# Load test dataset
with h5py.File('data/my_dataset/test.h5', 'r') as f:
    costs = []
    for key in f.keys():
        n = f[key]['n'][()]
        C = f[key]['cost_matrix'][:].reshape(n, n)
        costs.append(C)

# Process in batches
solver = SeededLAPSolver()
results = []

for C in costs:
    row_features = compute_row_features(C)
    with torch.no_grad():
        u_pred = model.predict_duals(row_features, C)
    v_pred = np.min(C - u_pred[:, None], axis=0)
    row_assign, col_assign, cost = solver.solve(C, u_pred, v_pred)
    results.append({'assignment': row_assign, 'cost': cost})
```

---

## 💡 Tips and Best Practices

### Dataset Generation
- Start with small sizes (512-1024) for quick iteration
- Use `--instances-per-family 2000+` for production models
- Include diverse problem families for robustness
- Set `--seed` for reproducible experiments

### Training
- Start with smaller models (hidden=128, layers=3) and scale up
- Use learning rate warmup for stable training
- Monitor validation loss to avoid overfitting
- Save checkpoints frequently (automatic in training scripts)
- Use mixed-size training for models that generalize across scales

### Benchmarking
- Use `--repeats 20` for stable timing measurements
- Exclude first few runs (warmup) when reporting results
- Test on held-out problem families for true generalization
- Compare wall-clock time (not just iterations)

### GPU Memory
If you encounter OOM errors:
- Reduce `--batch-size`
- Reduce `--hidden` dimension
- Use gradient accumulation (modify training scripts)
- Train on smaller matrix sizes first

### CPU Fallback
If no GPU available:
- Use `--device cpu`
- Reduce batch size to 8-16
- Expect slower training (10-50x depending on model)
- Consider training on small sizes only

---

## 📁 Directory Structure After Setup

```
your_project/
├── data/
│   ├── my_dataset/
│   │   ├── train.h5
│   │   ├── val.h5
│   │   └── test.h5
│   ├── small/
│   ├── medium/
│   └── large/
├── checkpoints/
│   ├── my_model.pt
│   ├── dual_gnn.pt
│   └── progressive_model.pt
└── results/
    └── benchmark_20251104/
        ├── speedup_plot.png
        ├── results.csv
        └── summary.txt
```

---

## 🆘 Troubleshooting

### Import Errors
```bash
# Make sure LAP package is installed
pip install -e LAP

# Add project to PYTHONPATH
export PYTHONPATH=/path/to/GNN-Accelerated-LAP-Warm-Start-Pipeline:$PYTHONPATH
```

### HDF5 Dataset Issues
```python
# Check dataset contents
python -c "
import h5py
with h5py.File('data/my_dataset/train.h5', 'r') as f:
    print(f'Keys: {len(f.keys())}')
    key = list(f.keys())[0]
    print(f'Sample: n={f[key][\"n\"][()]}, family={f[key][\"family\"][()] }')
"
```

### Model Not Learning
- Check learning rate (try 0.0005 - 0.005 range)
- Verify dataset quality (check for NaN values)
- Ensure features are normalized properly
- Try simpler problem families first (uniform, sparse)
- Increase model capacity (more layers/hidden units)

### Slow Inference
- Use `--device cuda` if GPU available
- Reduce `--topk` for OneGNN (try 8 or 16)
- Consider OneGNN instead of DualGNN for large problems
- Batch inference when processing multiple instances

---

## 📚 Further Reading

- **ARCHITECTURE_OVERVIEW.md**: Detailed explanation of the pipeline design
- **PIPELINE_FLOW.md**: Visual architecture diagrams
- **README.md**: Full project documentation
- Paper references in README for theoretical background
