# LAP Benchmarking Suite - Logging System

## Overview

The LAP Benchmarking Suite now includes a comprehensive logging system that tracks all experiments, results, and performance metrics over time. This enables reproducible research and easy analysis of benchmark runs.

## Quick Start

### Basic Usage

```bash
# Run a logged benchmark
python scripts/logged_benchmark.py --name "my_experiment" --sizes 100 200 500

# List all previous experiments
python scripts/logged_benchmark.py --list

# View summary of a specific experiment
python scripts/logged_benchmark.py --summary experiment_id_20250922_171443
```

### Advanced Options

```bash
# Custom experiment with specific settings
python scripts/logged_benchmark.py \
    --name "sparse_analysis" \
    --sizes 50 100 200 \
    --trials 10 \
    --no-advanced

# Skip sparse solvers for faster runs
python scripts/logged_benchmark.py \
    --name "quick_test" \
    --sizes 100 \
    --no-sparse \
    --trials 3
```

## Logging Structure

The logging system creates organized directories in `logs/`:

```
logs/
├── performance/          # CSV files with timing data
├── experiments/          # JSON files with complete metadata
├── detailed/            # Detailed text logs with full output
└── summaries/           # Generated summary reports
```

## Log File Formats

### CSV Performance Logs
Located in `logs/performance/experiment_id.csv`

Columns:
- `timestamp`: When the test was run
- `experiment_id`: Unique identifier for the experiment
- `dataset`: Problem identifier (e.g., "uniform_100x100")
- `problem_size`: Matrix dimension (n)
- `problem_type`: Type of cost matrix (uniform, sparse, etc.)
- `solver_name`: Which solver was used
- `time_ms`: Execution time in milliseconds
- `cost`: Optimal assignment cost found
- `status`: success/failed/error
- `notes`: Additional information

### JSON Experiment Files
Located in `logs/experiments/experiment_id.json`

Contains complete experiment metadata:
- Environment information (Python version, hardware, etc.)
- All solver results with detailed timing statistics
- Problem configurations and parameters
- Comparison data between solvers

### Detailed Text Logs
Located in `logs/detailed/experiment_id.log`

Full console output from the benchmark run including:
- Environment summary
- Progress indicators
- Speedup calculations
- Validation results

## Analysis Examples

### Loading Experiment Data

```python
from solvers.logging_system import load_experiment
import pandas as pd

# Load a specific experiment
exp_data = load_experiment("test_logging_20250922_171443")

# Load CSV data for analysis
df = pd.read_csv("logs/performance/test_logging_20250922_171443.csv")

# Analyze speedups
scipy_times = df[df['solver_name'] == 'scipy']['time_ms']
seeded_times = df[df['solver_name'] == 'seeded']['time_ms']
speedups = scipy_times.values / seeded_times.values
print(f"Average speedup: {speedups.mean():.2f}x")
```

### Comparing Multiple Experiments

```python
from solvers.logging_system import list_experiments
import pandas as pd

# Get all experiments
experiments = list_experiments()

# Load and combine data
all_data = []
for exp_id in experiments:
    df = pd.read_csv(f"logs/performance/{exp_id}.csv")
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)

# Analyze trends over time
print(combined_df.groupby(['solver_name', 'problem_type'])['time_ms'].mean())
```

## Experiment Naming

Good experiment names help organize your research:

- `baseline_v1` - Initial baseline measurements
- `optimization_test` - Testing new optimization
- `scaling_analysis` - Performance scaling study
- `sparse_comparison` - Sparse vs dense solver comparison
- `production_validation` - Final validation before deployment

## Best Practices

1. **Use Descriptive Names**: Choose experiment names that clearly indicate the purpose
2. **Document Changes**: Include notes about what changed between experiments
3. **Regular Cleanup**: Archive old experiments periodically to keep logs manageable
4. **Consistent Settings**: Use consistent problem sizes and trials for comparable results
5. **Environment Tracking**: The system automatically tracks Python/NumPy versions for reproducibility

## Integration with Existing Scripts

All existing benchmark scripts can be enhanced with logging:

```python
from solvers.logging_system import BenchmarkLogger

# Create logger
logger = BenchmarkLogger(experiment_name="my_analysis")

# Log individual results
logger.log_result(
    dataset="uniform_100x100",
    problem_size=100,
    problem_type="uniform", 
    solver_name="seeded",
    time_seconds=0.05,
    cost=1.234,
    status="success"
)

# Save experiment
logger.save_experiment()
summary = logger.generate_summary()
print(summary)
```

## Automated Analysis

The logging system supports automated performance tracking:

```python
def analyze_latest_experiment():
    """Analyze the most recent experiment."""
    experiments = list_experiments()
    if not experiments:
        return "No experiments found"
    
    latest = experiments[-1]
    exp_data = load_experiment(latest)
    
    # Extract key metrics
    total_tests = len(exp_data['results'])
    success_rate = sum(1 for r in exp_data['results'] if r['status'] == 'success') / total_tests
    
    return f"Latest experiment {latest}: {total_tests} tests, {success_rate:.1%} success rate"
```

This comprehensive logging system enables reproducible research and makes it easy to track the evolution of your LAP solving algorithms over time.