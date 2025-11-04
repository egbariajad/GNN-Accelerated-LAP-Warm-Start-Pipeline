# Fallback Threshold Benchmark - Quick Reference

## Job Information
- **Job ID**: 1441
- **Status**: Running on ece-igum2
- **Started**: October 20, 2025 at 13:54

## What's Being Tested

Comparing two versions of `lapjv_seeded`:

### Version 1: CORRECTED (Count Before Row Tightening)
- Counts tight edges using **original GNN predictions**
- Fallback decision based on **seed quality**
- Theory: Better at detecting poor predictions early

### Version 2: ORIGINAL (Count After Row Tightening)  
- Counts tight edges **after forcing one zero per row**
- Fallback decision based on **post-processing state**
- Theory: Row tightening masks poor seeds (fallback rarely triggers)

## Thresholds Being Tested

Each version tested with 5 fallback thresholds:
- **1.1n**: Aggressive fallback (triggers easily)
- **1.2n**: Conservative fallback (current default)
- **1.3n**: Moderate
- **1.4n**: Permissive
- **1.5n**: Very permissive (rarely falls back)

## Test Configuration

- **Model**: progressive_clean_tie_best.pt (OneGNN, 192 hidden, 4 layers)
- **Problems**: 10 uniform + 10 sparse (2048×2048)
- **Metrics**: SciPy speedup, LAP speedup, MAE
- **Total tests**: 10 (2 versions × 5 thresholds)

## Monitoring Commands

```bash
# Check job status
squeue -j 1441

# Monitor progress
./monitor_fallback_benchmark.sh

# View live output
tail -f logs/fallback_threshold_benchmark_1441.out

# Check results so far
ls -lh fallback_threshold_results/
```

## Expected Timeline

- ✅ **1:54 PM**: Job started
- 🔄 **Now**: Testing in progress (5 CORRECTED tests, then 5 ORIGINAL tests)
- 📊 **~2:30 PM**: Expected completion (30-40 minutes total)

## What to Look For in Results

### Best Configuration Will Have:
1. **Highest average speedup** (both SciPy and LAP)
2. **Consistent performance** (small min-max range)
3. **Works well on both** uniform AND sparse

### Key Questions:
1. Does counting **before** vs **after** row tightening matter?
2. What's the optimal threshold value?
3. Can we improve on current performance (1.45x-1.89x)?

## Results Location

All output saved to:
```
/home/projects/nssl-prj10106/fallback_threshold_results/
```

Files:
- `corrected_threshold_*.txt` - Individual test outputs
- `original_threshold_*.txt` - Individual test outputs  
- `summary_*.md` - Comprehensive comparison report
- `quick_comparison_*.txt` - Quick text table

## Current Baseline Performance

From previous tests with ORIGINAL version (1.2n threshold):
- **Sparse**: SciPy 1.45x, LAP 1.37x
- **Uniform**: SciPy 1.89x, LAP 1.31x

Goal: Find configuration that beats these numbers!

## After Completion

1. Run monitor script to see final results
2. Read the summary report
3. Identify best version + threshold
4. Update default configuration
5. Test on other problem types (block, tie, etc.)
