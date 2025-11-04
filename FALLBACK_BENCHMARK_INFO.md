# Fallback Threshold Benchmark - Job Information

## Job Status

**Job ID**: 1441  
**Status**: RUNNING  
**Node**: ece-igum2  
**Submitted**: October 20, 2025

## Test Configuration

### Versions Being Tested

1. **lapjv_seeded_corrected.cpp** - Counts tight edges BEFORE row tightening
2. **lapjv_seeded_original.cpp** - Counts tight edges AFTER row tightening

### Fallback Thresholds

Testing each version with 5 different thresholds:
- 1.1n
- 1.2n
- 1.3n
- 1.4n
- 1.5n

**Total Tests**: 10 (2 versions × 5 thresholds)

### Test Parameters

- **Model**: progressive_clean_tie_best.pt
- **Problem Size**: 2048×2048
- **Instances per Type**: 10
- **Problem Types**: uniform, sparse
- **Total Problem Instances**: 20 per test (10 uniform + 10 sparse)

## What the Script Does

1. For each version (corrected, original):
   - For each threshold (1.1n through 1.5n):
     - Copy the version file to `lapjv_seeded.cpp`
     - Update the fallback threshold in the code
     - Rebuild the LAP module
     - Run benchmark on uniform and sparse problems
     - Save results to file

2. Generate summary report comparing all configurations
3. Create quick comparison table

## Monitoring the Job

### Check Job Status
```bash
squeue -j 1441
```

### View Live Output
```bash
tail -f /home/projects/nssl-prj10106/logs/fallback_threshold_benchmark_1441.out
```

### View Errors (if any)
```bash
tail -f /home/projects/nssl-prj10106/logs/fallback_threshold_benchmark_1441.err
```

### Check Intermediate Results
```bash
ls -lh /home/projects/nssl-prj10106/fallback_threshold_results/
```

## Expected Runtime

- Each rebuild: ~30 seconds
- Each benchmark run: ~2-3 minutes (20 instances)
- **Total estimated time**: ~30-40 minutes

## Output Files

All results will be saved to:
```
/home/projects/nssl-prj10106/fallback_threshold_results/
```

### Result Files

1. **Individual test results**:
   - `corrected_threshold_1.1n_<timestamp>.txt`
   - `corrected_threshold_1.2n_<timestamp>.txt`
   - ... (10 files total)

2. **Summary reports**:
   - `summary_<timestamp>.md` - Detailed markdown report
   - `quick_comparison_<timestamp>.txt` - Quick text comparison

## What We're Looking For

### Key Metrics

1. **SciPy Speedup**: How much faster than SciPy's linear_sum_assignment
2. **LAP Speedup**: How much faster than unseeded LAP (cold-start)
3. **MAE**: Mean Absolute Error of dual potential predictions

### Success Criteria

- Higher speedup values are better
- Consistent speedup across both problem types is ideal
- Lower MAE indicates better GNN predictions

### Questions to Answer

1. **Which version is better?**
   - CORRECTED (measures seed quality)
   - ORIGINAL (measures post-tightening state)

2. **What's the optimal threshold?**
   - Too low (1.1n): Falls back too often, misses warm-start benefits
   - Too high (1.5n): Tries warm-start on poor seeds, wastes time
   - Sweet spot: Maximizes speedup by falling back only when needed

3. **Does the counting order matter?**
   - Does counting BEFORE vs AFTER row tightening affect performance?
   - Theory: BEFORE should be better (measures true seed quality)
   - Practice: Need data to confirm!

## Commands for Analysis

After the job completes:

### View Summary Report
```bash
cat /home/projects/nssl-prj10106/fallback_threshold_results/summary_*.md
```

### View Quick Comparison
```bash
cat /home/projects/nssl-prj10106/fallback_threshold_results/quick_comparison_*.txt
```

### Find Best Configuration
```bash
# Look for highest speedup values
grep -h "SciPy.*LAP" /home/projects/nssl-prj10106/fallback_threshold_results/*.txt | sort -k3 -rn
```

## Understanding the Results

### Result Format

Each benchmark produces output like:
```
Type            Count    SciPy        LAP          MAE         
------------------------------------------------------------
sparse          10        1.45x [1.30-1.68]  1.37x [0.68-1.87]   0.0971
uniform         10        1.89x [1.50-2.14]  1.31x [0.71-2.86]   0.0742
```

### Interpretation

- **SciPy 1.45x**: Seeded solver is 1.45× faster than SciPy
- **LAP 1.37x**: Seeded solver is 1.37× faster than unseeded LAP
- **[1.30-1.68]**: Range across 10 instances (min-max speedup)
- **MAE 0.0971**: Average dual potential prediction error

### What Makes a Good Configuration?

✅ **Ideal**:
- SciPy speedup > 1.5×
- LAP speedup > 1.3×
- Consistent across both problem types
- Low variation (tight min-max range)

⚠️ **Acceptable**:
- SciPy speedup > 1.0× (at least not slower)
- LAP speedup > 1.0×

❌ **Poor**:
- Speedup < 1.0× (slower than baseline)
- High variation (wide min-max range)
- One problem type good, other terrible

## Next Steps After Results

1. **Identify winner**: Best version × threshold combination
2. **Update code**: Set the optimal configuration as default
3. **Document findings**: Understand why it works
4. **Test on other problem types**: Validate on block, tie, etc.
