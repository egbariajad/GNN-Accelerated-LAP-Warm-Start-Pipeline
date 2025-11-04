#!/bin/bash
# Comprehensive benchmark comparing lapjv_seeded versions with different fallback thresholds
# Tests: corrected vs original with thresholds 1.1n, 1.2n, 1.3n, 1.4n, 1.5n

set -e

PROJECT_DIR="/home/projects/nssl-prj10106"
LAP_DIR="$PROJECT_DIR/LAP"
RESULTS_DIR="$PROJECT_DIR/fallback_threshold_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create results directory
mkdir -p "$RESULTS_DIR"

# Test configuration
MODELS="progressive_clean_tie_best.pt"
SIZES="2048"
INSTANCES="10"
TYPES="uniform sparse"
THRESHOLDS=(1.1 1.2 1.3 1.4 1.5)

echo "========================================================================"
echo "LAPJV SEEDED FALLBACK THRESHOLD BENCHMARK"
echo "========================================================================"
echo "Date: $(date)"
echo "Models: $MODELS"
echo "Sizes: $SIZES"
echo "Instances per type: $INSTANCES"
echo "Problem types: $TYPES"
echo "Thresholds: ${THRESHOLDS[@]}"
echo "========================================================================"
echo ""

# Function to update threshold in a file
update_threshold() {
    local file=$1
    local threshold=$2
    
    # Create temporary file with updated threshold
    sed "s/total_tight_edges < [0-9.]\+ \* n/total_tight_edges < $threshold * n/g" "$file" > "${file}.tmp"
    mv "${file}.tmp" "$file"
    
    echo "  ✓ Updated threshold to ${threshold}n in $(basename $file)"
}

# Function to run benchmark
run_benchmark() {
    local version_name=$1
    local version_file=$2
    local threshold=$3
    local output_file=$4
    
    echo "----------------------------------------"
    echo "Testing: $version_name with threshold ${threshold}n"
    echo "----------------------------------------"
    
    # Copy version to active file
    cp "$version_file" "$LAP_DIR/_lapjv_cpp/lapjv_seeded.cpp"
    
    # Update threshold
    update_threshold "$LAP_DIR/_lapjv_cpp/lapjv_seeded.cpp" "$threshold"
    
    # Rebuild
    echo "  Building..."
    cd "$LAP_DIR"
    python setup.py build_ext --inplace > /dev/null 2>&1
    
    # Run benchmark
    echo "  Running benchmark..."
    cd "$PROJECT_DIR"
    python scripts/gnn_benchmark_by_type.py \
        --models $MODELS \
        --sizes $SIZES \
        --instances-per-type $INSTANCES \
        --types $TYPES \
        > "$output_file" 2>&1
    
    # Extract key results
    echo "  Results:"
    grep -A 2 "Type.*Count.*SciPy.*LAP" "$output_file" | tail -3 || echo "  (parsing failed)"
    echo ""
}

# =============================================================================
# TEST CORRECTED VERSION (counts tight edges BEFORE row tightening)
# =============================================================================

echo ""
echo "========================================================================"
echo "TESTING: lapjv_seeded_corrected.cpp (Count BEFORE row tightening)"
echo "========================================================================"
echo ""

CORRECTED_FILE="$LAP_DIR/_lapjv_cpp/lapjv_seeded_corrected.cpp"

if [ ! -f "$CORRECTED_FILE" ]; then
    echo "ERROR: $CORRECTED_FILE not found!"
    exit 1
fi

for threshold in "${THRESHOLDS[@]}"; do
    output_file="$RESULTS_DIR/corrected_threshold_${threshold}n_${TIMESTAMP}.txt"
    run_benchmark "CORRECTED" "$CORRECTED_FILE" "$threshold" "$output_file"
done

# =============================================================================
# TEST ORIGINAL VERSION (counts tight edges AFTER row tightening)
# =============================================================================

echo ""
echo "========================================================================"
echo "TESTING: lapjv_seeded_original.cpp (Count AFTER row tightening)"
echo "========================================================================"
echo ""

ORIGINAL_FILE="$LAP_DIR/_lapjv_cpp/lapjv_seeded_original.cpp"

if [ ! -f "$ORIGINAL_FILE" ]; then
    echo "ERROR: $ORIGINAL_FILE not found!"
    exit 1
fi

for threshold in "${THRESHOLDS[@]}"; do
    output_file="$RESULTS_DIR/original_threshold_${threshold}n_${TIMESTAMP}.txt"
    run_benchmark "ORIGINAL" "$ORIGINAL_FILE" "$threshold" "$output_file"
done

# =============================================================================
# GENERATE SUMMARY REPORT
# =============================================================================

echo ""
echo "========================================================================"
echo "GENERATING SUMMARY REPORT"
echo "========================================================================"

SUMMARY_FILE="$RESULTS_DIR/summary_${TIMESTAMP}.md"

cat > "$SUMMARY_FILE" << 'EOF'
# LAPJV Seeded Fallback Threshold Benchmark Results

## Test Configuration

- **Models**: progressive_clean_tie_best.pt
- **Problem Size**: 2048x2048
- **Instances per Type**: 10
- **Problem Types**: uniform, sparse
- **Thresholds Tested**: 1.1n, 1.2n, 1.3n, 1.4n, 1.5n

## Results Summary

### CORRECTED Version (Count tight edges BEFORE row tightening)

EOF

echo "Extracting results for CORRECTED version..."

for threshold in "${THRESHOLDS[@]}"; do
    output_file="$RESULTS_DIR/corrected_threshold_${threshold}n_${TIMESTAMP}.txt"
    
    echo "#### Threshold: ${threshold}n" >> "$SUMMARY_FILE"
    echo '```' >> "$SUMMARY_FILE"
    
    # Extract performance table
    grep -A 5 "Type.*Count.*SciPy.*LAP" "$output_file" | head -8 >> "$SUMMARY_FILE" || echo "No data" >> "$SUMMARY_FILE"
    
    echo '```' >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
done

cat >> "$SUMMARY_FILE" << 'EOF'

### ORIGINAL Version (Count tight edges AFTER row tightening)

EOF

echo "Extracting results for ORIGINAL version..."

for threshold in "${THRESHOLDS[@]}"; do
    output_file="$RESULTS_DIR/original_threshold_${threshold}n_${TIMESTAMP}.txt"
    
    echo "#### Threshold: ${threshold}n" >> "$SUMMARY_FILE"
    echo '```' >> "$SUMMARY_FILE"
    
    # Extract performance table
    grep -A 5 "Type.*Count.*SciPy.*LAP" "$output_file" | head -8 >> "$SUMMARY_FILE" || echo "No data" >> "$SUMMARY_FILE"
    
    echo '```' >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
done

# Add comparison section
cat >> "$SUMMARY_FILE" << 'EOF'

## Analysis

### Key Findings

1. **Best Threshold for CORRECTED Version**: 
   - Compare SciPy and LAP speedup values above
   - Look for highest consistent speedup

2. **Best Threshold for ORIGINAL Version**:
   - Compare SciPy and LAP speedup values above
   - Look for highest consistent speedup

3. **CORRECTED vs ORIGINAL**:
   - CORRECTED measures seed quality before modification
   - ORIGINAL measures post-tightening state (always shows high density)

### Recommendations

Based on the results above:

- **Winner**: [To be determined from data]
- **Optimal Threshold**: [To be determined from data]

EOF

echo ""
echo "========================================================================"
echo "BENCHMARK COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Summary report: $SUMMARY_FILE"
echo ""
echo "To view summary:"
echo "  cat $SUMMARY_FILE"
echo ""
echo "To view detailed results:"
echo "  ls -lh $RESULTS_DIR/"
echo ""

# Create quick comparison table
echo "========================================================================"
echo "QUICK COMPARISON TABLE"
echo "========================================================================"
echo ""

COMPARISON_FILE="$RESULTS_DIR/quick_comparison_${TIMESTAMP}.txt"

cat > "$COMPARISON_FILE" << EOF
CORRECTED VERSION (Before Row Tightening)
==========================================
EOF

for threshold in "${THRESHOLDS[@]}"; do
    output_file="$RESULTS_DIR/corrected_threshold_${threshold}n_${TIMESTAMP}.txt"
    echo "Threshold ${threshold}n:" >> "$COMPARISON_FILE"
    grep "sparse\|uniform" "$output_file" | grep -v "Type:" | head -2 >> "$COMPARISON_FILE" 2>/dev/null || echo "  No data" >> "$COMPARISON_FILE"
    echo "" >> "$COMPARISON_FILE"
done

cat >> "$COMPARISON_FILE" << EOF

ORIGINAL VERSION (After Row Tightening)
========================================
EOF

for threshold in "${THRESHOLDS[@]}"; do
    output_file="$RESULTS_DIR/original_threshold_${threshold}n_${TIMESTAMP}.txt"
    echo "Threshold ${threshold}n:" >> "$COMPARISON_FILE"
    grep "sparse\|uniform" "$output_file" | grep -v "Type:" | head -2 >> "$COMPARISON_FILE" 2>/dev/null || echo "  No data" >> "$COMPARISON_FILE"
    echo "" >> "$COMPARISON_FILE"
done

cat "$COMPARISON_FILE"

echo ""
echo "Quick comparison saved to: $COMPARISON_FILE"
echo ""
