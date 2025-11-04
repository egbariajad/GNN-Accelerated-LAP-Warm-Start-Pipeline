#!/bin/bash
# Monitor the fallback threshold benchmark job

JOB_ID=1441
LOG_FILE="/home/projects/nssl-prj10106/logs/fallback_threshold_benchmark_${JOB_ID}.out"
RESULTS_DIR="/home/projects/nssl-prj10106/fallback_threshold_results"

echo "========================================================================"
echo "Fallback Threshold Benchmark - Job Monitor"
echo "========================================================================"
echo ""

# Check if job is still running
if squeue -j $JOB_ID &> /dev/null; then
    echo "✓ Job Status: RUNNING"
    squeue -j $JOB_ID
    echo ""
else
    echo "✗ Job Status: COMPLETED or NOT FOUND"
    echo ""
fi

# Show last 30 lines of output
echo "========================================================================"
echo "Recent Output (last 30 lines):"
echo "========================================================================"
if [ -f "$LOG_FILE" ]; then
    tail -30 "$LOG_FILE"
else
    echo "Log file not yet created: $LOG_FILE"
fi

echo ""
echo "========================================================================"
echo "Results Generated So Far:"
echo "========================================================================"
if [ -d "$RESULTS_DIR" ]; then
    ls -lht "$RESULTS_DIR" | head -15
    echo ""
    echo "Total files: $(ls -1 $RESULTS_DIR 2>/dev/null | wc -l)"
else
    echo "Results directory not yet created"
fi

echo ""
echo "========================================================================"
echo "To Monitor Live:"
echo "========================================================================"
echo "  tail -f $LOG_FILE"
echo ""
echo "To Check Status:"
echo "  squeue -j $JOB_ID"
echo ""
echo "To Cancel Job:"
echo "  scancel $JOB_ID"
echo ""
