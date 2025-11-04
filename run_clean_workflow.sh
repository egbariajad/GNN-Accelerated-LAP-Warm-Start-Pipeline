#!/bin/bash
#
# Master script to execute the complete clean dataset workflow
# Usage: ./run_clean_workflow.sh [step]
#   step 1: Generate datasets
#   step 2: Verify datasets
#   step 3: Submit training jobs
#   step all: Run all steps
#

set -e

STEP="${1:-menu}"

show_menu() {
    echo "========================================="
    echo "CLEAN DATASET WORKFLOW"
    echo "========================================="
    echo ""
    echo "Select a step to execute:"
    echo "  1) Generate clean datasets (~30-60 min)"
    echo "  2) Verify datasets (~2 min)"
    echo "  3) Submit training jobs (3-5 days)"
    echo "  4) Run ALL steps (auto-pilot)"
    echo "  5) Show status"
    echo "  q) Quit"
    echo ""
    read -p "Enter choice [1-5 or q]: " choice
    
    case $choice in
        1) run_step_1 ;;
        2) run_step_2 ;;
        3) run_step_3 ;;
        4) run_all ;;
        5) show_status ;;
        q) exit 0 ;;
        *) echo "Invalid choice"; exit 1 ;;
    esac
}

run_step_1() {
    echo ""
    echo "========================================="
    echo "STEP 1: Generate Clean Datasets"
    echo "========================================="
    echo "This will create data/generated/processed_clean/"
    echo "Estimated time: 30-60 minutes"
    echo "Starting generation..."
    echo ""
    
    chmod +x generate_clean_datasets.sh
    ./generate_clean_datasets.sh
    
    echo ""
    echo "✅ Step 1 complete!"
    echo "Next: Run './run_clean_workflow.sh 2' to verify"
}

run_step_2() {
    echo ""
    echo "========================================="
    echo "STEP 2: Verify Clean Datasets"
    echo "========================================="
    echo "This will check all generated HDF5 files"
    echo "Estimated time: 2 minutes"
    echo ""
    
    python verify_clean_datasets.py
    
    status=$?
    echo ""
    if [ $status -eq 0 ]; then
        echo "✅ Step 2 complete! All datasets are clean."
        echo "Next: Run './run_clean_workflow.sh 3' to submit training jobs"
    else
        echo "❌ Verification failed. Please check errors above."
        exit 1
    fi
}

run_step_3() {
    echo ""
    echo "========================================="
    echo "STEP 3: Submit Training Jobs"
    echo "========================================="
    echo "This will submit 3 SLURM jobs:"
    echo "  - one_gnn_small_full_clean"
    echo "  - one_gnn_mid1536_full_clean"
    echo "  - one_gnn_large4096_clean"
    echo ""
    echo "Total estimated time: 2-4 days"
    echo "Submitting jobs..."
    echo ""
    
    sbatch run_train_one_gnn_small_full_clean.slurm
    sbatch run_train_one_gnn_mid1536_full_clean.slurm
    sbatch run_train_one_gnn_large4096_clean.slurm
    
    echo ""
    echo "✅ Step 3 complete! Jobs submitted."
    echo ""
    echo "Monitor progress with:"
    echo "  squeue -u \$USER"
    echo "  tail -f logs/slurm/one_gnn_small_full_clean-*.out"
    echo ""
    echo "After training completes, benchmark with:"
    echo "  python scripts/gnn_benchmark.py --models one_gnn_small_full_clean.pt --sizes 512"
}

run_all() {
    echo ""
    echo "========================================="
    echo "AUTO-PILOT: Running All Steps"
    echo "========================================="
    echo "This will:"
    echo "  1. Generate all clean datasets"
    echo "  2. Verify datasets"
    echo "  3. Submit training jobs"
    echo ""
    echo "Starting auto-pilot mode..."
    echo ""
    
    run_step_1
    echo ""
    echo "Waiting 3 seconds before verification..."
    sleep 3
    
    run_step_2
    echo ""
    echo "Waiting 3 seconds before job submission..."
    sleep 3
    
    run_step_3
    
    echo ""
    echo "========================================="
    echo "✅ ALL STEPS COMPLETE"
    echo "========================================="
}

show_status() {
    echo ""
    echo "========================================="
    echo "CLEAN WORKFLOW STATUS"
    echo "========================================="
    echo ""
    
    # Check datasets
    if [ -d "data/generated/processed_clean" ]; then
        dataset_count=$(find data/generated/processed_clean -name "*.h5" -type f 2>/dev/null | wc -l)
        echo "✅ Clean datasets directory exists"
        echo "   Found $dataset_count HDF5 files (expected 21)"
    else
        echo "❌ Clean datasets not generated yet"
        echo "   Run: ./run_clean_workflow.sh 1"
    fi
    
    echo ""
    
    # Check checkpoints
    if [ -d "gnn/checkpoints_clean" ]; then
        checkpoint_count=$(ls gnn/checkpoints_clean/*.pt 2>/dev/null | wc -l)
        echo "✅ Clean checkpoints directory exists"
        echo "   Found $checkpoint_count models (expected 5)"
    else
        echo "⏳ No clean checkpoints yet (created during training)"
    fi
    
    echo ""
    
    # Check running jobs
    running_jobs=$(squeue -u $USER -h 2>/dev/null | grep -c "one_gnn.*clean" || echo "0")
    if [ "$running_jobs" -gt 0 ]; then
        echo "🔄 Currently running: $running_jobs clean training jobs"
        echo ""
        squeue -u $USER | grep "one_gnn.*clean" || true
    else
        echo "⏳ No clean training jobs currently running"
    fi
    
    echo ""
    echo "========================================="
}

case $STEP in
    1) run_step_1 ;;
    2) run_step_2 ;;
    3) run_step_3 ;;
    all) run_all ;;
    status) show_status ;;
    menu) show_menu ;;
    *) 
        echo "Usage: $0 [1|2|3|all|status]"
        echo "  1      - Generate datasets"
        echo "  2      - Verify datasets"
        echo "  3      - Submit training jobs"
        echo "  all    - Run all steps"
        echo "  status - Show current status"
        echo "  (no arg) - Show interactive menu"
        exit 1
        ;;
esac
