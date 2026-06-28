#!/bin/bash
# Run from the OVH JupyterLab terminal on the 4-GPU notebook.
# Launches both NCP v7 seeds in parallel, locked to physical GPUs 2 and 3.
# GPUs 0 and 1 are reserved for the other agent — this script never touches them.
#
# Usage:
#   bash launch_notebook.sh          # train both seeds in parallel
#   bash launch_notebook.sh seed1    # seed 1 only (GPU 2)
#   bash launch_notebook.sh seed2    # seed 2 only (GPU 3)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

# Verify GPU layout before doing anything
echo "=== GPU inventory (all 4 physical GPUs) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""
echo "This script will use GPUs 2 and 3 only."
echo "GPUs 0 and 1 are reserved for the other agent."
echo ""

# Sanity check: confirm GPUs 2 and 3 are visible and free-ish
for GPU in 2 3; do
    MEM_USED=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [ "$MEM_USED" -gt 2000 ]; then
        echo "WARNING: GPU $GPU already has ${MEM_USED}MB used — another process may be running on it."
    fi
done

MODE=${1:-both}

run_seed() {
    local SEED=$1
    local GPU=$((SEED + 1))   # seed1→GPU2, seed2→GPU3
    local LOG="$LOG_DIR/train_v7_seed${SEED}.log"

    echo "Starting seed=$SEED on physical GPU $GPU → log: $LOG"
    SEED=$SEED \
    PYTHONPATH="$REPO_DIR" \
    nohup python "$REPO_DIR/train_v7_walkforward.py" \
        > "$LOG" 2>&1 &
    echo $! > "$LOG_DIR/seed${SEED}.pid"
    echo "  PID: $! (saved to $LOG_DIR/seed${SEED}.pid)"
}

case "$MODE" in
    seed1) run_seed 1 ;;
    seed2) run_seed 2 ;;
    both)
        run_seed 1
        run_seed 2
        echo ""
        echo "Both seeds running. Monitor with:"
        echo "  tail -f $LOG_DIR/train_v7_seed1.log"
        echo "  tail -f $LOG_DIR/train_v7_seed2.log"
        echo ""
        echo "Watch GPU usage (GPUs 2+3 only):"
        echo "  watch -n5 'nvidia-smi -i 2,3'"
        echo ""
        echo "Stop if needed:"
        echo "  kill \$(cat $LOG_DIR/seed1.pid) \$(cat $LOG_DIR/seed2.pid)"
        ;;
    *)
        echo "Usage: $0 [seed1|seed2|both]"
        exit 1
        ;;
esac
