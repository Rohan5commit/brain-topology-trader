#!/bin/bash
# OVH AI Training entry point.
# Environment variables expected (set via --env in ovhai job run):
#   SEED      — integer seed (1 or 2)
#   GIT_REPO  — https URL to the repo (no auth needed for public repo)
# OVH mounts:
#   /workspace/output  — Object Storage bucket for output weights
#   /workspace/altdata — Object Storage bucket containing alt_data/ parquet files

set -euo pipefail
SEED=${SEED:-1}
GIT_REPO=${GIT_REPO:-"https://github.com/Rohan5commit/brain-topology-trader"}

echo "=== OVH AI Training | NCP v7 | seed=$SEED ==="
nvidia-smi

# Clone repo into /workspace/btt
cd /workspace
git clone "$GIT_REPO" btt
cd btt

# Symlink alt_data into data/ so features.py finds it at the relative path
# features.py looks for: <repo>/data/alt_data/
mkdir -p data/alt_data
if [ -d /workspace/altdata/alt_data ]; then
    # mounted as /workspace/altdata/alt_data/
    ln -sfn /workspace/altdata/alt_data data/alt_data
    echo "Alt data linked from /workspace/altdata/alt_data"
elif [ -d /workspace/altdata ]; then
    # parquet files directly in the bucket root
    for f in earnings_surprises.parquet short_interest.parquet options_snapshot.parquet; do
        [ -f /workspace/altdata/$f ] && cp /workspace/altdata/$f data/alt_data/$f
    done
    echo "Alt data copied from /workspace/altdata/"
else
    echo "WARNING: No alt data found at /workspace/altdata — new features will use defaults"
fi

ls -lh data/alt_data/ 2>/dev/null || true

# Run training
WEIGHTS_NAME="ncp_v7_seed${SEED}.pt"
OUTPUT_PATH="/workspace/output/${WEIGHTS_NAME}"

PYTHONPATH=/workspace/btt python train_v7_walkforward.py \
    2>&1 | tee /workspace/output/train_v7_seed${SEED}.log

# The training script saves to /workspace/btt/ncp_v7_seed1.pt (or seed2)
# Copy to output volume so it persists after job ends
LOCAL_WEIGHTS="/workspace/btt/ncp_v7_seed${SEED}.pt"
if [ -f "$LOCAL_WEIGHTS" ]; then
    cp "$LOCAL_WEIGHTS" "$OUTPUT_PATH"
    echo "=== Weights saved to $OUTPUT_PATH ==="
    ls -lh "$OUTPUT_PATH"
else
    echo "ERROR: weights file not found at $LOCAL_WEIGHTS"
    exit 1
fi

echo "=== NCP v7 seed=$SEED COMPLETE ==="
