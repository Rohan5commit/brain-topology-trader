#!/bin/bash
# Run this from an OVH JupyterLab terminal or any machine with ovhai CLI logged in.
# Usage:
#   bash submit_ovh_job.sh seed1       # train seed 1
#   bash submit_ovh_job.sh seed2       # train seed 2
#   bash submit_ovh_job.sh both        # submit both jobs (sequential — 1 GPU)
#
# Prerequisites:
#   1. ovhai CLI installed and logged in (ovhai login)
#   2. Two OVH Object Storage containers created:
#        btt-output   — receives trained weights + logs
#        btt-altdata  — contains the 3 alt data parquet files (upload once)
#   3. Alt data uploaded to btt-altdata container (see SETUP below)

set -euo pipefail

REGION="GRA"                       # change to your OVH region (GRA/BHS/SBG/WAW)
OUTPUT_CONTAINER="btt-output"
ALTDATA_CONTAINER="btt-altdata"
DOCKER_IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
GIT_REPO="https://github.com/Rohan5commit/brain-topology-trader"
FLAVOR="ai1-1-gpu"                 # 1x H100 SXM5 80GB on OVH

MODE=${1:-seed1}

# ── SETUP (run once) ─────────────────────────────────────────────────────────
# If containers don't exist yet, create them:
#   ovhai data store create btt-output --region $REGION
#   ovhai data store create btt-altdata --region $REGION
# Upload alt data (download from Modal volume first with `modal volume get`):
#   ovhai data upload btt-altdata@$REGION earnings_surprises.parquet alt_data/earnings_surprises.parquet
#   ovhai data upload btt-altdata@$REGION short_interest.parquet alt_data/short_interest.parquet
#   ovhai data upload btt-altdata@$REGION options_snapshot.parquet alt_data/options_snapshot.parquet

run_job() {
    local SEED=$1
    echo "Submitting NCP v7 seed=$SEED job..."
    ovhai job run \
        --name "btt-ncp-v7-seed${SEED}" \
        --flavor "$FLAVOR" \
        --region "$REGION" \
        --volume "$ALTDATA_CONTAINER@$REGION:/workspace/altdata:ro" \
        --volume "$OUTPUT_CONTAINER@$REGION:/workspace/output:rw" \
        --env "SEED=$SEED" \
        --env "GIT_REPO=$GIT_REPO" \
        --unsecure-http \
        "$DOCKER_IMAGE" \
        -- bash -c "
            apt-get update -q && apt-get install -y -q git &&
            pip install -q ncps yfinance pyarrow &&
            git clone '$GIT_REPO' /workspace/btt &&
            cd /workspace/btt &&
            mkdir -p data/alt_data &&
            cp /workspace/altdata/*.parquet data/alt_data/ 2>/dev/null || true &&
            ls -lh data/alt_data/ &&
            PYTHONPATH=/workspace/btt python train_v7_walkforward.py 2>&1 | tee /workspace/output/train_v7_seed${SEED}.log &&
            cp ncp_v7_seed${SEED}.pt /workspace/output/ &&
            echo DONE
        "
    echo "Job submitted. Monitor with: ovhai job logs <job-id> --follow"
}

case "$MODE" in
    seed1) run_job 1 ;;
    seed2) run_job 2 ;;
    both)
        run_job 1
        echo "Seed 1 submitted. Submitting seed 2..."
        run_job 2
        ;;
    *)
        echo "Usage: $0 [seed1|seed2|both]"
        exit 1
        ;;
esac
