#!/bin/bash
# Run locally after OVH training is complete.
# Downloads trained weights from OVH Object Storage and uploads to Modal volume.
# Requires: ovhai CLI (run from OVH JupyterLab, or x86 machine with ovhai)

set -euo pipefail

REGION="GRA"
OUTPUT_CONTAINER="btt-output"
MODAL_VOLUME="trading-data"

echo "=== Downloading weights from OVH Object Storage ==="
ovhai data download "$OUTPUT_CONTAINER@$REGION" ncp_v7_seed1.pt /tmp/ncp_v7_seed1.pt 2>/dev/null && \
    echo "Downloaded seed1" || echo "seed1 not ready yet"

ovhai data download "$OUTPUT_CONTAINER@$REGION" ncp_v7_seed2.pt /tmp/ncp_v7_seed2.pt 2>/dev/null && \
    echo "Downloaded seed2" || echo "seed2 not ready yet"

echo ""
echo "=== Uploading to Modal volume: $MODAL_VOLUME ==="
[ -f /tmp/ncp_v7_seed1.pt ] && modal volume put --force $MODAL_VOLUME /tmp/ncp_v7_seed1.pt ncp_v7_seed1.pt && echo "Uploaded seed1 to Modal"
[ -f /tmp/ncp_v7_seed2.pt ] && modal volume put --force $MODAL_VOLUME /tmp/ncp_v7_seed2.pt ncp_v7_seed2.pt && echo "Uploaded seed2 to Modal"

echo ""
echo "=== Done. Verify with: modal volume ls trading-data ==="
