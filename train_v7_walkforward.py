#!/usr/bin/env python3
"""NCP v7 — walk-forward + 29 features + multi-task + mixup + recency weights.

GPU isolation: on multi-GPU nodes (4-GPU), seed1→GPU2 and seed2→GPU3.
On single-GPU nodes (OVH AI Training h100-1-gpu), CUDA_VISIBLE_DEVICES=0.
The mapping uses env var SINGLE_GPU=1 to force GPU0 regardless of seed.
"""
import os

# ── GPU isolation — MUST be set before any torch import ──────────────────────
SEED = int(os.environ.get("SEED", 1))
assert SEED in (1, 2), f"SEED must be 1 or 2, got {SEED}"
# SINGLE_GPU=1 → force GPU0 (OVH AI Training single-GPU job)
if os.environ.get("SINGLE_GPU", "0") == "1":
    _PHYSICAL_GPU = "0"
else:
    _PHYSICAL_GPU = {1: "2", 2: "3"}[SEED]       # multi-GPU node: seed1→GPU2, seed2→GPU3
os.environ["CUDA_VISIBLE_DEVICES"] = _PHYSICAL_GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ─────────────────────────────────────────────────────────────────────────────

import subprocess, sys
try:
    import ncps, yfinance, pyarrow  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ncps", "yfinance", "pyarrow"], check=True)

import torch as _t
assert _t.cuda.is_available(), "No CUDA visible"
assert _t.cuda.device_count() == 1, (
    f"HARD STOP: expected exactly 1 visible GPU (physical {_PHYSICAL_GPU}), "
    f"got {_t.cuda.device_count()}. Check CUDA_VISIBLE_DEVICES."
)
print(f"[NCP v7 seed={SEED}] Using physical GPU {_PHYSICAL_GPU} → cuda:0 ({_t.cuda.get_device_name(0)})")
del _t

DEVICE_ID = 0   # always 0 after CUDA_VISIBLE_DEVICES remapping

import logging, random
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger(__name__)

import numpy as np, torch
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED); random.seed(SEED)
log.info("NCP v7 walk-forward | seed: %d | device: cuda:%d", SEED, DEVICE_ID)

import config
from model.train_walkforward import WalkForwardTrainer

# Script lives at /workspace/btt/train_v7_walkforward.py — write weights alongside
WORK = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(WORK, f"ncp_v7_seed{SEED}.pt")

def checkpoint_fn(model, fold, epoch):
    if fold == 15:  # only checkpoint final fold
        torch.save(model.state_dict(), WEIGHTS_PATH)
        log.info("Checkpoint fold %d epoch %d → %s", fold, epoch, WEIGHTS_PATH)

trainer = WalkForwardTrainer()
model = trainer.train(tickers=config.TICKER_UNIVERSE, checkpoint_fn=checkpoint_fn)

torch.save(model.state_dict(), WEIGHTS_PATH)
log.info("Done. Weights: %s", WEIGHTS_PATH)
print(f"\n=== NCP V7 WALK-FORWARD COMPLETE — {WEIGHTS_PATH} ===")
