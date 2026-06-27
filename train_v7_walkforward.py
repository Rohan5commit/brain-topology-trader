#!/usr/bin/env python3
"""NCP v7 — walk-forward + 29 features + multi-task + mixup + recency weights — OVH H100 cuda:0."""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
SEED = 1
DEVICE_ID = 0

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ncps", "yfinance", "pyarrow"], check=True)

import torch as _t
assert _t.cuda.is_available(), "No CUDA"
print(f"GPU {DEVICE_ID}: {_t.cuda.get_device_name(DEVICE_ID)}")
del _t

import logging, random
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger(__name__)

import numpy as np, torch
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED); random.seed(SEED)
log.info("NCP v7 walk-forward | seed: %d | device: cuda:%d", SEED, DEVICE_ID)

import config
from model.train_walkforward import WalkForwardTrainer

WORK = "/workspace"
WEIGHTS_PATH = f"{WORK}/ncp_v7_seed1.pt"

def checkpoint_fn(model, fold, epoch):
    if fold == 15:  # only checkpoint final fold
        torch.save(model.state_dict(), WEIGHTS_PATH)
        log.info("Checkpoint fold %d epoch %d → %s", fold, epoch, WEIGHTS_PATH)

trainer = WalkForwardTrainer()
model = trainer.train(tickers=config.TICKER_UNIVERSE, checkpoint_fn=checkpoint_fn)

torch.save(model.state_dict(), WEIGHTS_PATH)
log.info("Done. Weights: %s", WEIGHTS_PATH)
print(f"\n=== NCP V7 WALK-FORWARD COMPLETE — {WEIGHTS_PATH} ===")
