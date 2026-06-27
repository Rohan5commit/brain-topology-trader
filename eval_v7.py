#!/usr/bin/env python3
"""Evaluate NCP v7 ensemble OOS accuracy (2024-2025)."""
import os, sys, logging
import numpy as np, pandas as pd, torch, torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, "/workspace/btt")
import config
from model.ncp_model_v5 import NCPTradingModelV5
from data.features import FeatureEngineer, FEATURE_NAMES
from model.train_walkforward import WalkForwardTrainer

OOS_START = "2024-01-01"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Device: %s", DEVICE)

MODEL_KWARGS = dict(
    num_stocks=len(config.TICKER_UNIVERSE),
    num_features=config.NUM_FEATURES,
    input_size=config.INPUT_SIZE,
    ncp_units=config.NCP_UNITS,
    ncp_output_size=config.NCP_OUTPUT_SIZE,
    ncp_sparsity=config.NCP_SPARSITY,
    embedding_dim=config.EMBEDDING_DIM,
    num_sectors=config.NUM_SECTORS,
    sector_embedding_dim=config.SECTOR_EMBEDDING_DIM,
    cs_heads=4, cs_dropout=0.1, dropout=0.0,
)

WEIGHT_FILES = {
    "v7_seed1": "/workspace/ncp_v7_seed1.pt",
    "v7_seed2": "/workspace/ncp_v7_seed2.pt",
    "v5_seed1": "/data/ncp_v5_seed1.pt",
    "v5_seed2": "/data/ncp_v5_seed2.pt",
}

models = {}
for name, path in WEIGHT_FILES.items():
    if os.path.exists(path):
        m = NCPTradingModelV5(**MODEL_KWARGS).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        models[name] = m
        log.info("Loaded: %s", name)
    else:
        log.warning("Missing: %s at %s", name, path)

# ── Build OOS samples via the walk-forward trainer ───────────────────────────
trainer = WalkForwardTrainer()
all_feat_dfs, all_close, macro_df, spy_5d_fwd, spy_3d_cum, vix_series = \
    trainer._fetch_all_data(config.TICKER_UNIVERSE, DEVICE)

returns_panel = pd.DataFrame({t: fd["returns_20d"] for t, fd in all_feat_dfs.items()})
rank_panel = returns_panel.rank(axis=1, pct=True).fillna(0.5)
sector_ret_panel = trainer._compute_sector_ret_panel(returns_panel)
fwd_excess = trainer._compute_excess_returns(all_feat_dfs, all_close, spy_5d_fwd)
excess_df = pd.DataFrame(fwd_excess)
q_low_full = excess_df.quantile(config.QUARTILE_THRESHOLD, axis=1)
q_high_full = excess_df.quantile(1 - config.QUARTILE_THRESHOLD, axis=1)
fwd_excess_20d = trainer._compute_excess_returns_nd(all_feat_dfs, all_close, spy_5d_fwd, n=20)
excess_20d_df = pd.DataFrame(fwd_excess_20d)
q_low_20d_full = excess_20d_df.quantile(config.QUARTILE_THRESHOLD, axis=1)
q_high_20d_full = excess_20d_df.quantile(1 - config.QUARTILE_THRESHOLD, axis=1)
ticker_to_idx = {t: i for i, t in enumerate(config.TICKER_UNIVERSE)}

X, y, y_20d, idx, sec, dates = trainer._build_samples(
    all_feat_dfs, rank_panel, sector_ret_panel, spy_3d_cum, fwd_excess,
    q_low_full, q_high_full, fwd_excess_20d, q_low_20d_full, q_high_20d_full,
    ticker_to_idx, date_start=OOS_START, date_end="2025-12-31",
)
log.info("OOS samples: %d", len(X))

X_t = torch.FloatTensor(X).to(DEVICE)
y_t = torch.LongTensor(y)
idx_t = torch.LongTensor(idx).to(DEVICE)
sec_t = torch.LongTensor(sec).to(DEVICE)

BS = 4096
results = {}
for name, m in models.items():
    all_probs = []
    with torch.no_grad():
        for s in range(0, len(X_t), BS):
            xb = X_t[s:s+BS]; ib = idx_t[s:s+BS]; sb = sec_t[s:s+BS]
            logits, _ = m(xb, ib, sb)
            all_probs.append(F.softmax(logits, dim=-1).cpu())
    probs = torch.cat(all_probs, dim=0)
    preds = probs.argmax(dim=1)
    acc = (preds == y_t).float().mean().item()
    results[name] = acc
    log.info("%-12s  OOS acc: %.4f%%", name, acc * 100)

# Ensemble: v7 seeds
v7_keys = [k for k in results if k.startswith("v7")]
if len(v7_keys) >= 2:
    all_probs_v7 = []
    with torch.no_grad():
        for s in range(0, len(X_t), BS):
            xb = X_t[s:s+BS]; ib = idx_t[s:s+BS]; sb = sec_t[s:s+BS]
            batch_probs = [F.softmax(models[k](xb, ib, sb)[0], dim=-1) for k in v7_keys]
            all_probs_v7.append(torch.stack(batch_probs).mean(0).cpu())
    ens_probs = torch.cat(all_probs_v7, dim=0)
    ens_acc = (ens_probs.argmax(1) == y_t).float().mean().item()
    log.info("v7 ensemble    OOS acc: %.4f%%", ens_acc * 100)

print("\n=== RESULTS ===")
for name, acc in sorted(results.items()):
    print(f"  {name:<14} {acc*100:.4f}%")
if len(v7_keys) >= 2:
    print(f"  v7_ensemble    {ens_acc*100:.4f}%")
print(f"  v5_baseline    54.5100%  (target to beat)")
