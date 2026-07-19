"""Nightly weight update via supervised CE fine-tune. Triggered by GitHub Actions at 22:00 UTC weekdays."""
import os
import sys
import pickle
import logging
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

import config
from data.ingest import DataIngestor
from model.ncp_model_v5 import NCPTradingModelV5
from utils.logger import get_logger

DATA_DIR = os.environ.get("DATA_DIR", "/teamspace/studios/this_studio/data")

config.DAILY_SNAPSHOT_PATH = os.path.join(DATA_DIR, "daily_snapshot.parquet")
config.DAILY_FEATURES_PATH = os.path.join(DATA_DIR, "daily_features.pkl")

log = get_logger("weight_update")
log.info("=== Weight Update start %s ===", datetime.now(timezone.utc).isoformat())

if not os.path.exists(config.DAILY_SNAPSHOT_PATH):
    log.warning("No daily snapshot found — skipping update")
    sys.exit(0)
if not os.path.exists(config.DAILY_FEATURES_PATH):
    log.warning("No daily features found — skipping update")
    sys.exit(0)

snapshot_df = pd.read_parquet(config.DAILY_SNAPSHOT_PATH)
log.info("Loaded snapshot: %d tickers from %s", len(snapshot_df),
         snapshot_df["date"].iloc[0] if "date" in snapshot_df.columns else "?")

with open(config.DAILY_FEATURES_PATH, "rb") as f:
    saved_features = pickle.load(f)
log.info("Loaded saved features: %d tickers", len(saved_features))

ingestor = DataIngestor()
closing = ingestor.fetch_closing_prices(config.TICKER_UNIVERSE)
log.info("Closing prices fetched: %d tickers", len(closing))

_V5_NUM_FEATURES = 22
_V5_INPUT_SIZE = _V5_NUM_FEATURES + config.EMBEDDING_DIM + config.SECTOR_EMBEDDING_DIM
device = torch.device("cpu")

model = NCPTradingModelV5(
    num_stocks=653,
    num_features=_V5_NUM_FEATURES,
    input_size=_V5_INPUT_SIZE,
    ncp_units=config.NCP_UNITS,
    ncp_output_size=config.NCP_OUTPUT_SIZE,
    ncp_sparsity=config.NCP_SPARSITY,
    embedding_dim=config.EMBEDDING_DIM,
    num_sectors=config.NUM_SECTORS,
    sector_embedding_dim=config.SECTOR_EMBEDDING_DIM,
    cs_heads=4,
    cs_dropout=0.1,
    dropout=0.0,
).to(device)

online_path = os.path.join(DATA_DIR, "ncp_v5_online.pt")
seed1_path = os.path.join(DATA_DIR, "ncp_v5_seed1.pt")
base = online_path if os.path.exists(online_path) else seed1_path
if os.path.exists(base):
    model.load_state_dict(torch.load(base, map_location=device), strict=False)
    log.info("Loaded base weights: %s", base)

ticker_to_idx = {t: i for i, t in enumerate(config.TICKER_UNIVERSE)}
optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

samples = []
for ticker in snapshot_df.index:
    row = snapshot_df.loc[ticker]
    prev_close = float(row.get("prev_close", 0.0))
    feat_arr = saved_features.get(ticker)
    if feat_arr is None:
        continue
    curr_close = closing.get(ticker, 0.0)
    if prev_close <= 0 or curr_close <= 0:
        continue
    actual_up = 1 if curr_close > prev_close else 0
    samples.append((ticker, feat_arr, actual_up))

log.info("Training samples: %d", len(samples))
if not samples:
    log.warning("No valid training samples — skipping update")
    sys.exit(0)

model.train()
_BATCH = 64
total_loss = 0.0
n_batches = 0
for batch_start in range(0, len(samples), _BATCH):
    batch = samples[batch_start: batch_start + _BATCH]
    xs = torch.stack([torch.FloatTensor(feat_arr) for _, feat_arr, _ in batch]).to(device)
    idxs = torch.LongTensor([ticker_to_idx.get(t, 0) for t, _, _ in batch]).to(device)
    secs = torch.LongTensor([config.TICKER_SECTOR.get(t, 12) for t, _, _ in batch]).to(device)
    labels = torch.LongTensor([label for _, _, label in batch]).to(device)

    logits, _ = model(xs, idxs, secs)
    loss = criterion(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    total_loss += loss.item()
    n_batches += 1

avg_loss = total_loss / max(n_batches, 1)
log.info("Supervised update: %d samples, %d batches, avg_loss=%.4f", len(samples), n_batches, avg_loss)

torch.save(model.state_dict(), online_path)
log.info("Online weights saved to %s", online_path)
