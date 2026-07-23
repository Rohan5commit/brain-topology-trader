"""Daily inference + trade execution. Triggered by GitHub Actions at 17:30 UTC weekdays."""
import os
import sys
import pickle
import logging
import torch
import torch.nn.functional as F
import pandas as pd
from datetime import datetime, timezone

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(__file__))

import config
from data.ingest import DataIngestor
from data.features import FeatureEngineer
from model.ncp_model_v5 import NCPTradingModelV5 as NCPTradingModel
from execution.signals import SignalProcessor
from execution.sizing import KellySizer
from execution.broker import AlpacaBroker
from utils.logger import get_logger
from utils.notify import send_daily_report

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(REPO_DIR, "run_data"))
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", os.path.join(REPO_DIR, "weights"))
os.makedirs(DATA_DIR, exist_ok=True)

config.DAILY_SNAPSHOT_PATH = os.path.join(DATA_DIR, "daily_snapshot.parquet")
config.DAILY_FEATURES_PATH = os.path.join(DATA_DIR, "daily_features.pkl")
config.SIGNALS_PATH = os.path.join(DATA_DIR, "signals_history.parquet")
config.POSITIONS_PATH = os.path.join(DATA_DIR, "positions_state.parquet")

log = get_logger("inference")
log.info("=== Inference + Execution start %s ===", datetime.now(timezone.utc).isoformat())

ingestor = DataIngestor()
ohlcv = ingestor.fetch_ohlcv_all(config.TICKER_UNIVERSE)
macro = ingestor.fetch_macro()
sentiment = ingestor.fetch_sentiment(config.TICKER_UNIVERSE)
log.info("Data fetched: %d tickers OHLCV", len(ohlcv))

engineer = FeatureEngineer()
features = engineer.compute_features(ohlcv, macro, sentiment, ticker_sector=config.TICKER_SECTOR)
log.info("Features ready for %d tickers", len(features))

device = torch.device("cpu")

_V5_NUM_FEATURES = 22
_V5_INPUT_SIZE = _V5_NUM_FEATURES + config.EMBEDDING_DIM + config.SECTOR_EMBEDDING_DIM  # 62

_model_kwargs = dict(
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
)

_online_path = os.path.join(DATA_DIR, "ncp_v5_online.pt")
_seed1_path = os.path.join(WEIGHTS_DIR, "ncp_v5_seed1.pt")
_seed2_path = os.path.join(WEIGHTS_DIR, "ncp_v5_seed2.pt")

_seed_weight_paths = [
    _online_path if os.path.exists(_online_path) else _seed1_path,
    _seed2_path,
]
ensemble_models = []
for wp in _seed_weight_paths:
    if os.path.exists(wp):
        m = NCPTradingModel(**_model_kwargs).to(device)
        m.load_state_dict(torch.load(wp, map_location=device, weights_only=False), strict=False)
        m.eval()
        ensemble_models.append(m)
        log.info("Loaded ensemble member: %s", wp)
    else:
        log.warning("Ensemble weight missing (skipped): %s", wp)

if not ensemble_models:
    log.warning("No ensemble weights found — using random init")
    ensemble_models = [NCPTradingModel(**_model_kwargs).to(device)]
    ensemble_models[0].eval()

log.info("Ensemble size: %d models", len(ensemble_models))

ticker_to_idx = {t: i for i, t in enumerate(config.TICKER_UNIVERSE)}

eligible = [
    (t, feat_seq)
    for t, feat_seq in features.items()
    if feat_seq is not None and len(feat_seq) >= config.SEQUENCE_LENGTH
]
log.info("Eligible tickers for inference: %d", len(eligible))

raw_signals: dict[str, list[float]] = {}
_BATCH = 64
with torch.no_grad():
    for batch_start in range(0, len(eligible), _BATCH):
        batch = eligible[batch_start: batch_start + _BATCH]
        xs = torch.stack([
            torch.FloatTensor(feat_seq[-config.SEQUENCE_LENGTH:, :_V5_NUM_FEATURES])
            for _, feat_seq in batch
        ]).to(device)
        idxs = torch.LongTensor([ticker_to_idx.get(t, 0) for t, _ in batch]).to(device)
        secs = torch.LongTensor([config.TICKER_SECTOR.get(t, 12) for t, _ in batch]).to(device)
        member_probs = [F.softmax(m(xs, idxs, secs)[0], dim=-1) for m in ensemble_models]
        probs = torch.stack(member_probs).mean(dim=0)
        for i, (ticker, _) in enumerate(batch):
            raw_signals[ticker] = probs[i].cpu().tolist()

log.info("Inference complete: %d tickers", len(raw_signals))

today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
prev_closes = {t: float(ohlcv[t]["close"].iloc[-1]) for t in raw_signals if t in ohlcv}
snapshot_records = []
for ticker, probs in raw_signals.items():
    snapshot_records.append({
        "ticker": ticker,
        "date": today_str,
        "p_down": probs[0],
        "p_up": probs[1],
        "chosen_action": 1 if probs[1] > probs[0] else 0,
        "prev_close": prev_closes.get(ticker, 0.0),
    })
snapshot_df = pd.DataFrame(snapshot_records).set_index("ticker")
snapshot_df.to_parquet(config.DAILY_SNAPSHOT_PATH)
log.info("Daily snapshot saved: %d tickers → %s", len(snapshot_records), config.DAILY_SNAPSHOT_PATH)

feat_for_update = {
    t: feat_seq[-config.SEQUENCE_LENGTH:, :_V5_NUM_FEATURES]
    for t, feat_seq in eligible
}
with open(config.DAILY_FEATURES_PATH, "wb") as f:
    pickle.dump(feat_for_update, f)
log.info("Feature tensors saved: %d tickers → %s", len(feat_for_update), config.DAILY_FEATURES_PATH)

processor = SignalProcessor()
smoothed = processor.smooth_and_rank(raw_signals)

broker = AlpacaBroker()
sizer = KellySizer()
portfolio_value = broker.get_portfolio_value()
broker.close_stale_positions(smoothed, config.SIGNAL_THRESHOLD, config.MIN_HOLD_DAYS)

orders = []
open_positions = broker.get_open_positions()
longs = [
    (t, s) for t, s in smoothed.items()
    if s["confidence"] > config.SIGNAL_THRESHOLD
    and s["side"] == "buy"
    and t not in open_positions
]
shorts = [
    (t, s) for t, s in smoothed.items()
    if s["confidence"] > config.SIGNAL_THRESHOLD
    and s["side"] == "sell"
    and t not in open_positions
]
candidates = sorted(longs, key=lambda x: -x[1]["score"])[:10] + \
             sorted(shorts, key=lambda x: x[1]["score"])[:5]

for ticker, sig in candidates:
    notional = sizer.kelly_notional(
        p=sig["confidence"],
        b=config.KELLY_B,
        portfolio_value=portfolio_value,
        max_pct=config.MAX_POSITION_PCT,
    )
    if notional <= 0:
        continue
    order = broker.place_order(ticker=ticker, side=sig["side"], notional=notional)
    if order:
        orders.append(order)

processor.save_signals(raw_signals)

filled_longs = [o["ticker"] for o in orders if o["side"] == "buy"]
filled_shorts = [o["ticker"] for o in orders if o["side"] == "sell"]
log.info("Longs: %s", filled_longs)
log.info("Shorts: %s", filled_shorts)

send_daily_report({
    "date": today_str,
    "tickers_analyzed": len(raw_signals),
    "orders_placed": len(orders),
    "portfolio_value": portfolio_value,
    "top_longs": filled_longs or [t for t, _ in longs[:5]],
    "top_shorts": filled_shorts or [t for t, _ in shorts[:5]],
})
log.info("Done — %d orders placed", len(orders))
