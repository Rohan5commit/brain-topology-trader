import modal

app = modal.App("brain-topology-trader")

vol = modal.Volume.from_name("trading-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.1.0",
        "ncps>=0.0.7",
        "alpaca-py>=0.26.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "finnhub-python>=2.4.20",
        "requests>=2.31.0",
        "pyarrow>=14.0.0",
        "scikit-learn>=1.3.0",
        "yfinance>=0.2.36",
    ])
    # Upload local source packages so `import config`, `import model.train`, etc. work in container
    .add_local_python_source("config", "data", "model", "execution", "reward", "utils")
)

_secrets = [
    modal.Secret.from_name("alpaca-secret"),
    modal.Secret.from_name("twelvedata-secret"),
    modal.Secret.from_name("finnhub-secret"),
    modal.Secret.from_name("notify-secret"),
]


@app.function(
    image=image,
    secrets=_secrets,
    volumes={"/data": vol},
    cpu=4,
    memory=16384,
    gpu="T4",
    timeout=3600,
    schedule=modal.Cron("30 17 * * 1-5"),
)
def run_inference_and_execute():
    """Inference + execution. Manually trigger: modal run modal_app.py::run_inference_and_execute"""
    import os
    import torch
    from datetime import datetime, timezone

    import config
    from data.ingest import DataIngestor
    from data.features import FeatureEngineer
    from model.ncp_model_v5 import NCPTradingModelV5 as NCPTradingModel
    from execution.signals import SignalProcessor
    from execution.sizing import KellySizer
    from execution.broker import AlpacaBroker
    from utils.logger import get_logger
    from utils.notify import send_daily_report

    log = get_logger("inference")
    log.info("=== Inference + Execution start %s ===", datetime.now(timezone.utc).isoformat())

    ingestor = DataIngestor()
    ohlcv = ingestor.fetch_ohlcv_all(config.TICKER_UNIVERSE)
    macro = ingestor.fetch_macro()
    sentiment = ingestor.fetch_sentiment(config.TICKER_UNIVERSE)
    log.info("Data fetched: %d tickers OHLCV", len(ohlcv))

    engineer = FeatureEngineer()
    features = engineer.compute_features(ohlcv, macro, sentiment)
    log.info("Features ready for %d tickers", len(features))

    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # v5 was trained with 22 features; config.NUM_FEATURES is 29 (v7).
    # Hardcode v5 dims here to avoid mismatch.
    _V5_NUM_FEATURES = 22
    _V5_INPUT_SIZE = _V5_NUM_FEATURES + config.EMBEDDING_DIM + config.SECTOR_EMBEDDING_DIM  # 62

    _model_kwargs = dict(
        num_stocks=653,  # v5 was trained on 653 tickers; don't use current universe size
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

    # Use online weights for seed1 if available (daily fine-tuned), fall back to base seed weights
    _seed_weight_paths = [
        "/data/ncp_v5_online.pt" if os.path.exists("/data/ncp_v5_online.pt") else "/data/ncp_v5_seed1.pt",
        "/data/ncp_v5_seed2.pt",
    ]
    ensemble_models = []
    for wp in _seed_weight_paths:
        if os.path.exists(wp):
            m = NCPTradingModel(**_model_kwargs).to(device)
            m.load_state_dict(torch.load(wp, map_location=device), strict=False)
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

    # Build batched tensors for all eligible tickers — cross-sectional attention
    # works properly with the full batch rather than per-ticker (batch=1) calls.
    eligible = [
        (t, feat_seq)
        for t, feat_seq in features.items()
        if feat_seq is not None and len(feat_seq) >= config.SEQUENCE_LENGTH
    ]
    log.info("Eligible tickers for inference: %d", len(eligible))

    raw_signals: dict[str, list[float]] = {}
    _BATCH = 64  # process in chunks to stay within GPU memory
    with torch.no_grad():
        for batch_start in range(0, len(eligible), _BATCH):
            batch = eligible[batch_start: batch_start + _BATCH]
            xs = torch.stack([
                torch.FloatTensor(feat_seq[-config.SEQUENCE_LENGTH:, :_V5_NUM_FEATURES])
                for _, feat_seq in batch
            ]).to(device)  # (B, T, 22)
            idxs = torch.LongTensor([ticker_to_idx.get(t, 0) for t, _ in batch]).to(device)
            secs = torch.LongTensor([config.TICKER_SECTOR.get(t, 12) for t, _ in batch]).to(device)
            # Average softmax probabilities across ensemble members (primary 5d head)
            member_probs = [F.softmax(m(xs, idxs, secs)[0], dim=-1) for m in ensemble_models]
            probs = torch.stack(member_probs).mean(dim=0)  # (B, 2)
            for i, (ticker, _) in enumerate(batch):
                raw_signals[ticker] = probs[i].cpu().tolist()

    log.info("Inference complete: %d tickers", len(raw_signals))

    # Capture today's closing prices for reward computation in update_weights tonight
    import pandas as pd
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prev_closes = {t: float(ohlcv[t]["close"].iloc[-1]) for t in raw_signals if t in ohlcv}
    snapshot_records = []
    for ticker, probs in raw_signals.items():
        snapshot_records.append({
            "ticker": ticker,
            "date": today_str,
            "p_down": probs[0],
            "p_up": probs[1],
            "chosen_action": 1 if probs[1] > probs[0] else 0,  # 1=UP, 0=DOWN
            "prev_close": prev_closes.get(ticker, 0.0),
        })
    snapshot_df = pd.DataFrame(snapshot_records).set_index("ticker")
    snapshot_df.to_parquet(config.DAILY_SNAPSHOT_PATH)
    log.info("Daily snapshot saved: %d tickers → %s", len(snapshot_records), config.DAILY_SNAPSHOT_PATH)

    # Save feature tensors so update_weights doesn't re-fetch all OHLCV
    import pickle
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
    longs = [(t, s) for t, s in smoothed.items() if s["confidence"] > config.SIGNAL_THRESHOLD and s["side"] == "buy"]
    shorts = [(t, s) for t, s in smoothed.items() if s["confidence"] > config.SIGNAL_THRESHOLD and s["side"] == "sell"]
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
    vol.commit()

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


@app.function(
    image=image,
    secrets=_secrets,
    volumes={"/data": vol},
    gpu="A10G",
    timeout=7200,
    schedule=modal.Cron("0 22 * * 1-5"),
)
def update_weights():
    """EOD weight update — supervised fine-tune on today's actual direction labels.
    Manually trigger: modal run modal_app.py::update_weights"""
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pandas as pd
    from datetime import datetime, timezone

    import config
    from data.ingest import DataIngestor
    from model.ncp_model_v5 import NCPTradingModelV5
    from utils.logger import get_logger

    log = get_logger("weight_update")
    log.info("=== Weight Update start %s ===", datetime.now(timezone.utc).isoformat())

    if not os.path.exists(config.DAILY_SNAPSHOT_PATH):
        log.warning("No daily snapshot found — skipping update")
        return
    if not os.path.exists(config.DAILY_FEATURES_PATH):
        log.warning("No daily features found — skipping update")
        return

    snapshot_df = pd.read_parquet(config.DAILY_SNAPSHOT_PATH)
    log.info("Loaded snapshot: %d tickers from %s", len(snapshot_df),
             snapshot_df["date"].iloc[0] if "date" in snapshot_df.columns else "?")

    import pickle
    with open(config.DAILY_FEATURES_PATH, "rb") as f:
        saved_features = pickle.load(f)
    log.info("Loaded saved features: %d tickers", len(saved_features))

    # Fetch only closing prices — much faster than full OHLCV re-fetch
    ingestor = DataIngestor()
    closing = ingestor.fetch_closing_prices(config.TICKER_UNIVERSE)
    log.info("Closing prices fetched: %d tickers", len(closing))

    _V5_NUM_FEATURES = 22
    _V5_INPUT_SIZE = _V5_NUM_FEATURES + config.EMBEDDING_DIM + config.SECTOR_EMBEDDING_DIM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    seed1_path = "/data/ncp_v5_seed1.pt"
    online_path = "/data/ncp_v5_online.pt"
    base = online_path if os.path.exists(online_path) else seed1_path
    if os.path.exists(base):
        model.load_state_dict(torch.load(base, map_location=device), strict=False)
        log.info("Loaded base weights: %s", base)

    ticker_to_idx = {t: i for i, t in enumerate(config.TICKER_UNIVERSE)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Build training samples: prev_close from snapshot, curr_close from today's closing prices
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
        return

    model.train()
    _BATCH = 64
    total_loss = 0.0
    n_batches = 0
    for batch_start in range(0, len(samples), _BATCH):
        batch = samples[batch_start: batch_start + _BATCH]
        xs = torch.stack([
            torch.FloatTensor(feat_arr)  # already (120, 22) from snapshot
            for _, feat_arr, _ in batch
        ]).to(device)
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
    vol.commit()
    log.info("Online weights saved to %s", online_path)


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="A100",
    timeout=86400,
    memory=65536,
)
def train_historical():
    """One-time historical training.  Run: modal run --detach modal_app.py::train_historical"""
    import os
    import torch
    from datetime import datetime, timezone

    import config
    from model.train import HistoricalTrainer
    from utils.logger import get_logger

    log = get_logger("historical_training")
    log.info("=== Historical Training start %s ===", datetime.now(timezone.utc).isoformat())
    log.info("Period: %s → %s | tickers: %d", config.HISTORICAL_START, config.HISTORICAL_END, len(config.TICKER_UNIVERSE))

    # Resume from checkpoint if available
    _epoch_file = "/data/checkpoint_epoch.txt"
    start_epoch = 0
    weights_path = None
    if os.path.exists(_epoch_file) and os.path.exists(config.WEIGHTS_LATEST_PATH):
        with open(_epoch_file) as f:
            start_epoch = int(f.read().strip())
        weights_path = config.WEIGHTS_LATEST_PATH
        log.info("Resuming from epoch %d, weights: %s", start_epoch, weights_path)

    def _checkpoint(model, epoch):
        torch.save(model.state_dict(), config.WEIGHTS_LATEST_PATH)
        with open(_epoch_file, "w") as f:
            f.write(str(epoch))
        vol.commit()
        log.info("Checkpoint saved after epoch %d → %s", epoch, config.WEIGHTS_LATEST_PATH)

    trainer = HistoricalTrainer()
    model = trainer.train(
        tickers=config.TICKER_UNIVERSE,
        start_date=config.HISTORICAL_START,
        end_date=config.HISTORICAL_END,
        checkpoint_fn=_checkpoint,
        start_epoch=start_epoch,
        weights_path=weights_path,
    )

    torch.save(model.state_dict(), config.WEIGHTS_BASE_PATH)
    torch.save(model.state_dict(), config.WEIGHTS_LATEST_PATH)
    vol.commit()
    log.info("Training done — weights saved to %s", config.WEIGHTS_BASE_PATH)
