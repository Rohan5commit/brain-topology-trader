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


# ── Cron 1: disabled until Alpaca + SMTP are wired ──────────────────────────
# Re-enable by adding: schedule=modal.Cron("30 17 * * *")
@app.function(
    image=image,
    secrets=_secrets,
    volumes={"/data": vol},
    cpu=4,
    memory=16384,
    gpu="T4",
    timeout=3600,
)
def run_inference_and_execute():
    """Inference + execution. Manually trigger: modal run modal_app.py::run_inference_and_execute"""
    import os
    import torch
    from datetime import datetime, timezone

    import config
    from data.ingest import DataIngestor
    from data.features import FeatureEngineer
    from model.ncp_model import NCPTradingModel
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
    model = NCPTradingModel(
        num_stocks=len(config.TICKER_UNIVERSE),
        num_features=config.NUM_FEATURES,
        input_size=config.INPUT_SIZE,
        ncp_units=config.NCP_UNITS,
        ncp_output_size=config.NCP_OUTPUT_SIZE,
        ncp_sparsity=config.NCP_SPARSITY,
        embedding_dim=config.EMBEDDING_DIM,
        num_sectors=config.NUM_SECTORS,
        sector_embedding_dim=config.SECTOR_EMBEDDING_DIM,
        dropout=0.0,
    ).to(device)

    weights = config.WEIGHTS_LATEST_PATH
    if not os.path.exists(weights):
        weights = config.WEIGHTS_BASE_PATH
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=device))
        log.info("Loaded weights from %s", weights)
    else:
        log.warning("No weights found — using random init")

    model.eval()
    ticker_to_idx = {t: i for i, t in enumerate(config.TICKER_UNIVERSE)}

    raw_signals: dict[str, list[float]] = {}
    with torch.no_grad():
        for ticker, feat_seq in features.items():
            if feat_seq is None or len(feat_seq) < config.SEQUENCE_LENGTH:
                continue
            x = torch.FloatTensor(feat_seq[-config.SEQUENCE_LENGTH:]).unsqueeze(0).to(device)
            idx = torch.LongTensor([ticker_to_idx.get(ticker, 0)]).to(device)
            sec = torch.LongTensor([config.TICKER_SECTOR.get(ticker, 12)]).to(device)
            logits = model(x, idx, sec)
            probs = F.softmax(logits, dim=-1)
            raw_signals[ticker] = probs.squeeze(0).cpu().tolist()

    log.info("Inference complete: %d tickers", len(raw_signals))

    processor = SignalProcessor()
    smoothed = processor.smooth_and_rank(raw_signals)

    broker = AlpacaBroker()
    sizer = KellySizer()
    portfolio_value = broker.get_portfolio_value()
    broker.close_stale_positions(smoothed, config.SIGNAL_THRESHOLD, config.MIN_HOLD_DAYS)

    orders = []
    longs = [(t, s) for t, s in smoothed.items() if s["score"] > config.SIGNAL_THRESHOLD and s["side"] == "buy"]
    shorts = [(t, s) for t, s in smoothed.items() if s["score"] < -config.SIGNAL_THRESHOLD and s["side"] == "sell"]
    candidates = sorted(longs, key=lambda x: -x[1]["score"])[:20] + \
                 sorted(shorts, key=lambda x: x[1]["score"])[:20]

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

    send_daily_report({
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "tickers_analyzed": len(raw_signals),
        "orders_placed": len(orders),
        "portfolio_value": portfolio_value,
        "top_longs": [t for t, _ in longs[:5]],
        "top_shorts": [t for t, _ in shorts[:5]],
    })
    log.info("Done — %d orders placed", len(orders))


# ── Cron 2: disabled until Alpaca + SMTP are wired ──────────────────────────
# Re-enable by adding: schedule=modal.Cron("0 22 * * *")
@app.function(
    image=image,
    secrets=_secrets,
    volumes={"/data": vol},
    gpu="A10G",
    timeout=7200,
)
def update_weights():
    """EOD weight update. Manually trigger: modal run modal_app.py::update_weights"""
    import os
    import torch
    from datetime import datetime, timezone

    import config
    from data.ingest import DataIngestor
    from model.ncp_model import NCPTradingModel
    from model.update import OnlineUpdater
    from reward.compute import RewardComputer
    from utils.logger import get_logger

    log = get_logger("weight_update")
    log.info("=== Weight Update start %s ===", datetime.now(timezone.utc).isoformat())

    ingestor = DataIngestor()
    closing = ingestor.fetch_closing_prices(config.TICKER_UNIVERSE)

    if not os.path.exists(config.SIGNALS_PATH):
        log.warning("No signals file found — skipping update")
        return

    import pandas as pd
    signals_df = pd.read_parquet(config.SIGNALS_PATH)

    reward_computer = RewardComputer()
    rewards = reward_computer.compute(signals_df, closing, None,
                                      alpha=config.ALPHA, beta=config.BETA)
    log.info("Avg reward: %.4f over %d positions", sum(rewards.values()) / max(len(rewards), 1), len(rewards))

    device = torch.device("cuda")
    model = NCPTradingModel(
        num_stocks=len(config.TICKER_UNIVERSE),
        num_features=config.NUM_FEATURES,
        input_size=config.INPUT_SIZE,
        ncp_units=config.NCP_UNITS,
        ncp_output_size=config.NCP_OUTPUT_SIZE,
        ncp_sparsity=config.NCP_SPARSITY,
        embedding_dim=config.EMBEDDING_DIM,
        num_sectors=config.NUM_SECTORS,
        sector_embedding_dim=config.SECTOR_EMBEDDING_DIM,
        dropout=0.0,
    ).to(device)

    weights = config.WEIGHTS_LATEST_PATH
    if not os.path.exists(weights):
        weights = config.WEIGHTS_BASE_PATH
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=device))

    updater = OnlineUpdater(model, lr=config.LEARNING_RATE)
    avg_reward = updater.update(signals_df, rewards, device, config.TICKER_UNIVERSE)
    log.info("Update complete — avg reward: %.4f", avg_reward)

    torch.save(model.state_dict(), config.WEIGHTS_LATEST_PATH)
    vol.commit()
    log.info("Weights saved to %s", config.WEIGHTS_LATEST_PATH)


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
