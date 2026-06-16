"""One-time historical supervised training (25 years, 3-class labels)."""
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from data.ingest import DataIngestor
from data.features import FeatureEngineer
from model.ncp_model import NCPTradingModel

log = logging.getLogger(__name__)


def _next_day_label(returns_1d: np.ndarray) -> np.ndarray:
    """Convert next-day return into 3-class label: 0=buy, 1=hold, 2=sell."""
    labels = np.ones(len(returns_1d), dtype=np.int64)   # default: hold
    labels[returns_1d > 0.005] = 0   # buy  (+0.5% threshold)
    labels[returns_1d < -0.005] = 2  # sell (-0.5% threshold)
    return labels


class HistoricalTrainer:
    def train(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> NCPTradingModel:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Training on %s | tickers=%d", device, len(tickers))

        # ── Fetch data ───────────────────────────────────────────────────────
        log.info("Fetching 25-year OHLCV via yfinance (bulk)…")
        try:
            import yfinance as yf
            raw = yf.download(
                tickers, start=start_date, end=end_date,
                interval="1d", group_by="ticker", auto_adjust=True,
                threads=True, progress=False,
            )
        except Exception as exc:
            log.error("yfinance download failed: %s", exc)
            raise

        # ── Build feature sequences & labels ────────────────────────────────
        engineer = FeatureEngineer()
        macro_stub = {"vix": 20.0, "yield_curve_slope": 0.5, "fed_funds_rate": 5.0}
        sentiment_stub = {t: 0.0 for t in tickers}

        all_X: list[np.ndarray] = []
        all_y: list[int] = []
        all_idx: list[int] = []
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}

        for i, ticker in enumerate(tickers):
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    df = raw.xs(ticker, level=1, axis=1).dropna(how="all")
                else:
                    df = raw.dropna(how="all")
                df = df.rename(columns=str.lower)
                if len(df) < config.SEQUENCE_LENGTH + 5:
                    continue
                feat_map = engineer.compute_features({ticker: df}, macro_stub, {ticker: 0.0})
                feat_arr = feat_map.get(ticker)
                if feat_arr is None:
                    continue

                future_returns = pd.Series(df["close"].values).pct_change(1).shift(-1).values
                labels = _next_day_label(future_returns)

                seq = config.SEQUENCE_LENGTH
                for t in range(seq, len(feat_arr) - 1):
                    x_seq = feat_arr[t - seq: t]          # (seq, 17)
                    label = int(labels[t])
                    all_X.append(x_seq)
                    all_y.append(label)
                    all_idx.append(ticker_to_idx[ticker])

            except Exception as exc:
                log.warning("Skipping %s: %s", ticker, exc)

            if i % 50 == 0:
                log.info("Processed %d/%d tickers, samples=%d", i + 1, len(tickers), len(all_X))

        if not all_X:
            raise RuntimeError("No training samples generated")

        log.info("Total samples: %d", len(all_X))

        X = torch.FloatTensor(np.array(all_X))              # (N, seq, 17)
        y = torch.LongTensor(all_y)                          # (N,)
        idx_t = torch.LongTensor(all_idx)                    # (N,)

        dataset = TensorDataset(X, idx_t, y)
        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)

        # ── Model + optimizer ────────────────────────────────────────────────
        model = NCPTradingModel(
            num_stocks=len(tickers),
            input_size=config.INPUT_SIZE,
            ncp_units=config.NCP_UNITS,
            ncp_output_size=config.NCP_OUTPUT_SIZE,
            ncp_sparsity=config.NCP_SPARSITY,
            embedding_dim=config.EMBEDDING_DIM,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        # ── Training loop ────────────────────────────────────────────────────
        for epoch in range(config.HISTORICAL_EPOCHS):
            model.train()
            total_loss = 0.0
            correct = 0
            n = 0
            for xb, ib, yb in loader:
                xb, ib, yb = xb.to(device), ib.to(device), yb.to(device)
                optimizer.zero_grad()
                probs = model(xb, ib)                        # (B, 3)
                loss = criterion(probs, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(yb)
                correct += (probs.argmax(1) == yb).sum().item()
                n += len(yb)

            acc = correct / n
            log.info("Epoch %d/%d | loss=%.4f | acc=%.3f", epoch + 1, config.HISTORICAL_EPOCHS, total_loss / n, acc)

        return model
