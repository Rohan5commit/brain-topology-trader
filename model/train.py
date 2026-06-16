"""One-time historical supervised training (25 years, 3-class labels)."""
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from data.features import FeatureEngineer
from model.ncp_model import NCPTradingModel

log = logging.getLogger(__name__)


def _next_day_label(close_series: pd.Series) -> np.ndarray:
    """3-class label based on next-day return: 0=buy(>+0.5%), 1=hold, 2=sell(<-0.5%)."""
    ret = close_series.pct_change(1).shift(-1).values
    labels = np.ones(len(ret), dtype=np.int64)
    labels[ret > 0.005] = 0
    labels[ret < -0.005] = 2
    return labels


class HistoricalTrainer:
    def train(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> NCPTradingModel:
        import yfinance as yf

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Device: %s | tickers: %d", device, len(tickers))

        # ── Fetch 25 years of macro (yfinance) — one call ───────────────────
        log.info("Fetching macro history (^VIX, ^TNX, ^IRX, SPY)…")
        try:
            macro_raw = yf.download(
                ["^VIX", "^TNX", "^IRX", "SPY"],
                start=start_date, end=end_date,
                interval="1d", auto_adjust=True, progress=False,
            )
            mc = macro_raw["Close"] if isinstance(macro_raw.columns, pd.MultiIndex) else macro_raw
            mc.index = pd.to_datetime(mc.index).normalize()
            mc["yield_curve_slope"] = mc["^TNX"] - mc["^IRX"]
            mc["spy_1d_return"] = mc["SPY"].pct_change(1)
            mc = mc.rename(columns={"^VIX": "vix"})
            macro_df = mc[["vix", "yield_curve_slope", "spy_1d_return"]].fillna(method="ffill").fillna(0.0)
        except Exception as exc:
            log.warning("Macro history fetch failed: %s — using stubs", exc)
            macro_df = None

        # ── Fetch price history in batches of 100 ───────────────────────────
        log.info("Fetching 25-year OHLCV via yfinance (batched)…")
        engineer = FeatureEngineer()
        all_X: list[np.ndarray] = []
        all_y: list[int] = []
        all_idx: list[int] = []
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}

        batch_size = 100
        for batch_start in range(0, len(tickers), batch_size):
            batch = tickers[batch_start: batch_start + batch_size]
            log.info("Downloading batch %d–%d / %d", batch_start + 1, batch_start + len(batch), len(tickers))
            try:
                raw = yf.download(
                    batch, start=start_date, end=end_date,
                    interval="1d", auto_adjust=True,
                    group_by="ticker", progress=False,
                )
            except Exception as exc:
                log.warning("Batch download failed: %s", exc)
                continue

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        df = raw.copy()
                    elif isinstance(raw.columns, pd.MultiIndex):
                        df = raw.xs(ticker, level=1, axis=1)
                    else:
                        continue

                    df = df.rename(columns=str.lower).dropna(how="all")
                    df.index = pd.to_datetime(df.index).normalize()
                    if len(df) < config.SEQUENCE_LENGTH + 5:
                        continue

                    # Build per-day macro dict for each row
                    feat_rows: list[np.ndarray] = []
                    for date_idx in df.index:
                        if macro_df is not None and date_idx in macro_df.index:
                            row_macro = macro_df.loc[date_idx].to_dict()
                        else:
                            row_macro = {"vix": 20.0, "yield_curve_slope": 0.5, "spy_1d_return": 0.0}

                        # Slice up to this date for feature computation
                        df_slice = df.loc[:date_idx]
                        if len(df_slice) < 21:
                            feat_rows.append(None)
                            continue

                        feat_map = engineer.compute_features(
                            {ticker: df_slice},
                            row_macro,
                            {ticker: 0.0},
                        )
                        feat_arr = feat_map.get(ticker)
                        if feat_arr is None or len(feat_arr) == 0:
                            feat_rows.append(None)
                        else:
                            feat_rows.append(feat_arr[-1])  # latest day's features

                    # Build sequences
                    labels = _next_day_label(df["close"])
                    valid_indices = [i for i, r in enumerate(feat_rows) if r is not None]

                    for i in range(config.SEQUENCE_LENGTH, len(valid_indices) - 1):
                        window_idx = valid_indices[i - config.SEQUENCE_LENGTH: i]
                        if len(window_idx) < config.SEQUENCE_LENGTH:
                            continue
                        x_seq = np.stack([feat_rows[j] for j in window_idx])  # (seq, 17)
                        label = int(labels[valid_indices[i]])
                        if label not in (0, 1, 2):
                            continue
                        all_X.append(x_seq)
                        all_y.append(label)
                        all_idx.append(ticker_to_idx[ticker])

                except Exception as exc:
                    log.warning("Skipping %s: %s", ticker, exc)

            log.info("Samples so far: %d", len(all_X))

        if not all_X:
            raise RuntimeError("No training samples generated")

        log.info("Total training samples: %d", len(all_X))

        X = torch.FloatTensor(np.array(all_X))
        y = torch.LongTensor(all_y)
        idx_t = torch.LongTensor(all_idx)

        dataset = TensorDataset(X, idx_t, y)
        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

        # ── Model ────────────────────────────────────────────────────────────
        model = NCPTradingModel(
            num_stocks=len(tickers),
            input_size=config.INPUT_SIZE,
            ncp_units=config.NCP_UNITS,
            ncp_output_size=config.NCP_OUTPUT_SIZE,
            ncp_sparsity=config.NCP_SPARSITY,
            embedding_dim=config.EMBEDDING_DIM,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.HISTORICAL_EPOCHS)
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
                probs = model(xb, ib)
                loss = criterion(probs, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(yb)
                correct += (probs.argmax(1) == yb).sum().item()
                n += len(yb)
            scheduler.step()
            acc = correct / n
            log.info("Epoch %d/%d | loss=%.4f | acc=%.4f | lr=%.2e",
                     epoch + 1, config.HISTORICAL_EPOCHS,
                     total_loss / n, acc, scheduler.get_last_lr()[0])

        return model
