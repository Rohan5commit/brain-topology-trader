"""One-time historical supervised training (25 years, 3-class labels)."""
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from data.features import FeatureEngineer, FEATURE_NAMES
from model.ncp_model import NCPTradingModel

log = logging.getLogger(__name__)


def _next_day_label(close_series: pd.Series) -> pd.Series:
    """3-class label based on next-day return: 0=buy(>+0.5%), 1=hold, 2=sell(<-0.5%)."""
    ret = close_series.pct_change(1).shift(-1)
    labels = pd.Series(1, index=close_series.index, dtype=np.int64)
    labels[ret > 0.005] = 0
    labels[ret < -0.005] = 2
    labels[ret.isna()] = -1  # last row — exclude
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
        macro_df = None
        try:
            macro_raw = yf.download(
                ["^VIX", "^TNX", "^IRX", "SPY"],
                start=start_date, end=end_date,
                interval="1d", auto_adjust=True, progress=False,
            )
            # yfinance returns MultiIndex columns; "Close" is the price level
            if isinstance(macro_raw.columns, pd.MultiIndex):
                mc = macro_raw["Close"]
            else:
                mc = macro_raw
            mc.index = pd.to_datetime(mc.index).normalize()
            mc = mc.rename(columns={"^VIX": "vix"})
            mc["yield_curve_slope"] = mc["^TNX"] - mc["^IRX"]
            mc["spy_1d_return"] = mc["SPY"].pct_change(1)
            macro_df = mc[["vix", "yield_curve_slope", "spy_1d_return"]].ffill().fillna(0.0)
            log.info("Macro loaded: %d rows", len(macro_df))
        except Exception as exc:
            log.warning("Macro history fetch failed: %s — using stubs", exc)

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
            log.info(
                "Downloading batch %d–%d / %d tickers",
                batch_start + 1, batch_start + len(batch), len(tickers),
            )
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
                    # Extract per-ticker DataFrame
                    if len(batch) == 1:
                        df = raw.copy()
                    elif isinstance(raw.columns, pd.MultiIndex):
                        # yfinance 1.x: level 0 = ticker, level 1 = field
                        if ticker in raw.columns.get_level_values(0):
                            df = raw[ticker].copy()
                        elif ticker in raw.columns.get_level_values(1):
                            df = raw.xs(ticker, level=1, axis=1).copy()
                        else:
                            continue
                    else:
                        continue

                    df = df.rename(columns=str.lower).dropna(how="all")
                    df.index = pd.to_datetime(df.index).normalize()
                    if len(df) < config.SEQUENCE_LENGTH + 25:
                        continue

                    # ── O(n) feature computation ─────────────────────────────
                    # Call _stock_features ONCE per ticker (not per-day)
                    feat_df = engineer._stock_features(df)
                    if feat_df is None or len(feat_df) < config.SEQUENCE_LENGTH + 2:
                        continue

                    # Align daily macro values by date (vectorized join)
                    if macro_df is not None:
                        aligned = macro_df.reindex(feat_df.index, method="ffill")
                        feat_df["vix"] = aligned["vix"].fillna(20.0).values
                        feat_df["yield_curve_slope"] = aligned["yield_curve_slope"].fillna(0.5).values
                        feat_df["spy_1d_return"] = aligned["spy_1d_return"].fillna(0.0).values
                    else:
                        feat_df["vix"] = 20.0
                        feat_df["yield_curve_slope"] = 0.5
                        feat_df["spy_1d_return"] = 0.0

                    feat_df["sentiment_3d"] = 0.0
                    feat_df["momentum_rank"] = 0.5  # cross-sectional rank unavailable for historical

                    # Build labels aligned to feat_df index
                    label_series = _next_day_label(df["close"]).reindex(feat_df.index)

                    feat_arr = feat_df[FEATURE_NAMES].values.astype(np.float32)
                    feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
                    label_arr = label_series.values

                    # Build sliding windows
                    ticker_idx = ticker_to_idx[ticker]
                    for i in range(config.SEQUENCE_LENGTH, len(feat_arr) - 1):
                        label = int(label_arr[i]) if not np.isnan(label_arr[i]) else -1
                        if label not in (0, 1, 2):
                            continue
                        x_seq = feat_arr[i - config.SEQUENCE_LENGTH: i]
                        all_X.append(x_seq)
                        all_y.append(label)
                        all_idx.append(ticker_idx)

                except Exception as exc:
                    log.warning("Skipping %s: %s", ticker, exc)

            log.info("Samples so far: %d", len(all_X))

        if not all_X:
            raise RuntimeError("No training samples generated")

        log.info("Total training samples: %d", len(all_X))

        X = torch.FloatTensor(np.array(all_X))       # (N, seq_len, 17)
        y = torch.LongTensor(all_y)                   # (N,)
        idx_t = torch.LongTensor(all_idx)             # (N,)

        dataset = TensorDataset(X, idx_t, y)
        loader = DataLoader(
            dataset, batch_size=config.BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True,
        )

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.HISTORICAL_EPOCHS,
        )
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
            log.info(
                "Epoch %d/%d | loss=%.4f | acc=%.4f | lr=%.2e",
                epoch + 1, config.HISTORICAL_EPOCHS,
                total_loss / n, correct / n, scheduler.get_last_lr()[0],
            )

        return model
