"""One-time historical supervised training (25 years, 5-day labels, cross-sectional features)."""
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from data.features import FeatureEngineer, FEATURE_NAMES
from model.ncp_model import NCPTradingModel

log = logging.getLogger(__name__)


def _5day_label(close_series: pd.Series) -> pd.Series:
    """3-class label based on 5-day forward return: 0=buy(>+1%), 1=hold, 2=sell(<-1%)."""
    ret = close_series.pct_change(5).shift(-5)
    labels = pd.Series(1, index=close_series.index, dtype=np.int64)
    labels[ret > 0.01] = 0
    labels[ret < -0.01] = 2
    labels[ret.isna()] = -1
    return labels


class HistoricalTrainer:
    def train(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        checkpoint_fn=None,
        start_epoch: int = 0,
        weights_path: str = None,
    ) -> NCPTradingModel:
        import yfinance as yf

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Device: %s | tickers: %d", device, len(tickers))

        # ── Fetch macro once ─────────────────────────────────────────────────
        log.info("Fetching macro history (^VIX, ^TNX, ^IRX, SPY)…")
        macro_df = None
        try:
            macro_raw = yf.download(
                ["^VIX", "^TNX", "^IRX", "SPY"],
                start=start_date, end=end_date,
                interval="1d", auto_adjust=True, progress=False,
            )
            mc = macro_raw["Close"] if isinstance(macro_raw.columns, pd.MultiIndex) else macro_raw
            mc.index = pd.to_datetime(mc.index).normalize()
            mc = mc.rename(columns={"^VIX": "vix"})
            mc["yield_curve_slope"] = mc["^TNX"] - mc["^IRX"]
            mc["spy_1d_return"] = mc["SPY"].pct_change(1)
            macro_df = mc[["vix", "yield_curve_slope", "spy_1d_return"]].ffill().fillna(0.0)
            log.info("Macro loaded: %d rows", len(macro_df))
        except Exception as exc:
            log.warning("Macro fetch failed: %s — using stubs", exc)

        # Precompute SPY 3-day cumulative return for sentiment proxy
        spy_3d_cum = None
        if macro_df is not None:
            spy_3d_cum = macro_df["spy_1d_return"].rolling(3).sum().fillna(0.0)

        # ── Pass 1: download all tickers, build per-ticker feat_dfs ─────────
        log.info("Pass 1: downloading OHLCV and building features…")
        engineer = FeatureEngineer()
        all_feat_dfs: dict[str, pd.DataFrame] = {}
        all_close: dict[str, pd.Series] = {}
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}

        for batch_start in range(0, len(tickers), 100):
            batch = tickers[batch_start: batch_start + 100]
            log.info("Downloading batch %d–%d / %d", batch_start + 1, batch_start + len(batch), len(tickers))
            try:
                raw = yf.download(
                    batch, start=start_date, end=end_date,
                    interval="1d", auto_adjust=True, group_by="ticker", progress=False,
                )
            except Exception as exc:
                log.warning("Batch download failed: %s", exc)
                continue

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        df = raw.copy()
                    elif isinstance(raw.columns, pd.MultiIndex):
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
                    if len(df) < config.SEQUENCE_LENGTH + config.RETURN_HORIZON + 5:
                        continue

                    feat_df = engineer._stock_features(df)
                    if feat_df is None or len(feat_df) < config.SEQUENCE_LENGTH + config.RETURN_HORIZON:
                        continue

                    if macro_df is not None:
                        aligned = macro_df.reindex(feat_df.index, method="ffill")
                        feat_df["vix"] = aligned["vix"].fillna(20.0).values
                        feat_df["yield_curve_slope"] = aligned["yield_curve_slope"].fillna(0.5).values
                        feat_df["spy_1d_return"] = aligned["spy_1d_return"].fillna(0.0).values
                    else:
                        feat_df["vix"] = 20.0
                        feat_df["yield_curve_slope"] = 0.5
                        feat_df["spy_1d_return"] = 0.0

                    all_feat_dfs[ticker] = feat_df
                    all_close[ticker] = df["close"]

                except Exception as exc:
                    log.warning("Skipping %s: %s", ticker, exc)

            log.info("Feature DFs collected: %d tickers so far", len(all_feat_dfs))

        if not all_feat_dfs:
            raise RuntimeError("No feature data collected")

        log.info("Pass 1 done: %d tickers with features", len(all_feat_dfs))

        # ── Compute daily cross-sectional momentum rank ──────────────────────
        log.info("Computing daily cross-sectional momentum rank panel…")
        returns_panel = pd.DataFrame({t: fd["returns_20d"] for t, fd in all_feat_dfs.items()})
        rank_panel = returns_panel.rank(axis=1, pct=True).fillna(0.5)
        log.info("Rank panel: %d dates × %d tickers", len(rank_panel), len(rank_panel.columns))

        # ── Pass 2: fill cross-sectional features + build samples ────────────
        log.info("Pass 2: building training samples…")
        all_X: list[np.ndarray] = []
        all_y: list[int] = []
        all_idx: list[int] = []

        for ticker, feat_df in all_feat_dfs.items():
            try:
                feat_df = feat_df.copy()

                # Real cross-sectional momentum rank
                if ticker in rank_panel.columns:
                    feat_df["momentum_rank"] = (
                        rank_panel[ticker].reindex(feat_df.index).ffill().fillna(0.5).values
                    )
                else:
                    feat_df["momentum_rank"] = 0.5

                # Sentiment proxy: 3-day abnormal return vs SPY, normalised to [-1, 1]
                stock_3d = feat_df["returns_1d"].rolling(3).sum().fillna(0.0)
                if spy_3d_cum is not None:
                    spy_aligned = spy_3d_cum.reindex(feat_df.index, method="ffill").fillna(0.0)
                    abnormal = stock_3d - spy_aligned
                else:
                    abnormal = stock_3d
                feat_df["sentiment_3d"] = abnormal.clip(-0.15, 0.15) / 0.15

                # 5-day forward labels
                close = all_close[ticker].reindex(feat_df.index)
                label_arr = _5day_label(close).fillna(-1).astype(int).values

                feat_arr = feat_df[FEATURE_NAMES].values.astype(np.float32)
                feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)

                ticker_idx = ticker_to_idx.get(ticker, 0)
                for i in range(config.SEQUENCE_LENGTH, len(feat_arr) - config.RETURN_HORIZON):
                    label = int(label_arr[i])
                    if label not in (0, 1, 2):
                        continue
                    all_X.append(feat_arr[i - config.SEQUENCE_LENGTH: i])
                    all_y.append(label)
                    all_idx.append(ticker_idx)

            except Exception as exc:
                log.warning("Skipping %s in pass 2: %s", ticker, exc)

        if not all_X:
            raise RuntimeError("No training samples generated")

        log.info("Total training samples: %d", len(all_X))

        # Weighted CrossEntropy — inverse frequency weights to fix class imbalance
        y_np = np.array(all_y)
        counts = np.bincount(y_np, minlength=3).astype(np.float32)
        log.info("Class distribution — buy: %d | hold: %d | sell: %d", *counts.astype(int))
        weights = 1.0 / counts.clip(min=1)
        weights = weights / weights.mean()
        log.info("Class weights — buy: %.3f | hold: %.3f | sell: %.3f", *weights)

        X = torch.FloatTensor(np.array(all_X))
        y = torch.LongTensor(all_y)
        idx_t = torch.LongTensor(all_idx)

        dataset = TensorDataset(X, idx_t, y)
        loader = DataLoader(
            dataset, batch_size=config.BATCH_SIZE, shuffle=True,
            num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2,
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

        if weights_path and start_epoch > 0:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            log.info("Loaded checkpoint weights from %s (resuming epoch %d)", weights_path, start_epoch)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.HISTORICAL_EPOCHS, last_epoch=start_epoch
        )
        class_weights = torch.FloatTensor(weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # ── Training loop ────────────────────────────────────────────────────
        for epoch in range(start_epoch, config.HISTORICAL_EPOCHS):
            model.train()
            total_loss, correct, n = 0.0, 0, 0
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
            if checkpoint_fn is not None:
                checkpoint_fn(model, epoch + 1)

        return model
