"""Walk-forward cross-validation trainer for NCPTradingModelV5.

Folds:
  train 2000–2009, val 2010
  train 2000–2010, val 2011
  …
  train 2000–2022, val 2023
  train 2000–2023, val 2024  ← final fold (60 epochs, returned)

Each fold saves the best-val-acc checkpoint during training.
"""
import logging
import os
import tempfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import config
from data.features import FeatureEngineer, FEATURE_NAMES
from model.ncp_model_v5 import NCPTradingModelV5

log = logging.getLogger(__name__)

# Walk-forward constants
WF_FIRST_TRAIN_END = 2009   # first fold trains up to end of this year
WF_VAL_START = 2010         # first val year
WF_FINAL_VAL = 2024         # last val year (defines last fold)
WF_FOLD_EPOCHS = 20         # epochs per intermediate fold
WF_FINAL_EPOCHS = 60        # epochs for the final fold (== config.HISTORICAL_EPOCHS)
TRAIN_ORIGIN = 2000         # all folds train from this year


class FocalLoss(nn.Module):
    """Focal loss: down-weights easy examples so the model focuses on hard ones."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def _date_range(year_start: int, year_end_inclusive: int) -> tuple[str, str]:
    """Return (start_date, end_date) strings covering full calendar years."""
    return f"{year_start}-01-01", f"{year_end_inclusive}-12-31"


class WalkForwardTrainer:
    """Walk-forward cross-validation trainer.

    Produces one NCPTradingModelV5 per fold.  The model from the final fold
    (trained on 2000–2023, validated on 2024) is returned by ``train()``.
    """

    # ── public API ──────────────────────────────────────────────────────────

    def train(
        self,
        tickers: list[str],
        checkpoint_fn=None,
    ) -> NCPTradingModelV5:
        """Run all walk-forward folds and return the final fold model.

        ``checkpoint_fn(model, fold, epoch)`` is called after each epoch when
        provided.  ``fold`` is 1-indexed.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("WalkForwardTrainer | device: %s | tickers: %d", device, len(tickers))

        # ── Pre-fetch all data once ────────────────────────────────────────
        all_feat_dfs, all_close, macro_df, spy_5d_fwd, spy_3d_cum, vix_series = \
            self._fetch_all_data(tickers, device)

        # ── Build cross-sectional panels (uses entire date range) ─────────
        returns_panel = pd.DataFrame(
            {t: fd["returns_20d"] for t, fd in all_feat_dfs.items()}
        )
        rank_panel = returns_panel.rank(axis=1, pct=True).fillna(0.5)

        # Sector average 20d return panel (for sector_rel_momentum feature)
        sector_ret_panel = self._compute_sector_ret_panel(returns_panel)

        fwd_excess = self._compute_excess_returns(all_feat_dfs, all_close, spy_5d_fwd)
        excess_df = pd.DataFrame(fwd_excess)
        q_low_full = excess_df.quantile(config.QUARTILE_THRESHOLD, axis=1)
        q_high_full = excess_df.quantile(1 - config.QUARTILE_THRESHOLD, axis=1)

        # 20d forward excess return for auxiliary multi-task head
        fwd_excess_20d = self._compute_excess_returns_nd(all_feat_dfs, all_close, spy_5d_fwd, n=20)
        excess_20d_df = pd.DataFrame(fwd_excess_20d)
        q_low_20d_full = excess_20d_df.quantile(config.QUARTILE_THRESHOLD, axis=1)
        q_high_20d_full = excess_20d_df.quantile(1 - config.QUARTILE_THRESHOLD, axis=1)

        ticker_to_idx = {t: i for i, t in enumerate(tickers)}

        # ── Walk-forward folds ─────────────────────────────────────────────
        # val year goes from WF_VAL_START up to and including WF_FINAL_VAL
        val_years = list(range(WF_VAL_START, WF_FINAL_VAL + 1))
        final_model = None

        for fold_num, val_year in enumerate(val_years, start=1):
            train_end_year = val_year - 1   # inclusive
            is_final = (val_year == WF_FINAL_VAL)
            n_epochs = WF_FINAL_EPOCHS if is_final else WF_FOLD_EPOCHS

            train_start, train_end = _date_range(TRAIN_ORIGIN, train_end_year)
            val_start, val_end = _date_range(val_year, val_year)

            log.info(
                "=== Fold %d/%d | train %s–%s  val %s–%s | %d epochs ===",
                fold_num, len(val_years),
                train_start, train_end, val_start, val_end, n_epochs,
            )

            # Slice samples by date
            X_tr, y_tr, y_tr_20d, idx_tr, sec_tr, dates_tr = self._build_samples(
                all_feat_dfs, rank_panel, sector_ret_panel, spy_3d_cum, fwd_excess,
                q_low_full, q_high_full, fwd_excess_20d,
                q_low_20d_full, q_high_20d_full, ticker_to_idx,
                date_start=train_start, date_end=train_end,
            )
            X_val, y_val, y_val_20d, idx_val, sec_val, dates_val = self._build_samples(
                all_feat_dfs, rank_panel, sector_ret_panel, spy_3d_cum, fwd_excess,
                q_low_full, q_high_full, fwd_excess_20d,
                q_low_20d_full, q_high_20d_full, ticker_to_idx,
                date_start=val_start, date_end=val_end,
            )

            if len(X_tr) == 0:
                log.warning("Fold %d: no training samples, skipping", fold_num)
                continue
            if len(X_val) == 0:
                log.warning("Fold %d: no validation samples, skipping", fold_num)
                continue

            log.info(
                "Fold %d: train=%d samples  val=%d samples",
                fold_num, len(X_tr), len(X_val),
            )

            model = self._train_fold(
                fold_num=fold_num,
                n_epochs=n_epochs,
                X_tr=X_tr, y_tr=y_tr, y_tr_20d=y_tr_20d,
                idx_tr=idx_tr, sec_tr=sec_tr, dates_tr=dates_tr,
                X_val=X_val, y_val=y_val, y_val_20d=y_val_20d,
                idx_val=idx_val, sec_val=sec_val,
                tickers=tickers,
                device=device,
                checkpoint_fn=checkpoint_fn,
            )

            if is_final:
                final_model = model

        if final_model is None:
            raise RuntimeError("No folds completed — final model not produced")

        log.info("Walk-forward training complete.  Returning final fold model.")
        return final_model

    # ── private helpers ─────────────────────────────────────────────────────

    def _fetch_all_data(
        self,
        tickers: list[str],
        device: torch.device,
    ) -> tuple[
        dict[str, pd.DataFrame],
        dict[str, pd.Series],
        pd.DataFrame | None,
        pd.Series | None,
        pd.Series | None,
        pd.Series | None,
    ]:
        """Download and engineer features for the full date range 2000–2024."""
        import yfinance as yf

        start_date = f"{TRAIN_ORIGIN}-01-01"
        end_date = f"{WF_FINAL_VAL}-12-31"

        # ── Macro ─────────────────────────────────────────────────────────
        log.info("Fetching macro history (^VIX, ^TNX, ^IRX, SPY)…")
        macro_df = None
        spy_5d_fwd = None
        spy_3d_cum = None
        vix_series = None
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
            spy_5d_fwd = mc["SPY"].pct_change(5).shift(-5)
            macro_df = mc[["vix", "yield_curve_slope", "spy_1d_return"]].ffill().fillna(0.0)
            spy_3d_cum = macro_df["spy_1d_return"].rolling(3).sum().fillna(0.0)
            vix_series = macro_df["vix"]
            log.info("Macro loaded: %d rows", len(macro_df))
        except Exception as exc:
            log.warning("Macro fetch failed: %s — using stubs", exc)

        # ── Per-ticker OHLCV + features ───────────────────────────────────
        log.info("Downloading OHLCV and building features (%d tickers)…", len(tickers))
        engineer = FeatureEngineer()
        all_feat_dfs: dict[str, pd.DataFrame] = {}
        all_close: dict[str, pd.Series] = {}

        for batch_start in range(0, len(tickers), 100):
            batch = tickers[batch_start: batch_start + 100]
            log.info(
                "Downloading batch %d–%d / %d",
                batch_start + 1, batch_start + len(batch), len(tickers),
            )
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

                    feat_df = engineer._stock_features(df, ticker=ticker, vix_series=vix_series)
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

        log.info("Feature DFs collected: %d tickers", len(all_feat_dfs))
        if not all_feat_dfs:
            raise RuntimeError("No feature data collected")

        return all_feat_dfs, all_close, macro_df, spy_5d_fwd, spy_3d_cum, vix_series

    def _compute_excess_returns(
        self,
        all_feat_dfs: dict[str, pd.DataFrame],
        all_close: dict[str, pd.Series],
        spy_5d_fwd: pd.Series | None,
    ) -> dict[str, pd.Series]:
        """Compute 5-day forward excess return over SPY for each ticker."""
        fwd_excess: dict[str, pd.Series] = {}
        for ticker, feat_df in all_feat_dfs.items():
            close = all_close[ticker].reindex(feat_df.index)
            stock_5d = close.pct_change(5).shift(-5)
            if spy_5d_fwd is not None:
                spy_aligned = spy_5d_fwd.reindex(feat_df.index, method="ffill").fillna(0.0)
                fwd_excess[ticker] = stock_5d - spy_aligned
            else:
                fwd_excess[ticker] = stock_5d
        return fwd_excess

    def _compute_excess_returns_nd(
        self,
        all_feat_dfs: dict[str, pd.DataFrame],
        all_close: dict[str, pd.Series],
        spy_5d_fwd: pd.Series | None,
        n: int = 20,
    ) -> dict[str, pd.Series]:
        """Compute n-day forward return (no SPY adjustment) for auxiliary labels."""
        fwd: dict[str, pd.Series] = {}
        for ticker, feat_df in all_feat_dfs.items():
            close = all_close[ticker].reindex(feat_df.index)
            fwd[ticker] = close.pct_change(n).shift(-n)
        return fwd

    def _compute_sector_ret_panel(
        self,
        returns_panel: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute per-sector mean 20d return, aligned to the full date index.

        Returns a DataFrame indexed by date, columns = sector int (0–12),
        values = mean returns_20d of all tickers in that sector.
        """
        sector_cols: dict[int, list] = {}
        for ticker in returns_panel.columns:
            s = config.TICKER_SECTOR.get(ticker, 12)
            sector_cols.setdefault(s, []).append(ticker)
        rows = {}
        for s, tickers_s in sector_cols.items():
            rows[s] = returns_panel[tickers_s].mean(axis=1)
        return pd.DataFrame(rows).ffill().fillna(0.0)

    def _build_samples(
        self,
        all_feat_dfs: dict[str, pd.DataFrame],
        rank_panel: pd.DataFrame,
        sector_ret_panel: pd.DataFrame,
        spy_3d_cum: pd.Series | None,
        fwd_excess: dict[str, pd.Series],
        q_low_full: pd.Series,
        q_high_full: pd.Series,
        fwd_excess_20d: dict[str, pd.Series],
        q_low_20d_full: pd.Series,
        q_high_20d_full: pd.Series,
        ticker_to_idx: dict[str, int],
        date_start: str,
        date_end: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build (X, y_5d, y_20d, ticker_idx, sector_idx, dates_ordinal) arrays.

        dates_ordinal is days since 2000-01-01 — used by the trainer to compute
        exponential recency weights.
        """
        start_ts = pd.Timestamp(date_start)
        end_ts = pd.Timestamp(date_end)
        epoch = pd.Timestamp("2000-01-01")

        # Restrict quartile panels to dates in the window
        mask = (q_low_full.index >= start_ts) & (q_low_full.index <= end_ts)
        q_low = q_low_full[mask]
        q_high = q_high_full[mask]
        valid_dates = set(q_low.index)

        mask_20d = (q_low_20d_full.index >= start_ts) & (q_low_20d_full.index <= end_ts)
        q_low_20d = q_low_20d_full[mask_20d]
        q_high_20d = q_high_20d_full[mask_20d]

        all_X: list[np.ndarray] = []
        all_y: list[int] = []
        all_y_20d: list[int] = []
        all_idx: list[int] = []
        all_sector: list[int] = []
        all_dates: list[int] = []

        for ticker, feat_df in all_feat_dfs.items():
            try:
                feat_df = feat_df.copy()

                if ticker in rank_panel.columns:
                    feat_df["momentum_rank"] = (
                        rank_panel[ticker].reindex(feat_df.index).ffill().fillna(0.5).values
                    )
                else:
                    feat_df["momentum_rank"] = 0.5

                stock_3d = feat_df["returns_1d"].rolling(3).sum().fillna(0.0)
                if spy_3d_cum is not None:
                    spy_aligned = spy_3d_cum.reindex(feat_df.index, method="ffill").fillna(0.0)
                    abnormal = stock_3d - spy_aligned
                else:
                    abnormal = stock_3d
                feat_df["sentiment_3d"] = abnormal.clip(-0.15, 0.15) / 0.15

                # sector_rel_momentum: stock 20d return minus sector mean 20d return
                sector_idx = config.TICKER_SECTOR.get(ticker, 12)
                if sector_idx in sector_ret_panel.columns:
                    sec_avg = sector_ret_panel[sector_idx].reindex(feat_df.index, method="ffill").fillna(0.0)
                    feat_df["sector_rel_momentum"] = (feat_df["returns_20d"] - sec_avg).fillna(0.0).values
                else:
                    feat_df["sector_rel_momentum"] = 0.0

                feat_arr = feat_df[FEATURE_NAMES].values.astype(np.float32)
                feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)

                ticker_idx = ticker_to_idx.get(ticker, 0)
                ticker_excess = fwd_excess.get(ticker)
                ticker_excess_20d = fwd_excess_20d.get(ticker)

                for i in range(config.SEQUENCE_LENGTH, len(feat_arr) - config.RETURN_HORIZON):
                    if ticker_excess is None:
                        continue
                    date = feat_df.index[i]
                    if date not in valid_dates:
                        continue
                    exc_ret = ticker_excess.iloc[i] if i < len(ticker_excess) else np.nan
                    if pd.isna(exc_ret):
                        continue
                    ql = q_low.loc[date]
                    qh = q_high.loc[date]
                    if pd.isna(ql) or pd.isna(qh):
                        continue
                    if exc_ret >= qh:
                        label = 1
                    elif exc_ret <= ql:
                        label = 0
                    else:
                        continue

                    # 20d auxiliary label (soft: -1 = skip in aux loss)
                    label_20d = -1
                    if ticker_excess_20d is not None and date in q_low_20d.index:
                        exc_20d = ticker_excess_20d.iloc[i] if i < len(ticker_excess_20d) else np.nan
                        if not pd.isna(exc_20d):
                            ql_20d = q_low_20d.loc[date]
                            qh_20d = q_high_20d.loc[date]
                            if not (pd.isna(ql_20d) or pd.isna(qh_20d)):
                                if exc_20d >= qh_20d:
                                    label_20d = 1
                                elif exc_20d <= ql_20d:
                                    label_20d = 0

                    all_X.append(feat_arr[i - config.SEQUENCE_LENGTH: i])
                    all_y.append(label)
                    all_y_20d.append(label_20d)
                    all_idx.append(ticker_idx)
                    all_sector.append(sector_idx)
                    all_dates.append((date - epoch).days)

            except Exception as exc:
                log.warning("Skipping %s in sample build: %s", ticker, exc)

        if not all_X:
            empty = np.empty((0,))
            return empty, empty, empty, empty, empty, empty

        return (
            np.array(all_X, dtype=np.float32),
            np.array(all_y, dtype=np.int64),
            np.array(all_y_20d, dtype=np.int64),
            np.array(all_idx, dtype=np.int64),
            np.array(all_sector, dtype=np.int64),
            np.array(all_dates, dtype=np.float32),
        )

    def _train_fold(
        self,
        fold_num: int,
        n_epochs: int,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        y_tr_20d: np.ndarray,
        idx_tr: np.ndarray,
        sec_tr: np.ndarray,
        dates_tr: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        y_val_20d: np.ndarray,
        idx_val: np.ndarray,
        sec_val: np.ndarray,
        tickers: list[str],
        device: torch.device,
        checkpoint_fn=None,
    ) -> NCPTradingModelV5:
        """Train a fresh model for one fold, return the best-val-acc checkpoint."""

        # ── Recency weights ───────────────────────────────────────────────
        # exp(scale * days_since_epoch), normalised so mean=1.
        recency_w = np.exp(config.RECENCY_SCALE * dates_tr).astype(np.float32)
        recency_w = recency_w / recency_w.mean()

        # ── Move data to device ───────────────────────────────────────────
        X_t      = torch.FloatTensor(X_tr).to(device)
        y_t      = torch.LongTensor(y_tr).to(device)
        y_t_20d  = torch.LongTensor(y_tr_20d).to(device)
        idx_t    = torch.LongTensor(idx_tr).to(device)
        sec_t    = torch.LongTensor(sec_tr).to(device)
        w_t      = torch.FloatTensor(recency_w).to(device)

        X_v      = torch.FloatTensor(X_val).to(device)
        y_v      = torch.LongTensor(y_val).to(device)
        idx_v    = torch.LongTensor(idx_val).to(device)
        sec_v    = torch.LongTensor(sec_val).to(device)

        log.info(
            "Fold %d | train %.2f GB | val %.2f GB",
            fold_num,
            X_t.element_size() * X_t.nelement() / 1e9,
            X_v.element_size() * X_v.nelement() / 1e9,
        )

        train_ds = TensorDataset(X_t, idx_t, sec_t, y_t, y_t_20d, w_t)
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)

        val_ds = TensorDataset(X_v, idx_v, sec_v, y_v)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

        # ── Class weights & loss ──────────────────────────────────────────
        counts = np.bincount(y_tr, minlength=2).astype(np.float32)
        log.info(
            "Fold %d class dist — underperform: %d | outperform: %d",
            fold_num, int(counts[0]), int(counts[1]),
        )
        weights_np = 1.0 / counts.clip(min=1)
        weights_np = weights_np / weights_np.mean()
        class_weights = torch.FloatTensor(weights_np).to(device)
        criterion = FocalLoss(gamma=config.FOCAL_GAMMA, weight=class_weights)

        # ── Model ─────────────────────────────────────────────────────────
        model = NCPTradingModelV5(
            num_stocks=len(tickers),
            num_features=config.NUM_FEATURES,
            input_size=config.INPUT_SIZE,
            ncp_units=config.NCP_UNITS,
            ncp_output_size=config.NCP_OUTPUT_SIZE,
            ncp_sparsity=config.NCP_SPARSITY,
            embedding_dim=config.EMBEDDING_DIM,
            num_sectors=config.NUM_SECTORS,
            sector_embedding_dim=config.SECTOR_EMBEDDING_DIM,
            cs_heads=4,
            cs_dropout=0.1,
            dropout=config.DROPOUT,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
        )
        for pg in optimizer.param_groups:
            pg["initial_lr"] = config.LEARNING_RATE
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.SGDR_T0, T_mult=config.SGDR_T_MULT
        )

        # ── Training loop with best-val-acc checkpointing ─────────────────
        best_val_acc = -1.0

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            best_ckpt_path = tmp.name

        try:
            for epoch in range(n_epochs):
                model.train()
                total_loss, correct, n = 0.0, 0, 0
                for xb, ib, sb, yb, yb_20d, wb in train_loader:
                    optimizer.zero_grad()

                    # ── Mixup augmentation ────────────────────────────────
                    lam = float(np.random.beta(config.MIXUP_ALPHA, config.MIXUP_ALPHA))
                    perm = torch.randperm(xb.size(0), device=device)
                    xb_mix = lam * xb + (1 - lam) * xb[perm]
                    ib_mix = ib   # embeddings: use dominant sample's index
                    sb_mix = sb

                    logits_5d, logits_20d = model(xb_mix, ib_mix, sb_mix)

                    # Primary focal loss with recency weighting + mixup
                    ce_a = F.cross_entropy(logits_5d, yb, weight=class_weights, reduction="none")
                    ce_b = F.cross_entropy(logits_5d, yb[perm], weight=class_weights, reduction="none")
                    pt_a = torch.exp(-ce_a)
                    pt_b = torch.exp(-ce_b)
                    focal_a = (1 - pt_a) ** config.FOCAL_GAMMA * ce_a
                    focal_b = (1 - pt_b) ** config.FOCAL_GAMMA * ce_b
                    loss_primary = (wb * (lam * focal_a + (1 - lam) * focal_b)).mean()

                    # Auxiliary 20d head loss (only on labelled samples, no mixup)
                    mask_20d = yb_20d >= 0
                    if mask_20d.any():
                        loss_aux = criterion(logits_20d[mask_20d], yb_20d[mask_20d])
                    else:
                        loss_aux = torch.tensor(0.0, device=device)

                    loss = loss_primary + config.MULTITASK_AUX_WEIGHT * loss_aux

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss.item() * len(yb)
                    # Accuracy on un-mixed labels (dominant assignment)
                    correct += (logits_5d.argmax(1) == yb).sum().item()
                    n += len(yb)

                scheduler.step()
                train_acc = correct / n if n > 0 else 0.0
                train_loss = total_loss / n if n > 0 else 0.0

                # Validate (primary head only)
                model.eval()
                val_correct, val_n = 0, 0
                with torch.no_grad():
                    for xb, ib, sb, yb in val_loader:
                        logits_5d, _ = model(xb, ib, sb)
                        val_correct += (logits_5d.argmax(1) == yb).sum().item()
                        val_n += len(yb)
                val_acc = val_correct / val_n if val_n > 0 else 0.0

                log.info(
                    "Fold %d | Epoch %d/%d | loss=%.4f | train_acc=%.4f | val_acc=%.4f | lr=%.2e",
                    fold_num, epoch + 1, n_epochs,
                    train_loss, train_acc, val_acc,
                    scheduler.get_last_lr()[0],
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), best_ckpt_path)
                    log.info(
                        "Fold %d | New best val_acc=%.4f at epoch %d — checkpoint saved",
                        fold_num, best_val_acc, epoch + 1,
                    )

                if checkpoint_fn is not None:
                    checkpoint_fn(model, fold_num, epoch + 1)

            log.info(
                "Fold %d complete | best_val_acc=%.4f — restoring best checkpoint",
                fold_num, best_val_acc,
            )
            model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

        finally:
            try:
                os.unlink(best_ckpt_path)
            except OSError:
                pass

        return model
