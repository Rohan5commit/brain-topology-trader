"""Feature engineering: 29 features per stock per day.

Feature index reference
-----------------------
 0  returns_1d
 1  returns_5d
 2  returns_20d
 3  volume_zscore_20d
 4  obv_norm
 5  rsi_14
 6  macd_line
 7  macd_signal
 8  roc_10
 9  atr_14
10  bollinger_width
11  vix
12  yield_curve_slope
13  spy_1d_return
14  sentiment_3d
15  momentum_rank
16  price_to_sma20
17  dist_52w_high
18  dist_52w_low
19  volume_trend
20  roc_60
21  volatility_ratio
22  earnings_surprise       — normalised EPS surprise %, clipped ±3 σ
23  short_vol_ratio         — short volume fraction of total volume
24  pc_ratio                — put/call OI ratio (stock-level snapshot)
25  days_since_earnings_norm — calendar days since last earnings / 91, clipped [0,4]
26  daily_range_norm        — (high-low)/ATR, intraday volatility signal
27  vix_zscore_252          — VIX z-score over trailing 252 days (market regime)
28  sector_rel_momentum     — stock 20d return minus sector avg 20d return
"""
import logging
import os

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FEATURE_NAMES = [
    "returns_1d",          # 0
    "returns_5d",          # 1
    "returns_20d",         # 2
    "volume_zscore_20d",   # 3
    "obv_norm",            # 4
    "rsi_14",              # 5
    "macd_line",           # 6
    "macd_signal",         # 7
    "roc_10",              # 8
    "atr_14",              # 9
    "bollinger_width",     # 10
    "vix",                 # 11
    "yield_curve_slope",   # 12
    "spy_1d_return",       # 13
    "sentiment_3d",        # 14
    "momentum_rank",       # 15
    "price_to_sma20",      # 16
    "dist_52w_high",       # 17  — distance from 52-week high (mean reversion)
    "dist_52w_low",        # 18  — distance from 52-week low  (support signal)
    "volume_trend",        # 19  — vol MA20/MA60 ratio (institutional activity)
    "roc_60",              # 20  — 60-day momentum (intermediate trend)
    "volatility_ratio",    # 21  — current ATR vs 60-day avg (vol regime)
    "earnings_surprise",        # 22  — normalised EPS surprise %, clipped ±3 σ
    "short_vol_ratio",          # 23  — short volume fraction of total volume
    "pc_ratio",                 # 24  — put/call OI ratio (stock-level snapshot)
    "days_since_earnings_norm", # 25  — days since last earnings / 91, clipped [0,4]
    "daily_range_norm",         # 26  — (high-low)/ATR, intraday vol signal
    "vix_zscore_252",           # 27  — VIX z-score over 252 days
    "sector_rel_momentum",      # 28  — stock 20d return minus sector avg 20d return
]
assert len(FEATURE_NAMES) == 29

# ── Alt-data file paths ───────────────────────────────────────────────────────
_ALT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "alt_data")
_EARNINGS_PATH = os.path.join(_ALT_DATA_DIR, "earnings_surprises.parquet")
_SHORT_INT_PATH = os.path.join(_ALT_DATA_DIR, "short_interest.parquet")
_OPTIONS_PATH  = os.path.join(_ALT_DATA_DIR, "options_snapshot.parquet")


def _load_parquet_safe(path: str, label: str) -> pd.DataFrame | None:
    """Load a parquet file, returning None and logging a warning on failure."""
    try:
        df = pd.read_parquet(path)
        log.info("Alt-data loaded: %s (%d rows)", label, len(df))
        return df
    except FileNotFoundError:
        log.info("Alt-data not found (will use defaults): %s", path)
        return None
    except Exception as exc:
        log.warning("Alt-data load error for %s: %s — using defaults", label, exc)
        return None


class FeatureEngineer:
    """Computes per-stock, per-day feature vectors (29 features).

    Alt-data parquet files are loaded once at ``__init__`` time.  If any file
    is missing or unreadable the corresponding feature column is filled with
    its default value so the rest of the pipeline is unaffected.

    Features 25-28 require additional context passed by the caller:
      - ``sector_ret20`` (dict[str, float]): sector-average 20d return, keyed by
        GICS sector int. Computed externally from the cross-sectional panel and
        passed into ``_stock_features``.  Default 0.0 if missing.
      - ``vix_series`` (pd.Series): full VIX daily history for computing the
        252-day z-score.  Passed into ``_stock_features``.  Default 0.0 if missing.

    Alt-data contracts
    ------------------
    earnings_surprises.parquet
        Columns: ``ticker`` (str), ``date`` (datetime-like), ``surprise_pct`` (float).
        ``surprise_pct`` is the raw EPS surprise as a percentage; it will be
        normalised (zero-mean, unit-std over the whole dataset) and clipped to
        ±3 σ inside this class.

    short_interest.parquet
        Columns: ``ticker`` (str), ``short_vol_ratio`` (float).
        Static per-ticker snapshot; ``short_vol_ratio`` clipped to [0, 1].

    options_snapshot.parquet
        Columns: ``ticker`` (str), ``pc_ratio`` (float).
        A static (single-row-per-ticker) snapshot of put/call OI ratios.
        Clipped to [0, 3], default 1.0.
    """

    def __init__(self) -> None:
        # ── earnings_surprise ──────────────────────────────────────────────
        # Build a (ticker, date) -> normalised_surprise lookup.
        # We keep it as a dict[str, pd.Series(date -> float)] for fast access.
        self._earnings: dict[str, pd.Series] = {}
        raw_earn = _load_parquet_safe(_EARNINGS_PATH, "earnings_surprises")
        if raw_earn is not None:
            try:
                raw_earn = raw_earn.copy()
                raw_earn["date"] = pd.to_datetime(raw_earn["date"]).dt.normalize()
                # Normalise surprise_pct globally (z-score) then clip to ±3 σ
                mu = raw_earn["surprise_pct"].mean()
                sigma = raw_earn["surprise_pct"].std()
                if sigma == 0 or pd.isna(sigma):
                    sigma = 1.0
                raw_earn["surprise_norm"] = ((raw_earn["surprise_pct"] - mu) / sigma).clip(-3.0, 3.0)
                for ticker, grp in raw_earn.groupby("ticker"):
                    s = grp.set_index("date")["surprise_norm"].sort_index()
                    self._earnings[ticker] = s
                log.info("earnings_surprise: %d tickers loaded", len(self._earnings))
            except Exception as exc:
                log.warning("Failed to process earnings_surprises: %s", exc)
                self._earnings = {}

        # ── short_vol_ratio ────────────────────────────────────────────────
        # Static per-ticker scalar from yfinance snapshot (shortPercentOfFloat).
        self._short_vol: dict[str, float] = {}
        raw_short = _load_parquet_safe(_SHORT_INT_PATH, "short_interest")
        if raw_short is not None:
            try:
                raw_short = raw_short.copy()
                if "short_vol_ratio" in raw_short.columns:
                    self._short_vol = dict(zip(
                        raw_short["ticker"],
                        raw_short["short_vol_ratio"].clip(0.0, 1.0).astype(float)
                    ))
                log.info("short_vol_ratio: %d tickers loaded", len(self._short_vol))
            except Exception as exc:
                log.warning("Failed to process short_interest: %s", exc)
                self._short_vol = {}

        # ── pc_ratio ──────────────────────────────────────────────────────
        # Static per-ticker scalar: dict[str, float].
        self._pc_ratio: dict[str, float] = {}
        raw_opt = _load_parquet_safe(_OPTIONS_PATH, "options_snapshot")
        if raw_opt is not None:
            try:
                raw_opt = raw_opt.copy()
                raw_opt["pc_ratio"] = raw_opt["pc_ratio"].clip(0.0, 3.0)
                self._pc_ratio = dict(zip(raw_opt["ticker"], raw_opt["pc_ratio"].astype(float)))
                log.info("pc_ratio: %d tickers loaded", len(self._pc_ratio))
            except Exception as exc:
                log.warning("Failed to process options_snapshot: %s", exc)
                self._pc_ratio = {}

    def compute_features(
        self,
        ohlcv: dict[str, pd.DataFrame],
        macro: dict[str, float],
        sentiment: dict[str, float],
        ticker_sector: dict[str, int] | None = None,
        vix_series: pd.Series | None = None,
    ) -> dict[str, np.ndarray | None]:
        """Return {ticker: ndarray(n_days, 29)} for all tickers with enough data."""
        import config as _cfg

        momentum_20d: dict[str, float] = {}
        stock_features: dict[str, pd.DataFrame] = {}

        for ticker, df in ohlcv.items():
            try:
                feat = self._stock_features(df, ticker, vix_series=vix_series)
                if feat is not None and len(feat) >= 21:
                    stock_features[ticker] = feat
                    momentum_20d[ticker] = float(feat["returns_20d"].iloc[-1])
            except Exception as exc:
                log.debug("Feature error %s: %s", ticker, exc)

        mom_series = pd.Series(momentum_20d)
        ranks = mom_series.rank(pct=True)

        # Compute per-sector average 20d return for sector_rel_momentum
        sector_ret20: dict[int, float] = {}
        if ticker_sector is not None:
            sector_rets: dict[int, list] = {}
            for t, ret in momentum_20d.items():
                s = ticker_sector.get(t, 12)
                sector_rets.setdefault(s, []).append(ret)
            sector_ret20 = {s: float(np.mean(v)) for s, v in sector_rets.items()}

        vix = macro.get("vix", 20.0)
        ycs = macro.get("yield_curve_slope", 0.5)
        spy_ret = macro.get("spy_1d_return", 0.0)

        result: dict[str, np.ndarray | None] = {}
        for ticker, feat in stock_features.items():
            feat = feat.copy()
            feat["vix"] = vix
            feat["yield_curve_slope"] = ycs
            feat["spy_1d_return"] = spy_ret
            feat["sentiment_3d"] = sentiment.get(ticker, 0.0)
            feat["momentum_rank"] = float(ranks.get(ticker, 0.5))
            # sector_rel_momentum: stock's last-row value was set per-ticker in _stock_features;
            # here we patch it with the cross-sectional sector average (not available inside).
            if ticker_sector is not None:
                s = ticker_sector.get(ticker, 12)
                sec_avg = sector_ret20.get(s, 0.0)
                stock_ret20 = float(feat["returns_20d"].iloc[-1])
                feat["sector_rel_momentum"] = stock_ret20 - sec_avg
            arr = feat[FEATURE_NAMES].values.astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            result[ticker] = arr

        log.info("Features assembled for %d tickers", len(result))
        return result

    def _stock_features(
        self,
        df: pd.DataFrame,
        ticker: str = "",
        vix_series: pd.Series | None = None,
    ) -> pd.DataFrame | None:
        if len(df) < 21:
            return None
        close = df["close"]
        volume = df["volume"]
        high = df["high"]
        low = df["low"]

        feat = pd.DataFrame(index=df.index)

        feat["returns_1d"] = close.pct_change(1)
        feat["returns_5d"] = close.pct_change(5)
        feat["returns_20d"] = close.pct_change(20)

        vol_mean = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std().replace(0, 1)
        feat["volume_zscore_20d"] = (volume - vol_mean) / vol_std

        direction = np.sign(close.diff())
        obv = (direction * volume).cumsum()
        obv_std = obv.rolling(20).std().replace(0, 1)
        feat["obv_norm"] = obv / obv_std

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, 1e-9)
        feat["rsi_14"] = 100 - 100 / (1 + gain / loss)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        feat["macd_line"] = macd
        feat["macd_signal"] = macd.ewm(span=9, adjust=False).mean()

        feat["roc_10"] = close.pct_change(10) * 100

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        raw_atr = tr.rolling(14).mean()
        feat["atr_14"] = raw_atr / close.replace(0, 1)

        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        feat["bollinger_width"] = (sma20 + 2 * std20 - (sma20 - 2 * std20)) / sma20.replace(0, 1)
        feat["price_to_sma20"] = (close / sma20.replace(0, 1)) - 1

        # 52-week high/low distance
        high_252 = close.rolling(252, min_periods=20).max()
        low_252 = close.rolling(252, min_periods=20).min()
        feat["dist_52w_high"] = (close / high_252.replace(0, 1)) - 1
        feat["dist_52w_low"] = (close / low_252.replace(0, 1)) - 1

        # Volume trend: 20-day vs 60-day moving average
        vol_ma20 = volume.rolling(20).mean()
        vol_ma60 = volume.rolling(60, min_periods=20).mean().replace(0, 1)
        feat["volume_trend"] = (vol_ma20 / vol_ma60) - 1

        # 60-day rate of change
        feat["roc_60"] = close.pct_change(60) * 100

        # Volatility regime: current ATR vs 60-day average
        atr_ma60 = raw_atr.rolling(60, min_periods=14).mean().replace(0, 1)
        feat["volatility_ratio"] = (raw_atr / atr_ma60) - 1

        # Macro + cross-sectional — filled by caller
        feat["vix"] = 0.0
        feat["yield_curve_slope"] = 0.0
        feat["spy_1d_return"] = 0.0
        feat["sentiment_3d"] = 0.0
        feat["momentum_rank"] = 0.0

        # ── Alt-data features ──────────────────────────────────────────────

        # earnings_surprise (index 22)
        # For each row date, look up the most recent earnings event before that
        # date and carry it forward (as-of join).  Default: 0.0 if no data.
        feat["earnings_surprise"] = 0.0
        if ticker and ticker in self._earnings:
            earn_s = self._earnings[ticker]
            # Use merge_asof: sort feat index, align to earn_s (sorted), ffill
            feat_dates = feat.index
            aligned = pd.Series(index=feat_dates, dtype=float)
            # merge_asof needs sorted arrays
            earn_sorted = earn_s.sort_index()
            earn_df = earn_sorted.reset_index()
            earn_df.columns = ["date", "value"]
            feat_df_tmp = pd.DataFrame({"date": feat_dates})
            merged = pd.merge_asof(
                feat_df_tmp.sort_values("date"),
                earn_df.sort_values("date"),
                on="date",
                direction="backward",
            )
            merged = merged.set_index("date")["value"].reindex(feat_dates)
            feat["earnings_surprise"] = merged.fillna(0.0).values

        # short_vol_ratio (index 23)
        # Static per-ticker scalar (shortPercentOfFloat snapshot). Default: 0.5.
        feat["short_vol_ratio"] = self._short_vol.get(ticker, 0.5) if ticker else 0.5

        # pc_ratio (index 24)
        # Static per-ticker scalar.  Default: 1.0.
        feat["pc_ratio"] = self._pc_ratio.get(ticker, 1.0) if ticker else 1.0

        # days_since_earnings_norm (index 25)
        # Calendar days since last known earnings date, divided by 91 (one quarter),
        # clipped to [0, 4].  Captures pre-earnings drift and PEAD decay.
        feat["days_since_earnings_norm"] = 1.0  # default: neutral (one quarter ago)
        if ticker and ticker in self._earnings:
            earn_dates = self._earnings[ticker].index  # sorted DatetimeIndex
            feat_dates_arr = feat.index
            earn_arr = earn_dates.values
            norms = []
            for d in feat_dates_arr:
                past = earn_arr[earn_arr <= d]
                if len(past) == 0:
                    norms.append(4.0)  # no history → max cap
                else:
                    days = (d - past[-1]).days
                    norms.append(min(days / 91.0, 4.0))
            feat["days_since_earnings_norm"] = norms

        # daily_range_norm (index 26)
        # (high - low) / ATR — normalised intraday range.  Captures intraday
        # vol relative to the trailing regime; complements bollinger_width.
        feat["daily_range_norm"] = ((high - low) / raw_atr.replace(0, 1)).clip(0, 5)

        # vix_zscore_252 (index 27)
        # VIX z-score over the trailing 252 trading days.
        # Uses the vix_series passed by the walk-forward trainer (full history).
        # If not available, falls back to 0.0 (filled by caller later).
        feat["vix_zscore_252"] = 0.0
        if vix_series is not None:
            aligned_vix = vix_series.reindex(feat.index, method="ffill").fillna(20.0)
            vix_mu = aligned_vix.rolling(252, min_periods=20).mean()
            vix_std = aligned_vix.rolling(252, min_periods=20).std().replace(0, 1)
            feat["vix_zscore_252"] = ((aligned_vix - vix_mu) / vix_std).clip(-3, 3).fillna(0.0).values

        # sector_rel_momentum (index 28)
        # Stock 20d return minus sector average 20d return.
        # This column is a placeholder set to 0.0 here; compute_features() and the
        # walk-forward trainer patch it with the real cross-sectional value.
        feat["sector_rel_momentum"] = 0.0

        return feat.dropna(subset=["returns_20d"])
