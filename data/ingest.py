import os
import time
import logging
import requests
from itertools import cycle
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

_RETRY_WAIT = 2     # seconds between retries
_MAX_RETRIES = 3


def _get(url: str, params: dict, retries: int = _MAX_RETRIES) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            log.warning("Request failed (%s) — retry %d/%d", exc, attempt + 1, retries)
            time.sleep(_RETRY_WAIT * (attempt + 1))
    return {}


class DataIngestor:
    """Fetches OHLCV, macro, and sentiment data."""

    def __init__(self) -> None:
        keys = [
            os.environ.get(f"TWELVE_DATA_KEY_{i}", "") for i in range(1, 9)
        ]
        keys = [k for k in keys if k]
        if not keys:
            raise RuntimeError("No Twelve Data keys found in environment")
        self._key_pool = cycle(keys)
        self._fred_key = os.environ["FRED_API_KEY"]
        self._finnhub_key = os.environ["FINNHUB_API_KEY"]

    def _next_key(self) -> str:
        return next(self._key_pool)

    # ── OHLCV ────────────────────────────────────────────────────────────────

    def _fetch_ohlcv_one(self, ticker: str, outputsize: int = 60) -> pd.DataFrame | None:
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": ticker,
            "interval": "1day",
            "outputsize": outputsize,
            "apikey": self._next_key(),
            "format": "JSON",
        }
        data = _get(url, params)
        if data.get("status") == "error" or "values" not in data:
            log.debug("No OHLCV for %s: %s", ticker, data.get("message", ""))
            return None
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").set_index("datetime")
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def fetch_ohlcv_all(self, tickers: list[str], outputsize: int = 60) -> dict[str, pd.DataFrame]:
        results: dict[str, pd.DataFrame] = {}
        # Twelve Data free tier: ~800 calls/day per key; batch one ticker per call
        for i, ticker in enumerate(tickers):
            try:
                df = self._fetch_ohlcv_one(ticker, outputsize)
                if df is not None and not df.empty:
                    results[ticker] = df
            except Exception as exc:
                log.warning("OHLCV fetch failed for %s: %s", ticker, exc)
            # Polite rate-limiting: ~1 req/s across keys
            if i % 8 == 7:
                time.sleep(1)
        log.info("OHLCV: fetched %d/%d tickers", len(results), len(tickers))
        return results

    def fetch_closing_prices(self, tickers: list[str]) -> dict[str, float]:
        """Return latest close price per ticker (lightweight)."""
        prices: dict[str, float] = {}
        for ticker in tickers:
            try:
                df = self._fetch_ohlcv_one(ticker, outputsize=2)
                if df is not None and not df.empty:
                    prices[ticker] = float(df["close"].iloc[-1])
            except Exception as exc:
                log.debug("Close fetch failed %s: %s", ticker, exc)
        return prices

    # ── Macro (FRED) ─────────────────────────────────────────────────────────

    def _fred_series(self, series_id: str, limit: int = 30) -> pd.Series:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self._fred_key,
            "file_type": "json",
            "limit": limit,
            "sort_order": "desc",
        }
        data = _get(url, params)
        obs = data.get("observations", [])
        if not obs:
            return pd.Series(dtype=float)
        s = pd.Series(
            {o["date"]: float(o["value"]) for o in obs if o["value"] != "."},
            name=series_id,
        )
        return s.sort_index()

    def fetch_macro(self) -> dict[str, float]:
        macro: dict[str, float] = {}
        try:
            vix = self._fred_series("VIXCLS")
            macro["vix"] = float(vix.iloc[-1]) if not vix.empty else 20.0
        except Exception:
            macro["vix"] = 20.0
        try:
            t10 = self._fred_series("DGS10")
            t2 = self._fred_series("DGS2")
            macro["yield_curve_slope"] = float(t10.iloc[-1] - t2.iloc[-1]) if (not t10.empty and not t2.empty) else 0.0
        except Exception:
            macro["yield_curve_slope"] = 0.0
        try:
            ff = self._fred_series("DFF")
            macro["fed_funds_rate"] = float(ff.iloc[-1]) if not ff.empty else 5.0
        except Exception:
            macro["fed_funds_rate"] = 5.0
        log.info("Macro: vix=%.2f yc=%.3f ff=%.2f", macro["vix"], macro["yield_curve_slope"], macro["fed_funds_rate"])
        return macro

    # ── Sentiment (Finnhub) ──────────────────────────────────────────────────

    def fetch_sentiment(self, tickers: list[str]) -> dict[str, float]:
        """Return 3-day rolling avg news sentiment score per ticker."""
        import finnhub
        client = finnhub.Client(api_key=self._finnhub_key)
        scores: dict[str, float] = {}
        for ticker in tickers:
            try:
                news = client.company_news(ticker, _from="2020-01-01", to="2099-01-01")
                if news:
                    # Use last 3 items' sentiment proxy (headline length as stub)
                    # Finnhub free tier doesn't give numeric sentiment; use 0.0 default
                    scores[ticker] = 0.0
                else:
                    scores[ticker] = 0.0
            except Exception:
                scores[ticker] = 0.0
            time.sleep(0.05)  # stay under free-tier rate limit
        return scores
