import os
import time
import logging
import requests
from itertools import cycle

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

_RETRY_WAIT = 2
_MAX_RETRIES = 3

_MACRO_TICKERS = ["^VIX", "^TNX", "^IRX", "SPY", "QQQ"]


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
    """Fetches OHLCV, macro (yfinance), and sentiment data."""

    def __init__(self) -> None:
        keys = [os.environ.get(f"TWELVE_DATA_KEY_{i}", "") for i in range(1, 9)]
        keys = [k for k in keys if k]
        if not keys:
            raise RuntimeError("No Twelve Data keys found in environment")
        self._key_pool = cycle(keys)
        self._finnhub_key = os.environ.get("FINNHUB_API_KEY", "")

    def _next_key(self) -> str:
        return next(self._key_pool)

    # ── OHLCV (Twelve Data) ──────────────────────────────────────────────────

    def _fetch_ohlcv_one(self, ticker: str, outputsize: int = 150) -> pd.DataFrame | None:
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

    def fetch_ohlcv_all(self, tickers: list[str], outputsize: int = 150) -> dict[str, pd.DataFrame]:
        results: dict[str, pd.DataFrame] = {}
        for i, ticker in enumerate(tickers):
            try:
                df = self._fetch_ohlcv_one(ticker, outputsize)
                if df is not None and not df.empty:
                    results[ticker] = df
            except Exception as exc:
                log.warning("OHLCV fetch failed for %s: %s", ticker, exc)
            if i % 8 == 7:  # throttle: 1s pause every 8 tickers across 8 rotating keys
                time.sleep(1)
        log.info("OHLCV: fetched %d/%d tickers", len(results), len(tickers))
        return results

    def fetch_closing_prices(self, tickers: list[str]) -> dict[str, float]:
        prices: dict[str, float] = {}
        for ticker in tickers:
            try:
                df = self._fetch_ohlcv_one(ticker, outputsize=2)
                if df is not None and not df.empty:
                    prices[ticker] = float(df["close"].iloc[-1])
            except Exception as exc:
                log.debug("Close fetch failed %s: %s", ticker, exc)
        return prices

    # ── Macro (yfinance — no API key) ────────────────────────────────────────

    def fetch_macro(self) -> dict[str, float]:
        """
        Fetch macro via yfinance:
          ^VIX  → VIX level
          ^TNX  → 10Y Treasury yield
          ^IRX  → 2Y Treasury yield (13-week, closest proxy)
          SPY   → S&P 500 1-day return (market breadth)
          QQQ   → NASDAQ 1-day return (tech breadth)

        yield_curve_slope = ^TNX close - ^IRX close
        """
        defaults = {"vix": 20.0, "yield_curve_slope": 0.5, "spy_1d_return": 0.0}
        try:
            raw = yf.download(
                _MACRO_TICKERS,
                period="5d",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

            def _last(col: str) -> float:
                s = closes[col].dropna()
                return float(s.iloc[-1]) if not s.empty else float("nan")

            vix = _last("^VIX")
            tnx = _last("^TNX")
            irx = _last("^IRX")
            spy_ret = float(closes["SPY"].pct_change().dropna().iloc[-1]) if "SPY" in closes else 0.0

            macro = {
                "vix": vix if not (vix != vix) else 20.0,
                "yield_curve_slope": (tnx - irx) if not ((tnx != tnx) or (irx != irx)) else 0.5,
                "spy_1d_return": spy_ret,
            }
        except Exception as exc:
            log.warning("Macro yfinance fetch failed: %s — using defaults", exc)
            macro = defaults

        log.info("Macro: vix=%.2f yc_slope=%.3f spy_1d=%.4f",
                 macro["vix"], macro["yield_curve_slope"], macro["spy_1d_return"])
        return macro

    # ── Sentiment (Finnhub) ──────────────────────────────────────────────────

    def fetch_sentiment(self, tickers: list[str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        if not self._finnhub_key:
            return {t: 0.0 for t in tickers}
        try:
            import finnhub
            client = finnhub.Client(api_key=self._finnhub_key)
            for ticker in tickers:
                try:
                    client.company_news(ticker, _from="2020-01-01", to="2099-01-01")
                    scores[ticker] = 0.0
                except Exception:
                    scores[ticker] = 0.0
                time.sleep(0.05)
        except Exception as exc:
            log.warning("Finnhub client error: %s", exc)
            scores = {t: 0.0 for t in tickers}
        return scores
