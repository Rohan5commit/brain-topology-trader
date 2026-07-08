"""Signal smoothing, confidence thresholding, and trade ranking — v3 binary (down/up)."""
import logging
import os
from collections import deque

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

# Class indices for binary output
DOWN, UP = 0, 1


class SignalProcessor:
    def __init__(self) -> None:
        self._history: dict[str, deque] = {}

    def _load_history(self) -> dict[str, list[list[float]]]:
        """Load last N days of raw signals from Modal Volume parquet."""
        if not os.path.exists(config.SIGNALS_PATH):
            return {}
        df = pd.read_parquet(config.SIGNALS_PATH)
        history: dict[str, list[list[float]]] = {}
        for ticker in df.index:
            row = df.loc[ticker]
            history[ticker] = []
            for day in range(config.SIGNAL_SMOOTH_DAYS):
                col_p = f"day{day}_probs"
                if col_p in row.index and isinstance(row[col_p], (list, __import__('numpy').ndarray)):
                    history[ticker].append(row[col_p])
        return history

    def smooth_and_rank(
        self,
        raw_signals: dict[str, list[float]],  # {ticker: [p_down, p_up]}
    ) -> dict[str, dict]:
        """
        Merge today's raw signals with the last N-1 days from Volume.
        Returns {ticker: {side, score, confidence, p_up, p_down}}.
        """
        history = self._load_history()
        smoothed: dict[str, dict] = {}

        for ticker, probs in raw_signals.items():
            past = history.get(ticker, [])
            window = past[-(config.SIGNAL_SMOOTH_DAYS - 1):] + [probs]
            arr = np.array(window)              # (<=3, 2)
            mean_probs = arr.mean(axis=0)       # (2,)

            p_down = float(mean_probs[DOWN])
            p_up = float(mean_probs[UP])
            score = p_up - p_down               # positive → bullish

            if p_up > p_down:
                side = "buy"
                confidence = p_up
            else:
                side = "sell"
                confidence = p_down

            smoothed[ticker] = {
                "side": side,
                "score": score,
                "confidence": confidence,
                "p_up": p_up,
                "p_down": p_down,
            }

        log.info("Smoothed signals for %d tickers", len(smoothed))
        return smoothed

    def save_signals(self, raw_signals: dict[str, list[float]]) -> None:
        """Persist today's raw signals, rotating out oldest day."""
        history = self._load_history()
        records = []
        for ticker, probs in raw_signals.items():
            past = history.get(ticker, [])
            window = (past + [probs])[-config.SIGNAL_SMOOTH_DAYS:]
            row = {"ticker": ticker}
            for i, p in enumerate(window):
                row[f"day{i}_probs"] = p
            records.append(row)

        if not records:
            return
        df = pd.DataFrame(records).set_index("ticker")
        df.to_parquet(config.SIGNALS_PATH)
        log.info("Saved signals for %d tickers → %s", len(records), config.SIGNALS_PATH)
