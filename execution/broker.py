"""Alpaca Paper Trading client — place and manage orders."""
import logging
import math
import os
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd

import config

log = logging.getLogger(__name__)

_RETRY = 3
_WAIT = 2

# Alpaca 4xx codes that are permanent — don't retry
_NO_RETRY_CODES = {
    40310000,  # insufficient buying power
    40310001,  # insufficient qty
    42210000,  # fractional short sell
    40410000,  # symbol not found
    42110000,  # market closed
}


def _is_permanent(exc: Exception) -> bool:
    msg = str(exc)
    try:
        import json
        body = json.loads(msg)
        return body.get("code") in _NO_RETRY_CODES
    except Exception:
        return False


def _retry(fn, *args, **kwargs):
    for attempt in range(_RETRY):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == _RETRY - 1 or _is_permanent(exc):
                raise
            log.warning("Alpaca call failed (%s) — retry %d/%d", exc, attempt + 1, _RETRY)
            time.sleep(_WAIT * (attempt + 1))


class AlpacaBroker:
    def __init__(self) -> None:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, PositionSide
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest

        self._tc = TradingClient(
            api_key=os.environ["ALPACA_API_KEY"],
            secret_key=os.environ["ALPACA_SECRET_KEY"],
            paper=True,
        )
        self._dc = StockHistoricalDataClient(
            api_key=os.environ["ALPACA_API_KEY"],
            secret_key=os.environ["ALPACA_SECRET_KEY"],
        )
        self._MarketOrderRequest = MarketOrderRequest
        self._OrderSide = OrderSide
        self._TimeInForce = TimeInForce
        self._StockLatestTradeRequest = StockLatestTradeRequest

    def get_portfolio_value(self) -> float:
        account = _retry(self._tc.get_account)
        equity = float(account.equity)
        # Return effective (leveraged) value; Alpaca paper supports 2x intraday margin
        return equity * 2

    def get_open_positions(self) -> dict[str, dict]:
        """Return {symbol: {side, qty, avg_entry, entry_date}}."""
        positions = _retry(self._tc.get_all_positions)
        state: dict[str, dict] = {}
        pos_file = config.POSITIONS_PATH
        if os.path.exists(pos_file):
            df = pd.read_parquet(pos_file)
            entry_dates = df.set_index("ticker")["entry_date"].to_dict()
        else:
            entry_dates = {}

        for pos in positions:
            state[pos.symbol] = {
                "side": "buy" if float(pos.qty) > 0 else "sell",
                "qty": float(pos.qty),
                "avg_entry": float(pos.avg_entry_price),
                "entry_date": entry_dates.get(pos.symbol, datetime.now(timezone.utc).date().isoformat()),
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl),
            }
        return state

    def close_stale_positions(
        self,
        smoothed: dict[str, dict],
        threshold: float,
        min_hold_days: int,
    ) -> int:
        """Close positions where signal flipped or confidence dropped below threshold, if held >= min_hold_days."""
        positions = self.get_open_positions()
        today = datetime.now(timezone.utc).date()
        closed = 0

        for symbol, pos in positions.items():
            entry = date.fromisoformat(pos["entry_date"])
            days_held = (today - entry).days
            if days_held < min_hold_days:
                continue

            sig = smoothed.get(symbol)
            if sig is None:
                # Ticker no longer in signal set — close
                _retry(self._tc.close_position, symbol)
                log.info("Closed %s (signal missing, held %d days)", symbol, days_held)
                closed += 1
                continue

            conf = sig["confidence"]
            signal_side = sig["side"]
            pos_side = pos["side"]

            # Close if: confidence below threshold OR signal flipped direction
            if conf < threshold or signal_side != pos_side:
                _retry(self._tc.close_position, symbol)
                log.info("Closed %s (%s→%s, conf=%.3f, held=%d)", symbol, pos_side, signal_side, conf, days_held)
                closed += 1

        self._save_positions()
        return closed

    def _get_latest_price(self, ticker: str) -> float | None:
        try:
            resp = self._dc.get_stock_latest_trade(
                self._StockLatestTradeRequest(symbol_or_symbols=ticker)
            )
            return float(resp[ticker].price)
        except Exception as exc:
            log.warning("Could not fetch latest price for %s: %s", ticker, exc)
            return None

    def place_order(self, ticker: str, side: str, notional: float) -> dict | None:
        if notional < 1.0:
            return None
        order_side = self._OrderSide.BUY if side == "buy" else self._OrderSide.SELL

        if side == "sell":
            # Alpaca paper rejects fractional short sells — convert to whole shares
            price = self._get_latest_price(ticker)
            if not price or price <= 0:
                log.warning("Skipping short %s — could not get price", ticker)
                return None
            qty = math.floor(notional / price)
            if qty < 1:
                log.warning("Skipping short %s — qty=0 at price=%.2f notional=%.0f", ticker, price, notional)
                return None
            req = self._MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=order_side,
                time_in_force=self._TimeInForce.DAY,
            )
        else:
            req = self._MarketOrderRequest(
                symbol=ticker,
                notional=round(notional, 2),
                side=order_side,
                time_in_force=self._TimeInForce.DAY,
            )
        try:
            order = _retry(self._tc.submit_order, req)
            log.info("Order: %s %s $%.0f → id=%s", side.upper(), ticker, notional, order.id)
            self._record_entry(ticker, side)
            return {"ticker": ticker, "side": side, "notional": notional, "id": str(order.id)}
        except Exception as exc:
            log.error("Order failed %s %s: %s", side, ticker, exc)
            return None

    def _save_positions(self) -> None:
        positions = self.get_open_positions()
        records = [{"ticker": t, **v} for t, v in positions.items()]
        # Always write (even empty) so stale entry dates don't persist after positions close
        pd.DataFrame(records).to_parquet(config.POSITIONS_PATH, index=False)

    def _record_entry(self, ticker: str, side: str) -> None:
        pos_file = config.POSITIONS_PATH
        today = datetime.now(timezone.utc).date().isoformat()
        if os.path.exists(pos_file):
            df = pd.read_parquet(pos_file)
        else:
            df = pd.DataFrame(columns=["ticker", "side", "entry_date"])
        new_row = pd.DataFrame([{"ticker": ticker, "side": side, "entry_date": today}])
        df = pd.concat([df[df["ticker"] != ticker], new_row], ignore_index=True)
        df.to_parquet(pos_file, index=False)
