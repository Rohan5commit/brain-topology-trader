"""Composite reward: alpha * direction_accuracy + beta * pnl_return."""
import logging

import pandas as pd

log = logging.getLogger(__name__)

BUY, HOLD, SELL = 0, 1, 2


class RewardComputer:
    def compute(
        self,
        signals_df: pd.DataFrame,
        closing_prices: dict[str, float],
        today: str,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> dict[str, float]:
        """
        For each ticker that had a position today:
          direction_accuracy = 1 if predicted direction matches actual next-day move, else -1
          pnl_return = normalized daily PnL on that position
          reward = alpha * direction_accuracy + beta * pnl_return

        Returns {ticker: reward}.
        """
        rewards: dict[str, float] = {}

        if signals_df.empty or not closing_prices:
            return rewards

        for ticker in signals_df.index:
            try:
                row = signals_df.loc[ticker]
                p_buy = float(row.get("p_buy", 0.333))
                p_sell = float(row.get("p_sell", 0.333))
                chosen_action = int(row.get("chosen_action", HOLD))
                prev_close = float(row.get("prev_close", 0.0))
                curr_close = closing_prices.get(ticker)

                if not curr_close or prev_close <= 0:
                    continue

                actual_return = (curr_close - prev_close) / prev_close

                # Direction accuracy
                predicted_up = chosen_action == BUY
                actual_up = actual_return > 0
                predicted_down = chosen_action == SELL
                actual_down = actual_return < 0

                if (predicted_up and actual_up) or (predicted_down and actual_down):
                    direction_acc = 1.0
                elif chosen_action == HOLD:
                    direction_acc = 0.0
                else:
                    direction_acc = -1.0

                # PnL return (sign-adjusted for short positions)
                if chosen_action == SELL:
                    pnl = -actual_return
                elif chosen_action == BUY:
                    pnl = actual_return
                else:
                    pnl = 0.0

                reward = alpha * direction_acc + beta * pnl
                rewards[ticker] = float(reward)

            except Exception as exc:
                log.debug("Reward error %s: %s", ticker, exc)

        log.info("Rewards computed: %d tickers, avg=%.4f",
                 len(rewards),
                 sum(rewards.values()) / max(len(rewards), 1))
        return rewards
