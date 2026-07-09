"""Composite reward: alpha * direction_accuracy + beta * pnl_return."""
import logging

import pandas as pd

log = logging.getLogger(__name__)

DOWN, UP = 0, 1   # matches model output and daily_snapshot chosen_action encoding


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
        For each ticker in the daily snapshot:
          direction_accuracy = 1 if chosen direction matches next-day move, else -1
          pnl_return = realized daily return for the chosen direction
          reward = alpha * direction_accuracy + beta * pnl_return

        signals_df columns: p_down, p_up, chosen_action (0=DOWN, 1=UP), prev_close
        """
        rewards: dict[str, float] = {}

        if signals_df.empty or not closing_prices:
            return rewards

        for ticker in signals_df.index:
            try:
                row = signals_df.loc[ticker]
                chosen_action = int(row.get("chosen_action", UP))
                prev_close = float(row.get("prev_close", 0.0))
                curr_close = closing_prices.get(ticker)

                if not curr_close or prev_close <= 0:
                    continue

                actual_return = (curr_close - prev_close) / prev_close

                predicted_up = chosen_action == UP
                actual_up = actual_return > 0

                direction_acc = 1.0 if (predicted_up == actual_up) else -1.0
                pnl = actual_return if predicted_up else -actual_return

                reward = alpha * direction_acc + beta * pnl
                rewards[ticker] = float(reward)

            except Exception as exc:
                log.debug("Reward error %s: %s", ticker, exc)

        log.info("Rewards computed: %d tickers, avg=%.4f",
                 len(rewards),
                 sum(rewards.values()) / max(len(rewards), 1))
        return rewards
