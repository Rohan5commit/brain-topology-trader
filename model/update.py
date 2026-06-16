"""Daily online RL weight update via policy gradient."""
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class OnlineUpdater:
    def __init__(self, model: nn.Module, lr: float = 1e-4) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def update(
        self,
        signals_df: pd.DataFrame,
        rewards: dict[str, float],
        device: torch.device,
        ticker_universe: list[str],
    ) -> float:
        """
        Policy-gradient update: maximize E[reward * log P(chosen_action)].

        signals_df columns: ticker, date, p_buy, p_hold, p_sell, chosen_action
        rewards: {ticker: scalar_reward}
        Returns average reward.
        """
        ticker_to_idx = {t: i for i, t in enumerate(ticker_universe)}

        if not rewards:
            log.warning("No rewards available — skipping update")
            return 0.0

        self.model.train()

        log_probs: list[torch.Tensor] = []
        reward_vals: list[float] = []

        today_col = [c for c in signals_df.columns if c not in ("ticker", "chosen_action")]
        if not today_col:
            return 0.0

        for ticker, reward in rewards.items():
            if ticker not in signals_df.index:
                continue
            row = signals_df.loc[ticker]
            p_buy = float(row.get("p_buy", 0.333))
            p_hold = float(row.get("p_hold", 0.334))
            p_sell = float(row.get("p_sell", 0.333))
            chosen = int(row.get("chosen_action", 1))  # default hold

            probs_t = torch.tensor([p_buy, p_hold, p_sell], dtype=torch.float32, device=device)
            probs_t = probs_t.clamp(min=1e-8)
            log_prob = torch.log(probs_t[chosen])
            log_probs.append(log_prob)
            reward_vals.append(reward)

        if not log_probs:
            return 0.0

        rewards_t = torch.tensor(reward_vals, dtype=torch.float32, device=device)
        # Normalize rewards for stable gradients
        if rewards_t.std() > 1e-6:
            rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

        log_probs_t = torch.stack(log_probs)
        loss = -(log_probs_t * rewards_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        avg_reward = float(torch.tensor(reward_vals).mean())
        log.info("PG update: loss=%.4f avg_reward=%.4f samples=%d", loss.item(), avg_reward, len(log_probs))
        return avg_reward
