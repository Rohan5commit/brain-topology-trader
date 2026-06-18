"""Daily online RL weight update via policy gradient — v3 binary (down/up)."""
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config

log = logging.getLogger(__name__)

# v3 binary class indices
DOWN, UP = 0, 1


class OnlineUpdater:
    def __init__(self, model: nn.Module, lr: float = 1e-4) -> None:
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)

    def update(
        self,
        signals_df: pd.DataFrame,
        rewards: dict[str, float],
        device: torch.device,
        ticker_universe: list[str],
    ) -> float:
        """
        Policy-gradient update: maximize E[reward * log P(chosen_action)].

        signals_df columns: ticker, p_down, p_up, chosen_action
        rewards: {ticker: scalar_reward}
        Returns average reward.
        """
        if not rewards:
            log.warning("No rewards available — skipping update")
            return 0.0

        self.model.train()

        log_probs: list[torch.Tensor] = []
        reward_vals: list[float] = []

        for ticker, reward in rewards.items():
            if ticker not in signals_df.index:
                continue
            row = signals_df.loc[ticker]
            p_down = float(row.get("p_down", 0.5))
            p_up = float(row.get("p_up", 0.5))
            chosen = int(row.get("chosen_action", UP))

            probs_t = torch.tensor([p_down, p_up], dtype=torch.float32, device=device)
            probs_t = probs_t.clamp(min=1e-8)
            log_prob = torch.log(probs_t[chosen])
            log_probs.append(log_prob)
            reward_vals.append(reward)

        if not log_probs:
            return 0.0

        rewards_t = torch.tensor(reward_vals, dtype=torch.float32, device=device)
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
