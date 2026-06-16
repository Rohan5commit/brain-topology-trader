"""Kelly Criterion position sizing with concentration cap."""
import logging

log = logging.getLogger(__name__)


class KellySizer:
    @staticmethod
    def kelly_notional(
        p: float,
        b: float,
        portfolio_value: float,
        max_pct: float = 0.05,
    ) -> float:
        """
        f* = (p*b - q) / b   where q = 1 - p
        Returns dollar notional (0 if Kelly fraction <= 0).

        Args:
            p: Probability of winning (model confidence).
            b: Avg win / avg loss ratio (default 1.5 from config).
            portfolio_value: Total effective portfolio in dollars.
            max_pct: Hard cap per position as fraction of portfolio.
        """
        q = 1.0 - p
        f = (p * b - q) / b
        if f <= 0:
            return 0.0
        f = min(f, max_pct)
        notional = f * portfolio_value
        log.debug("Kelly: p=%.3f b=%.2f f=%.4f notional=$%.0f", p, b, f, notional)
        return notional
