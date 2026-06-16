"""NCP trading model: LTC with AutoNCP wiring + per-stock embedding."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import LTC
from ncps.wirings import AutoNCP


class NCPTradingModel(nn.Module):
    """
    Input per forward call:
        x    : (batch, seq_len, num_features)  — feature sequence
        idx  : (batch,)                          — stock index for embedding

    Output:
        probs: (batch, 3)  — softmax over [buy, hold, sell]
    """

    def __init__(
        self,
        num_stocks: int,
        input_size: int,        # num_features + embedding_dim
        ncp_units: int,
        ncp_output_size: int,
        ncp_sparsity: float,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_stocks, embedding_dim)
        self.embedding_dim = embedding_dim

        wiring = AutoNCP(
            units=ncp_units,
            output_size=ncp_output_size,
            sparsity_level=ncp_sparsity,
        )
        # LTC input_size = feature_dim + embedding_dim (embedding broadcast across seq)
        self.ltc = LTC(input_size=input_size, wiring=wiring, batch_first=True)

    def forward(
        self,
        x: torch.Tensor,          # (B, T, F)
        stock_idx: torch.Tensor,  # (B,)
        hx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.embedding(stock_idx)                     # (B, E)
        emb_expanded = emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, T, E)
        x_cat = torch.cat([x, emb_expanded], dim=-1)       # (B, T, F+E)

        output, _ = self.ltc(x_cat, hx)                    # (B, T, 3)
        last = output[:, -1, :]                             # (B, 3)
        return F.softmax(last, dim=-1)                      # (B, 3)
