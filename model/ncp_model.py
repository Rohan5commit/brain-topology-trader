"""NCP trading model v4: CfC + temporal attention + MLP head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC
from ncps.wirings import AutoNCP


class NCPTradingModel(nn.Module):
    """
    Input per forward call:
        x          : (batch, seq_len, num_features)
        stock_idx  : (batch,)
        sector_idx : (batch,)

    Output:
        logits: (batch, 2)  — raw logits [down, up]; apply softmax for probabilities
    """

    def __init__(
        self,
        num_stocks: int,
        num_features: int,
        input_size: int,
        ncp_units: int,
        ncp_output_size: int,        # motor neurons (64) → intermediate representation
        ncp_sparsity: float,
        embedding_dim: int,
        num_sectors: int = 13,
        sector_embedding_dim: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_stocks, embedding_dim)
        self.sector_embedding = nn.Embedding(num_sectors, sector_embedding_dim)
        self.feature_norm = nn.LayerNorm(num_features)
        self.dropout = nn.Dropout(dropout)

        wiring = AutoNCP(units=ncp_units, output_size=ncp_output_size, sparsity_level=ncp_sparsity)
        self.ltc = CfC(input_size, wiring, batch_first=True)

        # Temporal attention: weight each timestep's contribution
        self.attn = nn.Linear(ncp_output_size, 1)

        # MLP head: ncp_output_size → 2 logits
        self.head = nn.Sequential(
            nn.Linear(ncp_output_size, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(
        self,
        x: torch.Tensor,             # (B, T, F)
        stock_idx: torch.Tensor,     # (B,)
        sector_idx: torch.Tensor,    # (B,)
        hx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.feature_norm(x)
        emb = self.dropout(self.embedding(stock_idx))                # (B, E)
        sec = self.dropout(self.sector_embedding(sector_idx))        # (B, S)
        emb_exp = emb.unsqueeze(1).expand(-1, x.size(1), -1)
        sec_exp = sec.unsqueeze(1).expand(-1, x.size(1), -1)
        x_cat = torch.cat([x, emb_exp, sec_exp], dim=-1)            # (B, T, input_size)

        output, _ = self.ltc(x_cat, hx)                             # (B, T, ncp_output_size)

        attn_w = F.softmax(self.attn(output), dim=1)                 # (B, T, 1)
        context = (output * attn_w).sum(dim=1)                       # (B, ncp_output_size)

        return self.head(context)                                    # (B, 2) logits
