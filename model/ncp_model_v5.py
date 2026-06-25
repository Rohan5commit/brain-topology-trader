"""NCP trading model v5: CfC + temporal attention + cross-sectional attention + MLP head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC
from ncps.wirings import AutoNCP


class NCPTradingModelV5(nn.Module):
    """
    v5 upgrade over v4: adds a cross-sectional attention layer.

    After each stock produces its temporal context vector, we run a single
    multi-head self-attention across all stocks in the batch that share the
    same date.  Each stock can now attend to peer stocks' hidden states,
    learning relative strength signals (e.g. sector rotation, pairs).

    Because the training DataLoader shuffles individual (stock, date) samples,
    we can't guarantee date-aligned batches at training time.  Instead we
    use a lightweight approximation: treat the entire mini-batch as a
    "virtual peer group" and apply cross-sectional attention across batch
    members.  At inference time, callers should group same-date stocks into
    one batch for full benefit — but even random mini-batches provide a
    useful regularisation signal.

    Input per forward call:
        x          : (B, T, F)
        stock_idx  : (B,)
        sector_idx : (B,)

    Output:
        logits: (B, 2)
    """

    def __init__(
        self,
        num_stocks: int,
        num_features: int,
        input_size: int,
        ncp_units: int,
        ncp_output_size: int,
        ncp_sparsity: float,
        embedding_dim: int,
        num_sectors: int = 13,
        sector_embedding_dim: int = 8,
        cs_heads: int = 4,          # cross-sectional attention heads
        cs_dropout: float = 0.1,
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

        # Cross-sectional attention: stocks attend to peer stocks
        self.cs_norm = nn.LayerNorm(ncp_output_size)
        self.cs_attn = nn.MultiheadAttention(
            embed_dim=ncp_output_size,
            num_heads=cs_heads,
            dropout=cs_dropout,
            batch_first=True,
        )
        self.cs_proj_norm = nn.LayerNorm(ncp_output_size)

        # MLP head: ncp_output_size → 2 logits
        self.head = nn.Sequential(
            nn.Linear(ncp_output_size, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        x: torch.Tensor,             # (B, T, F)
        stock_idx: torch.Tensor,     # (B,)
        sector_idx: torch.Tensor,    # (B,)
        hx=None,
    ) -> torch.Tensor:
        x = self.feature_norm(x)
        emb = self.dropout(self.embedding(stock_idx))                # (B, E)
        sec = self.dropout(self.sector_embedding(sector_idx))        # (B, S)
        emb_exp = emb.unsqueeze(1).expand(-1, x.size(1), -1)
        sec_exp = sec.unsqueeze(1).expand(-1, x.size(1), -1)
        x_cat = torch.cat([x, emb_exp, sec_exp], dim=-1)            # (B, T, input_size)

        output, _ = self.ltc(x_cat, hx)                             # (B, T, ncp_output_size)

        # Temporal attention → per-stock context vector
        attn_w = F.softmax(self.attn(output), dim=1)                 # (B, T, 1)
        context = (output * attn_w).sum(dim=1)                       # (B, ncp_output_size)

        # Cross-sectional attention (residual)
        # Treat batch dim as the "sequence" of stocks attending to each other
        cs_in = self.cs_norm(context).unsqueeze(0)                   # (1, B, D)
        cs_out, _ = self.cs_attn(cs_in, cs_in, cs_in)               # (1, B, D)
        context = self.cs_proj_norm(context + cs_out.squeeze(0))     # (B, D) residual

        return self.head(context)                                    # (B, 2)
