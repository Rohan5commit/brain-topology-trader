"""NCP trading model: CfC (Closed-form CfC) with AutoNCP wiring + per-stock and sector embeddings."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC
from ncps.wirings import AutoNCP


class NCPTradingModel(nn.Module):
    """
    Input per forward call:
        x          : (batch, seq_len, num_features)  — feature sequence
        stock_idx  : (batch,)                         — stock index for embedding
        sector_idx : (batch,)                         — sector index (0-12) for embedding

    Output:
        logits: (batch, 2)  — raw logits for [down, up]; apply softmax for probabilities
    """

    def __init__(
        self,
        num_stocks: int,
        num_features: int,          # raw feature count before embeddings (for LayerNorm)
        input_size: int,            # num_features + embedding_dim + sector_embedding_dim
        ncp_units: int,
        ncp_output_size: int,
        ncp_sparsity: float,
        embedding_dim: int,
        num_sectors: int = 13,
        sector_embedding_dim: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_stocks, embedding_dim)
        self.embedding_dim = embedding_dim
        self.sector_embedding = nn.Embedding(num_sectors, sector_embedding_dim)
        self.sector_embedding_dim = sector_embedding_dim
        self.feature_norm = nn.LayerNorm(num_features)
        self.dropout = nn.Dropout(dropout)

        wiring = AutoNCP(
            units=ncp_units,
            output_size=ncp_output_size,
            sparsity_level=ncp_sparsity,
        )
        self.ltc = CfC(input_size, wiring, batch_first=True)

    def forward(
        self,
        x: torch.Tensor,             # (B, T, F)
        stock_idx: torch.Tensor,     # (B,)
        sector_idx: torch.Tensor,    # (B,)
        hx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.feature_norm(x)                                     # normalize raw features
        emb = self.dropout(self.embedding(stock_idx))                # (B, E)
        sec = self.dropout(self.sector_embedding(sector_idx))        # (B, S)
        emb_exp = emb.unsqueeze(1).expand(-1, x.size(1), -1)        # (B, T, E)
        sec_exp = sec.unsqueeze(1).expand(-1, x.size(1), -1)        # (B, T, S)
        x_cat = torch.cat([x, emb_exp, sec_exp], dim=-1)            # (B, T, F+E+S)

        output, _ = self.ltc(x_cat, hx)                             # (B, T, 2)
        return output[:, -1, :]                                      # (B, 2) raw logits
