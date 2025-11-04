"""DualGNN model for predicting warm-start dual potentials."""

from __future__ import annotations

import math
from typing import Optional

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use DualGNN components") from exc

from .features import EDGE_FEATURE_DIM, NODE_FEATURE_DIM


class DualLayer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by number of heads")

        self.heads = heads
        self.head_dim = hidden_dim // heads

        self.edge_mlp = nn.Sequential(
            nn.Linear(EDGE_FEATURE_DIM, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
        )

        self.row_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.col_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.row_val = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.col_val = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.attn_row_weight = nn.Parameter(torch.empty(heads, self.head_dim * 3))
        self.attn_col_weight = nn.Parameter(torch.empty(heads, self.head_dim * 3))
        self.attn_row_bias = nn.Parameter(torch.zeros(heads))
        self.attn_col_bias = nn.Parameter(torch.zeros(heads))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.row_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.col_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.row_norm = nn.LayerNorm(hidden_dim)
        self.col_norm = nn.LayerNorm(hidden_dim)

        nn.init.xavier_uniform_(self.attn_row_weight)
        nn.init.xavier_uniform_(self.attn_col_weight)

    def _prepare_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        B, N, H = tensor.shape
        tensor = tensor.view(B, N, self.heads, self.head_dim)
        return tensor

    def forward(
        self,
        edge_feat: torch.Tensor,
        row_embed: torch.Tensor,
        col_embed: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, _, _ = edge_feat.shape
        if mask is None:
            mask = edge_feat.new_ones((B, N), dtype=torch.bool)

        row_proj = self._prepare_heads(self.row_proj(row_embed))
        col_proj = self._prepare_heads(self.col_proj(col_embed))
        row_val = self._prepare_heads(self.row_val(row_embed))
        col_val = self._prepare_heads(self.col_val(col_embed))

        edge_emb = self.edge_mlp(edge_feat)
        edge_emb = edge_emb.view(B, N, N, self.heads, self.head_dim)

        # Row -> Col attention
        row_expand = row_proj.unsqueeze(2).expand(B, N, N, self.heads, self.head_dim)  # (B, N, N, H, D)
        col_expand = col_proj.unsqueeze(1).expand(B, N, N, self.heads, self.head_dim)  # (B, N, N, H, D)
        edge_heads = edge_emb

        attn_input = torch.cat([row_expand, col_expand, edge_heads], dim=-1)
        attn_input = attn_input.permute(0, 3, 1, 2, 4)  # (B, H, N, N, 3D)
        attn_scores = torch.einsum(
            "bhijm,hm->bhij", attn_input, self.attn_row_weight
        ) + self.attn_row_bias[None, :, None, None]
        attn_scores = self.leaky_relu(attn_scores)

        row_mask = mask[:, None, :, None]
        col_mask = mask[:, None, None, :]
        attn_scores = attn_scores.masked_fill(~row_mask, float("-inf"))
        attn_scores = attn_scores.masked_fill(~col_mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = torch.where(
            torch.isfinite(attn_weights), attn_weights, torch.zeros_like(attn_weights)
        )
        attn_weights = self.dropout(attn_weights)

        col_val_heads = col_val.permute(0, 2, 1, 3)  # (B, H, N, D)
        row_message = torch.einsum("bhij,bhjd->bhid", attn_weights, col_val_heads)
        row_message = row_message.permute(0, 2, 1, 3).reshape(B, N, -1)

        # Col -> Row attention (reverse direction)
        attn_input_col = torch.cat([col_expand, row_expand, edge_heads], dim=-1)
        attn_input_col = attn_input_col.permute(0, 3, 2, 1, 4)
        attn_scores_col = torch.einsum(
            "bhijm,hm->bhij", attn_input_col, self.attn_col_weight
        ) + self.attn_col_bias[None, :, None, None]
        attn_scores_col = self.leaky_relu(attn_scores_col)

        attn_scores_col = attn_scores_col.masked_fill(~col_mask, float("-inf"))
        attn_scores_col = attn_scores_col.masked_fill(~row_mask, float("-inf"))
        attn_weights_col = torch.softmax(attn_scores_col, dim=-1)
        attn_weights_col = torch.where(
            torch.isfinite(attn_weights_col),
            attn_weights_col,
            torch.zeros_like(attn_weights_col),
        )
        attn_weights_col = self.dropout(attn_weights_col)

        row_val_heads = row_val.permute(0, 2, 1, 3)
        col_message = torch.einsum("bhij,bhjd->bhid", attn_weights_col, row_val_heads)
        col_message = col_message.permute(0, 2, 1, 3).reshape(B, N, -1)

        row_update = self.row_update(torch.cat([row_embed, row_message], dim=-1))
        col_update = self.col_update(torch.cat([col_embed, col_message], dim=-1))

        row_embed = self.row_norm(row_embed + row_update)
        col_embed = self.col_norm(col_embed + col_update)

        return row_embed, col_embed


class DualGNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if layers <= 0:
            raise ValueError("DualGNN must have at least one layer")

        self.hidden_dim = hidden_dim
        self.heads = heads

        self.row_encoder = nn.Sequential(
            nn.Linear(NODE_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.col_encoder = nn.Sequential(
            nn.Linear(NODE_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.layers = nn.ModuleList(
            [DualLayer(hidden_dim, heads=heads, dropout=dropout) for _ in range(layers)]
        )
        self.row_out = nn.Linear(hidden_dim, 1)
        self.col_out = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        edge_feat: torch.Tensor,
        row_feat: torch.Tensor,
        col_feat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if edge_feat.ndim != 4:
            raise ValueError("edge_feat must have shape (batch, n, n, F)")

        row_embed = self.row_encoder(row_feat)
        col_embed = self.col_encoder(col_feat)

        for layer in self.layers:
            row_embed, col_embed = layer(edge_feat, row_embed, col_embed, mask)

        u = self.row_out(row_embed).squeeze(-1)
        v_hint = self.col_out(col_embed).squeeze(-1)

        mean_u = u.mean(dim=-1, keepdim=True)
        u = u - mean_u
        v_hint = v_hint + mean_u

        if mask is not None:
            u = u.masked_fill(~mask, 0.0)
            v_hint = v_hint.masked_fill(~mask, 0.0)

        return {"u": u, "v_hint": v_hint}
