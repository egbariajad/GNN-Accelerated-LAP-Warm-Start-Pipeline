"""Lightweight GNN that predicts only row duals (u) from row-level features.

Removes all O(n^2) edge processing to make inference fast. Pair with
column dual projection via the min-trick: v_j = min_i (C_ij - u_i).
"""

from __future__ import annotations

from typing import Optional

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use OneGNN components") from exc


class ResidualBlock(nn.Module):
    """Lightweight residual MLP block used in OneGNN."""

    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return self.norm(residual + out)


class OneGNN(nn.Module):
    """Minimal per-row MLP that outputs u.

    Inputs:
      - row_feat: (batch, n, F) float tensor of row-level features
      - mask (optional): (batch, n) bool tensor marking valid rows

    Output dict:
      - 'u': (batch, n) float tensor of centered row dual predictions
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
        topk: int = 16,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        if hidden < 2:
            raise ValueError("hidden dimension must be >= 2 for head projection")

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden, dropout) for _ in range(layers)])

        head_hidden = max(hidden // 2, 1)
        self.pre_out = nn.Linear(hidden, 1)
        self.row_out = nn.Sequential(
            nn.Linear(hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        self.topk = topk
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.message_norm = nn.LayerNorm(hidden)
        self.message_dropout = nn.Dropout(dropout)

    def forward(
        self,
        row_feat: torch.Tensor,
        *,
        cost: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if row_feat.ndim == 2:  # (n, F) -> (1, n, F)
            row_feat = row_feat.unsqueeze(0)
        if row_feat.ndim != 3:
            raise ValueError("row_feat must have shape (batch, n, F)")

        h = self.input_proj(row_feat)
        for block in self.blocks:
            h = block(h)

        u_pre = self.pre_out(h).squeeze(-1)
        if cost is not None:
            h = h + self._sparse_refine(h, cost, u_pre, mask)

        u = self.row_out(h).squeeze(-1)  # (B, N)

        # Gauge: center per-instance for stability
        mean_u = u.mean(dim=-1, keepdim=True)
        u = u - mean_u

        if mask is not None:
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            u = u.masked_fill(~mask, 0.0)

        return {"u": u}

    def _sparse_refine(
        self,
        h: torch.Tensor,
        cost: torch.Tensor,
        u_pre: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Top-k column aggregation to inject global structure."""
        B, N, H = h.shape
        if N == 0:
            return torch.zeros_like(h)

        if mask is not None:
            mask_rows = mask.unsqueeze(-1)  # (B, N, 1)
        else:
            mask_rows = None

        k = min(self.topk, cost.size(-1))
        if k <= 0:
            return torch.zeros_like(h)

        reduced = cost - u_pre.unsqueeze(-1)
        if mask_rows is not None:
            reduced = reduced.masked_fill(~mask_rows, float("inf"))

        values, indices = torch.topk(reduced, k=k, dim=-1, largest=False)
        valid = torch.isfinite(values)
        neg_vals = torch.where(valid, -values, torch.full_like(values, -float("inf")))
        weights = torch.softmax(neg_vals, dim=-1)
        weights = torch.where(valid, weights, torch.zeros_like(weights))

        edge_inputs = torch.where(valid.unsqueeze(-1), values.unsqueeze(-1), torch.zeros_like(values.unsqueeze(-1)))
        edge_emb = self.edge_mlp(edge_inputs)  # (B, N, k, H)
        message = (weights.unsqueeze(-1) * edge_emb).sum(dim=-2)  # (B, N, H)

        if mask_rows is not None:
            message = message * mask_rows

        return self.message_norm(self.message_dropout(message))
