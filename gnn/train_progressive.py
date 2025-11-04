"""Enhanced training script with bucket-aware batching and checkpoint resumption."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

if __package__ is None or __package__ == "":  # Allow running as script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # pragma: no cover - optional dependency
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to run training") from exc

import h5py

from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler

from gnn import DualGNN, compute_features, EDGE_FEATURE_DIM, NODE_FEATURE_DIM


@dataclass
class Batch:
    cost: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    row_feat: torch.Tensor
    col_feat: torch.Tensor
    edge_feat: torch.Tensor
    mask: torch.Tensor


@dataclass
class BucketConfig:
    """Configuration for bucket-aware training."""
    name: str
    max_size: int
    batch_size: int
    train_path: Path
    val_path: Path


class SizeBucket:
    """Utility to determine matrix size buckets."""
    
    @staticmethod
    def get_bucket_for_size(n: int) -> str:
        """Determine which bucket a matrix size belongs to."""
        if n <= 512:
            return "small"
        elif n <= 1024:
            return "small_1024"
        elif n <= 1536:
            return "mid_1536"
        elif n <= 2048:
            return "mid_2048"
        elif n <= 3072:
            return "mid_3072"
        else:
            return "large"


class LapDataset(Dataset):
    """Lazy reader for the generated HDF5 datasets with size filtering."""

    def __init__(self, path: Path, filter_size: Optional[int] = None) -> None:
        self.path = Path(path)
        self.filter_size = filter_size
        
        # Load and optionally filter indices based on size
        with h5py.File(self.path, "r") as f:
            all_sizes = f["n"][:]
            if filter_size is not None:
                # Create mask for samples matching the filter size
                self.valid_indices = np.where(all_sizes == filter_size)[0]
            else:
                self.valid_indices = np.arange(len(all_sizes))
            
        self.length = len(self.valid_indices)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # Map dataset index to actual file index
        actual_idx = self.valid_indices[idx]
        
        with h5py.File(self.path, "r") as f:
            n = int(f["n"][actual_idx])
            c = np.array(f["C"][actual_idx]).reshape(n, n)
            u = np.array(f["u"][actual_idx])[:n]
            v = np.array(f["v"][actual_idx])[:n]
        return {"cost": c, "u": u, "v": v, "n": n}


def collate(batch: Sequence[Dict[str, np.ndarray]]) -> Batch:
    max_n = max(item["n"] for item in batch)
    bsz = len(batch)

    cost = torch.zeros(bsz, max_n, max_n, dtype=torch.float32)
    u = torch.zeros(bsz, max_n, dtype=torch.float32)
    v = torch.zeros(bsz, max_n, dtype=torch.float32)
    row_feat = torch.zeros(bsz, max_n, NODE_FEATURE_DIM, dtype=torch.float32)
    col_feat = torch.zeros(bsz, max_n, NODE_FEATURE_DIM, dtype=torch.float32)
    edge_feat = torch.zeros(bsz, max_n, max_n, EDGE_FEATURE_DIM, dtype=torch.float32)
    mask = torch.zeros(bsz, max_n, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["n"]
        cost[i, :n, :n] = torch.from_numpy(item["cost"].astype(np.float32, copy=False))
        u[i, :n] = torch.from_numpy(item["u"].astype(np.float32, copy=False))
        v[i, :n] = torch.from_numpy(item["v"].astype(np.float32, copy=False))
        features = compute_features(item["cost"])
        row_feat[i, :n, :] = torch.from_numpy(features.row_feat)
        col_feat[i, :n, :] = torch.from_numpy(features.col_feat)
        edge_feat[i, :n, :n, :] = torch.from_numpy(features.edge_feat)
        mask[i, :n] = True

    return Batch(
        cost=cost,
        u=u,
        v=v,
        row_feat=row_feat,
        col_feat=col_feat,
        edge_feat=edge_feat,
        mask=mask,
    )


class WarmupCosineScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ) -> None:
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps, total_steps)
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        step = min(self.total_steps, self.last_epoch + 1)
        if self.total_steps <= self.warmup_steps:
            scale = step / self.total_steps
            return [base_lr * scale for base_lr in self.base_lrs]
        if step <= self.warmup_steps:
            scale = step / self.warmup_steps
            return [base_lr * scale for base_lr in self.base_lrs]
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]


def batch_to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        cost=batch.cost.to(device),
        u=batch.u.to(device),
        v=batch.v.to(device),
        row_feat=batch.row_feat.to(device),
        col_feat=batch.col_feat.to(device),
        edge_feat=batch.edge_feat.to(device),
        mask=batch.mask.to(device),
    )

def greedy_primal_upper(cost: torch.Tensor, reduced: torch.Tensor, n: int) -> torch.Tensor:
    """Greedy primal construction used in train.py to compute an upper bound.

    Mirrors the implementation in gnn/train.py for consistency.
    """
    if n == 0:
        return cost.new_zeros(())
    cost_np = cost[:n, :n].detach().cpu().numpy()
    reduced_np = reduced[:n, :n].detach().cpu().numpy()

    assignment = [-1] * n
    used_cols = set()
    row_order = np.argsort(reduced_np.min(axis=1))

    for row in row_order:
        for col in np.argsort(reduced_np[row]):
            if col not in used_cols:
                assignment[row] = int(col)
                used_cols.add(int(col))
                break
        if assignment[row] == -1:
            remaining = [c for c in range(n) if c not in used_cols]
            if remaining:
                best = min(remaining, key=lambda c: reduced_np[row, c])
                assignment[row] = int(best)
                used_cols.add(int(best))
            else:
                assignment[row] = int(np.argmin(reduced_np[row]))

    seen = {}
    for r, c in enumerate(assignment):
        seen.setdefault(c, []).append(r)

    available = [c for c in range(n) if c not in used_cols]
    for col, rows in seen.items():
        if len(rows) <= 1:
            continue
        for row in rows[1:]:
            if available:
                new_col = min(available, key=lambda c: reduced_np[row, c])
                assignment[row] = int(new_col)
                available.remove(new_col)

    total_cost = sum(cost_np[row, assignment[row]] for row in range(n))
    return cost.new_tensor(total_cost)


def compute_loss(batch: Batch, preds: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, list[float]]]:
    """Improved primal-dual loss aligned with gnn/train.py.

    - Use u prediction and project v via min-trick to enforce dual feasibility.
    - Penalize feasibility via hinge on reduced costs.
    - Build a greedy primal to estimate primal-dual gap.
    - Regularize v_hint toward v_proj.
    """
    device = batch.cost.device
    mask = batch.mask
    mask_float = mask.to(batch.cost.dtype)

    u_pred = preds["u"]
    v_hint = preds["v_hint"]

    # Project v from u via column-wise min of (C - u)
    big = batch.cost.new_tensor(1e9)
    cost_minus = batch.cost - u_pred.unsqueeze(-1)
    cost_minus = cost_minus.masked_fill(~mask.unsqueeze(-1), big)
    cost_minus = cost_minus.masked_fill(~mask.unsqueeze(1), big)
    v_proj = cost_minus.min(dim=1).values
    v_proj = torch.where(mask, v_proj, torch.zeros_like(v_proj))

    # Dual lower bound sum(u) + sum(v)
    dual_lower = (u_pred * mask_float).sum(dim=1) + (v_proj * mask_float).sum(dim=1)

    # Hinge feasibility on reduced costs (u_i + v_j - C_ij <= 0)
    hinge = torch.relu(u_pred.unsqueeze(-1) + v_proj.unsqueeze(-2) - batch.cost)
    mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)
    hinge = hinge * mask_2d.to(batch.cost.dtype)
    total_entries = mask_2d.sum(dim=(1, 2)).clamp(min=1).to(batch.cost.dtype)
    feas = hinge.sum(dim=(1, 2)) / total_entries

    # Greedy primal upper bound on each sample
    reduced = batch.cost - u_pred.unsqueeze(-1) - v_proj.unsqueeze(-2)
    primal_list = []
    for b in range(batch.cost.size(0)):
        n_i = int(mask[b].sum().item())
        primal_list.append(greedy_primal_upper(batch.cost[b], reduced[b], n_i))
    primal_upper = torch.stack(primal_list)

    # Primal-dual gap per sample
    primal_gap = primal_upper - dual_lower

    # Encourage v_hint to match v_proj
    v_reg = ((v_hint - v_proj) ** 2 * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)

    # Total loss
    loss = primal_gap.mean() + feas.mean() + 0.1 * v_reg.mean()

    metrics = {
        "primal_gap_values": primal_gap.detach().cpu().tolist(),
        "feas_values": feas.detach().cpu().tolist(),
    }

    return loss, metrics


def train_epoch(
    model: DualGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[WarmupCosineScheduler],
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    accum_steps: int = 1,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    count = 0
    primal_values: list[float] = []
    feas_values: list[float] = []

    step_idx = 0
    optimizer.zero_grad(set_to_none=True)
    for batch in loader:
        batch = batch_to_device(batch, device)
        with autocast('cuda', enabled=use_amp):
            preds = model(
                batch.edge_feat,
                batch.row_feat,
                batch.col_feat,
                mask=batch.mask,
            )
            loss, metrics = compute_loss(batch, preds)

        # Accumulate gradients
        scaled_loss = loss / max(1, accum_steps)
        scaler.scale(scaled_loss).backward()

        batch_size = batch.cost.size(0)
        total_loss += float(loss.item()) * batch_size
        count += batch_size
        primal_values.extend(metrics["primal_gap_values"])
        feas_values.extend(metrics["feas_values"])

        step_idx += 1
        if step_idx % max(1, accum_steps) == 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            prev_step_count = getattr(optimizer, "_step_count", 0)
            scaler.step(optimizer)
            scaler.update()
            stepped = getattr(optimizer, "_step_count", 0) > prev_step_count
            if stepped and scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    # Flush leftover gradients if the last step didn't hit an optimizer step
    if step_idx % max(1, accum_steps) != 0:
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        prev_step_count = getattr(optimizer, "_step_count", 0)
        scaler.step(optimizer)
        scaler.update()
        stepped = getattr(optimizer, "_step_count", 0) > prev_step_count
        if stepped and scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / max(1, count)
    primal_gap_median = float(np.median(primal_values)) if primal_values else 0.0
    primal_gap_mean = float(np.mean(primal_values)) if primal_values else 0.0
    feas_mean = float(np.mean(feas_values)) if feas_values else 0.0
    return {
        "loss": avg_loss,
        "primal_gap_median": primal_gap_median,
        "primal_gap_mean": primal_gap_mean,
        "feas_mean": feas_mean,
    }


@torch.no_grad()
def evaluate(model: DualGNN, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    primal_values: list[float] = []
    feas_values: list[float] = []

    for batch in loader:
        batch = batch_to_device(batch, device)
        preds = model(
            batch.edge_feat,
            batch.row_feat,
            batch.col_feat,
            mask=batch.mask,
        )
        loss, metrics = compute_loss(batch, preds)

        batch_size = batch.cost.size(0)
        total_loss += float(loss.item()) * batch_size
        count += batch_size
        primal_values.extend(metrics["primal_gap_values"])
        feas_values.extend(metrics["feas_values"])

    avg_loss = total_loss / max(1, count)
    primal_gap_median = float(np.median(primal_values)) if primal_values else 0.0
    primal_gap_mean = float(np.mean(primal_values)) if primal_values else 0.0
    feas_mean = float(np.mean(feas_values)) if feas_values else 0.0
    return {
        "loss": avg_loss,
        "primal_gap_median": primal_gap_median,
        "primal_gap_mean": primal_gap_mean,
        "feas_mean": feas_mean,
    }


def save_checkpoint(
    model: DualGNN,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[WarmupCosineScheduler],
    epoch: int,
    best_val_gap: float,
    args: argparse.Namespace,
    path: Path,
) -> None:
    """Save complete training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_val_gap": best_val_gap,
        "hidden_dim": args.hidden,
        "layers": args.layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "lr": args.lr,
    }
    
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    model: DualGNN,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[WarmupCosineScheduler],
    path: Path,
) -> tuple[int, float]:
    """Load checkpoint and return epoch and best_val_gap."""
    checkpoint = torch.load(path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    epoch = checkpoint.get("epoch", 0)
    best_val_gap = checkpoint.get("best_val_gap", float("inf"))
    
    print(f"Loaded checkpoint from {path} (epoch {epoch})")
    return epoch, best_val_gap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Path to train.h5")
    parser.add_argument("--val", type=Path, help="Optional validation split")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum-steps", type=int, default=1, help="Accumulate gradients over this many steps")
    parser.add_argument("--filter-size", type=int, help="Filter training data to specific matrix size")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--pin-memory", action="store_true", help="Pin host memory for faster HtoD copies")
    parser.add_argument("--output", type=Path, default=Path("gnn_checkpoint.pt"))
    parser.add_argument("--checkpoint-init", type=Path, help="Initialize from checkpoint")
    parser.add_argument("--checkpoint-save", type=Path, help="Save checkpoint every epoch")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint-save if exists")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = DualGNN(
        hidden_dim=args.hidden,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Handle dataset loading with optional size filtering
    train_dataset = LapDataset(args.train, filter_size=args.filter_size)
    val_dataset = LapDataset(args.val, filter_size=args.filter_size) if args.val else None
    
    print(f"Training on {len(train_dataset)} samples")
    if args.filter_size:
        print(f"Filtered to size {args.filter_size} matrices")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=collate,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=True if args.num_workers > 0 else False,
        )
        if val_dataset
        else None
    )

    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = min(2000, total_steps)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    scaler = GradScaler(enabled=device.type == "cuda")
    use_amp = device.type == "cuda"

    # Initialize training state
    start_epoch = 1
    best_val_gap = float("inf")
    best_state = None
    
    # Handle checkpoint loading
    if args.resume and args.checkpoint_save and args.checkpoint_save.exists():
        start_epoch, best_val_gap = load_checkpoint(model, optimizer, scheduler, args.checkpoint_save)
        start_epoch += 1  # Start from next epoch
    elif args.checkpoint_init and args.checkpoint_init.exists():
        # Load weights only, not training state
        checkpoint = torch.load(args.checkpoint_init, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Initialized model weights from {args.checkpoint_init}")

    patience = 0
    patience_limit = 5
    min_epochs = min(60, args.epochs)

    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            use_amp,
            accum_steps=args.accum_steps,
        )

        val_stats = evaluate(model, val_loader, device) if val_loader is not None else None

        log_msg = (
            f"epoch {epoch:03d} | train loss {train_stats['loss']:.4f}"
            f" gap_med {train_stats['primal_gap_median']:.4f}"
            f" feas {train_stats['feas_mean']:.4f}"
        )
        if val_stats is not None:
            log_msg += (
                f" | val loss {val_stats['loss']:.4f}"
                f" gap_med {val_stats['primal_gap_median']:.4f}"
                f" feas {val_stats['feas_mean']:.4f}"
            )
        print(log_msg)

        if val_stats is not None:
            current_gap = val_stats["primal_gap_median"]
            if current_gap < best_val_gap - 1e-6:
                best_val_gap = current_gap
                patience = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if epoch >= min_epochs and patience >= patience_limit:
                    print("Early stopping triggered based on validation primal gap.")
                    break

        # Save checkpoint if requested
        if args.checkpoint_save:
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_gap, args, args.checkpoint_save)

    # Save final model
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": args.hidden,
            "layers": args.layers,
            "heads": args.heads,
            "dropout": args.dropout,
        },
        args.output,
    )
    print(f"Saved final model to {args.output}")


if __name__ == "__main__":
    main()
