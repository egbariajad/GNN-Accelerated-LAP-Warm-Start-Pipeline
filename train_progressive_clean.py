#!/usr/bin/env python3
"""
Progressive Multi-Size Training with Clean Datasets

Trains OneGNN on multiple problem sizes with curriculum learning:
- Epochs 1-3: Small/mid sizes (512-2048)
- Epochs 4+: Add large sizes (3072-4096)
- Round-robin sampling across sizes
- Separate validation per size

Usage:
    python train_progressive_clean.py --config progressive_clean_config.yaml
"""

import argparse
import math
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch import optim

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from gnn import OneGNN
from solvers import make_feasible_duals


@dataclass
class DatasetInfo:
    """Information about a training/validation dataset."""
    name: str
    path: Path
    size: int
    weight: float = 1.0
    curriculum_start_epoch: int = 1


class MultiSizeDataset:
    """Dataset that loads from multiple HDF5 files of different sizes."""
    
    def __init__(self, dataset_infos: List[DatasetInfo]):
        self.dataset_infos = dataset_infos
        self.h5_files = {}
        self.lengths = {}
        
        # Open all HDF5 files and cache lengths
        for info in dataset_infos:
            f = h5py.File(info.path, 'r')
            self.h5_files[info.name] = f
            self.lengths[info.name] = len(f['C'])
    
    def __len__(self) -> int:
        return sum(self.lengths.values())
    
    def get_instance(self, dataset_name: str, idx: int) -> Dict:
        """Load a single instance from a specific dataset."""
        f = self.h5_files[dataset_name]
        n = int(f['n'][idx])  # Get actual size, not padded shape
        C = torch.from_numpy(np.array(f['C'][idx]).reshape(n, n)).float()
        u = torch.from_numpy(np.array(f['u'][idx])[:n]).float()
        v = torch.from_numpy(np.array(f['v'][idx])[:n]).float()
        rows = torch.from_numpy(np.array(f['rows'][idx])[:n]).long()
        cols = torch.from_numpy(np.array(f['cols'][idx])[:n]).long()
        
        return {
            'C': C,
            'u': u,
            'v': v,
            'rows': rows,
            'cols': cols,
            'size': n,
            'dataset': dataset_name
        }
    
    def close(self):
        """Close all HDF5 files."""
        for f in self.h5_files.values():
            f.close()


class RoundRobinSampler:
    """Samples batches in round-robin fashion from multiple datasets with curriculum."""
    
    def __init__(
        self,
        dataset_infos: List[DatasetInfo],
        lengths: Dict[str, int],
        batch_size_rules: List[Dict],
        batches_per_size: int = 6,
        curriculum_enabled: bool = True,
        warmup_epochs: int = 3,
        seed: int = 42
    ):
        self.dataset_infos = dataset_infos
        self.lengths = lengths
        self.batch_size_rules = sorted(batch_size_rules, key=lambda x: x['max_size'])
        self.batches_per_size = batches_per_size
        self.curriculum_enabled = curriculum_enabled
        self.warmup_epochs = warmup_epochs
        self.rng = np.random.default_rng(seed)
        self.current_epoch = 1
        
    def get_batch_size(self, max_size: int) -> int:
        """Determine batch size based on problem size."""
        for rule in self.batch_size_rules:
            if max_size <= rule['max_size']:
                return rule['batch_size']
        return self.batch_size_rules[-1]['batch_size']
    
    def get_active_datasets(self, epoch: int) -> List[DatasetInfo]:
        """Get datasets active for the current epoch (curriculum)."""
        if not self.curriculum_enabled:
            return self.dataset_infos
        
        active = []
        for info in self.dataset_infos:
            if epoch >= info.curriculum_start_epoch:
                active.append(info)
        return active
    
    def __iter__(self):
        """Generate batches in round-robin fashion."""
        active_datasets = self.get_active_datasets(self.current_epoch)
        
        if not active_datasets:
            return
        
        # Create shuffled indices for each dataset
        indices = {}
        for info in active_datasets:
            idx_list = list(range(self.lengths[info.name]))
            self.rng.shuffle(idx_list)
            indices[info.name] = idx_list
        
        # Round-robin through datasets
        dataset_idx = 0
        batches_yielded = {info.name: 0 for info in active_datasets}
        
        while any(batches_yielded[info.name] < self.batches_per_size for info in active_datasets):
            info = active_datasets[dataset_idx % len(active_datasets)]
            
            if batches_yielded[info.name] >= self.batches_per_size:
                dataset_idx += 1
                continue
            
            # Get batch size for this dataset
            batch_size = self.get_batch_size(info.size)
            
            # Sample batch indices
            start_idx = batches_yielded[info.name] * batch_size
            batch_indices = []
            
            for _ in range(batch_size):
                if indices[info.name]:
                    batch_indices.append(indices[info.name].pop(0))
                else:
                    # Reshuffle if we run out
                    idx_list = list(range(self.lengths[info.name]))
                    self.rng.shuffle(idx_list)
                    indices[info.name] = idx_list
                    batch_indices.append(indices[info.name].pop(0))
            
            yield (info.name, batch_indices)
            batches_yielded[info.name] += 1
            dataset_idx += 1
    
    def set_epoch(self, epoch: int):
        """Update current epoch for curriculum learning."""
        self.current_epoch = epoch


def collate_mixed_size(batch_list: List[Dict], device: str) -> Dict:
    """Collate batch of mixed-size instances with padding.
    Computes row features like train_one_gnn.py does.
    """
    from gnn import compute_row_features, ROW_FEATURE_DIM
    
    batch_size = len(batch_list)
    max_n = max(item['size'] for item in batch_list)
    
    # Debug: Check for unreasonable sizes
    if max_n > 5000:
        print(f"WARNING: max_n={max_n}, batch_sizes: {[item['size'] for item in batch_list]}")
        raise ValueError(f"max_n={max_n} is too large! Check data loading.")
    
    # Initialize padded tensors
    C_batch = torch.zeros(batch_size, max_n, max_n, dtype=torch.float32)
    u_batch = torch.zeros(batch_size, max_n, dtype=torch.float32)
    v_batch = torch.zeros(batch_size, max_n, dtype=torch.float32)
    row_feat_batch = torch.zeros(batch_size, max_n, ROW_FEATURE_DIM, dtype=torch.float32)
    mask_batch = torch.zeros(batch_size, max_n, dtype=torch.bool)
    
    for i, item in enumerate(batch_list):
        n = item['size']
        # Convert to numpy for feature computation, then back to torch
        C_np = item['C'].cpu().numpy()
        features = compute_row_features(C_np)
        
        C_batch[i, :n, :n] = item['C']
        u_batch[i, :n] = item['u']
        v_batch[i, :n] = item['v']
        row_feat_batch[i, :n, :] = torch.from_numpy(features)
        mask_batch[i, :n] = True
    
    # Move to device
    return {
        'C': C_batch.to(device),
        'u': u_batch.to(device),
        'v': v_batch.to(device),
        'row_feat': row_feat_batch.to(device),
        'mask': mask_batch.to(device),
        'max_n': max_n,
        'batch_size': batch_size
    }


@dataclass
class Batch:
    cost: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    row_feat: torch.Tensor
    mask: torch.Tensor


def greedy_primal_upper(cost: torch.Tensor, reduced: torch.Tensor, n: int) -> torch.Tensor:
    """Greedy primal solution from reduced costs (copied from train_one_gnn.py)."""
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
    return torch.tensor(total_cost, dtype=cost.dtype, device=cost.device)


def compute_loss(batch: Dict, preds: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute loss (copied from train_one_gnn.py)."""
    mask = batch['mask']
    mask_float = mask.to(batch['C'].dtype)

    u_pred = preds["u"]
    u_pred = torch.where(mask, u_pred, torch.zeros_like(u_pred))

    # Use safer big value
    big = batch['C'].new_tensor(1e6)
    cost_minus = batch['C'] - u_pred.unsqueeze(-1)
    cost_minus = cost_minus.masked_fill(~mask.unsqueeze(-1), big)
    cost_minus = cost_minus.masked_fill(~mask.unsqueeze(1), big)
    v_proj = cost_minus.min(dim=1).values
    v_proj = torch.where(mask, v_proj, torch.zeros_like(v_proj))

    dual_lower = (u_pred * mask_float).sum(dim=1) + (v_proj * mask_float).sum(dim=1)

    hinge = torch.relu(u_pred.unsqueeze(-1) + v_proj.unsqueeze(-2) - batch['C'])
    mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)
    hinge = hinge * mask_2d.to(batch['C'].dtype)
    total_entries = mask_2d.sum(dim=(1, 2)).clamp(min=1).to(batch['C'].dtype)
    feas = hinge.sum(dim=(1, 2)) / total_entries

    reduced = batch['C'] - u_pred.unsqueeze(-1) - v_proj.unsqueeze(-2)
    primal_list = []
    for b in range(batch['C'].size(0)):
        n_i = int(mask[b].sum().item())
        primal_list.append(greedy_primal_upper(batch['C'][b], reduced[b], n_i))
    primal_upper = torch.stack(primal_list)

    primal_gap = primal_upper - dual_lower

    u_reg = ((u_pred - batch['u']) ** 2 * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)

    # Per-instance auxiliary metrics
    valid_counts = mask_float.sum(dim=1).clamp(min=1)
    u_mae = (torch.abs(u_pred - batch['u']) * mask_float).sum(dim=1) / valid_counts
    v_mae = (torch.abs(v_proj - batch['v']) * mask_float).sum(dim=1) / valid_counts

    total_per_instance = primal_gap + feas + 0.1 * u_reg

    # Combined loss
    loss = total_per_instance.mean()
    
    metrics = {
        "total": total_per_instance.detach().cpu().tolist(),
        "primal_gap": primal_gap.detach().cpu().tolist(),
        "feasibility": feas.detach().cpu().tolist(),
        "u_mae": u_mae.detach().cpu().tolist(),
        "v_mae": v_mae.detach().cpu().tolist(),
    }

    return loss, metrics


def validate(
    model: nn.Module,
    dataset: MultiSizeDataset,
    dataset_info: DatasetInfo,
    device: str,
    loss_weights: Dict[str, float],
    max_instances: int = 100
) -> Dict[str, float]:
    """Validate on a single dataset split."""
    model.eval()
    
    metrics = defaultdict(list)
    n_instances = min(dataset.lengths[dataset_info.name], max_instances)
    
    with torch.no_grad():
        for idx in range(n_instances):
            item = dataset.get_instance(dataset_info.name, idx)
            
            # Prepare batch of size 1 with row features
            batch_list = [item]
            batch = collate_mixed_size(batch_list, device)
            
            # Forward pass
            preds = model(batch['row_feat'], mask=batch['mask'])
            
            # Compute loss
            _, batch_metrics = compute_loss(batch, preds)
            
            for key, values in batch_metrics.items():
                if isinstance(values, list):
                    metrics[key].extend(values)
                else:
                    metrics[key].append(values)
    
    # Aggregate metrics
    result = {}
    for key, values in metrics.items():
        if len(values) == 0:
            result[f'{key}_mean'] = float('nan')
            result[f'{key}_median'] = float('nan')
            result[f'{key}_std'] = float('nan')
        else:
            result[f'{key}_mean'] = float(np.mean(values))
            result[f'{key}_median'] = float(np.median(values))
            result[f'{key}_std'] = float(np.std(values))
    
    return result


def train_progressive(config_path: Path, init_checkpoint: Optional[Path] = None):
    """Main progressive training loop."""
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("="*80)
    print("PROGRESSIVE MULTI-SIZE TRAINING - CLEAN DATASETS")
    print("="*80)
    print(f"Config: {config_path}")
    print()
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Parse dataset infos
    train_infos = [
        DatasetInfo(
            name=ds['name'],
            path=Path(ds['path']),
            size=ds['size'],
            weight=ds.get('weight', 1.0),
            curriculum_start_epoch=ds.get('curriculum_start_epoch', 1)
        )
        for ds in config['training']['datasets']
    ]
    
    val_infos = [
        DatasetInfo(
            name=ds['name'],
            path=Path(ds['path']),
            size=ds['size']
        )
        for ds in config['validation']['datasets']
    ]
    
    print(f"Training datasets: {len(train_infos)}")
    for info in train_infos:
        print(f"  - {info.name}: {info.path} (size={info.size}, start_epoch={info.curriculum_start_epoch})")
    print()
    
    print(f"Validation datasets: {len(val_infos)}")
    for info in val_infos:
        print(f"  - {info.name}: {info.path} (size={info.size})")
    print()
    
    # Create datasets
    train_dataset = MultiSizeDataset(train_infos)
    val_dataset = MultiSizeDataset(val_infos)
    
    print(f"Total training instances: {len(train_dataset)}")
    print(f"Total validation instances: {len(val_dataset)}")
    print()
    
    # Create sampler
    sampler = RoundRobinSampler(
        dataset_infos=train_infos,
        lengths=train_dataset.lengths,
        batch_size_rules=config['training_config']['batch_size_rules'],
        batches_per_size=config['training_config']['sampling']['batches_per_size'],
        curriculum_enabled=config['training_config']['curriculum']['enabled'],
        warmup_epochs=config['training_config']['curriculum']['warmup_epochs']
    )
    
    # Create model
    model_config = config['model']
    model = OneGNN(
        in_dim=model_config['row_feat_dim'],
        hidden=model_config['hidden_dim'],
        layers=model_config['layers'],
        dropout=model_config['dropout'],
        topk=model_config['topk']
    ).to(device)

    if init_checkpoint is not None:
        ckpt_path = Path(init_checkpoint)
        if ckpt_path.exists():
            print(f"Loading initial weights from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  ⚠️ Missing keys: {missing}")
            if unexpected:
                print(f"  ⚠️ Unexpected keys: {unexpected}")
        else:
            print(f"⚠️ Provided init checkpoint not found: {ckpt_path}")
    
    print(f"Model: OneGNN")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Hidden dim: {model_config['hidden_dim']}")
    print(f"  Layers: {model_config['layers']}")
    print(f"  Dropout: {model_config['dropout']}")
    print(f"  TopK: {model_config['topk']}")
    print()
    
    # Optimizer
    opt_config = config['training_config']['optimizer']
    optimizer = optim.AdamW(
        model.parameters(),
        lr=opt_config['lr'],
        weight_decay=opt_config['weight_decay'],
        betas=opt_config['betas']
    )
    
    # Scheduler (warmup cosine)
    train_config = config['training_config']
    total_epochs = train_config['epochs']
    batches_per_epoch = len(train_infos) * train_config['sampling']['batches_per_size']
    total_steps = total_epochs * batches_per_epoch
    warmup_steps = int(total_steps * config['training_config']['scheduler']['warmup_ratio'])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=config['training_config']['scheduler']['min_lr']
    )
    
    # Training loop
    best_metric = float('inf')
    patience_counter = 0
    checkpoint_dir = Path(config['checkpointing']['output_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting training...")
    print("="*80)
    
    for epoch in range(1, total_epochs + 1):
        model.train()
        sampler.set_epoch(epoch)
        
        epoch_metrics = defaultdict(list)
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch}/{total_epochs}")
        print(f"Active datasets: {[info.name for info in sampler.get_active_datasets(epoch)]}")
        
        batch_count = 0
        for dataset_name, batch_indices in sampler:
            batch_count += 1
            
            # Load batch (features computed in collate)
            batch_list = [train_dataset.get_instance(dataset_name, idx) for idx in batch_indices]
            batch = collate_mixed_size(batch_list, device)
            
            # Forward pass (row_feat already computed in collate)
            preds = model(batch['row_feat'], mask=batch['mask'])
            
            # Compute loss (same as train_one_gnn.py)
            loss, metrics = compute_loss(batch, preds)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['gradient_clip'])
            optimizer.step()
            scheduler.step()
            
            # Log
            for key, val in metrics.items():
                if isinstance(val, (list, tuple)):
                    epoch_metrics[key].extend(val)
                else:
                    epoch_metrics[key].append(val)
            
            if batch_count % config['logging']['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                # Simplified logging (metrics from compute_loss are different)
                print(f"  Batch {batch_count}/{batches_per_epoch} [{dataset_name}] "
                      f"Loss: {loss.item():.4f}, LR: {lr:.2e}")
        
        epoch_time = time.time() - epoch_start
        
        # Epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Time: {epoch_time:.1f}s")
        for key in ['total', 'primal_gap', 'feasibility', 'u_mae', 'v_mae']:
            values = epoch_metrics.get(key, [])
            if values:
                print(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")
            else:
                print(f"  {key}: n/a")
        
        # Validation
        if epoch >= train_config['min_epochs'] or epoch % 5 == 0:
            print("\nValidation:")
            val_results = {}
            for val_info in val_infos:
                val_metrics = validate(
                    model, val_dataset, val_info, device,
                    train_config['loss_weights'],
                    max_instances=50
                )
                val_results[val_info.name] = val_metrics
                print(f"  {val_info.name}:")
                def _fmt(value: float) -> str:
                    return "n/a" if math.isnan(value) else f"{value:.4f}"

                print(f"    PGap median: {_fmt(val_metrics.get('primal_gap_median', float('nan')))}")
                print(f"    Feas mean: {_fmt(val_metrics.get('feasibility_mean', float('nan')))}")
                print(f"    uMAE: {_fmt(val_metrics.get('u_mae_mean', float('nan')))}")
            
            # Compute averaged metric for early stopping
            pgaps = [v.get('primal_gap_median', float('nan')) for v in val_results.values()]
            pgaps = [float(p) for p in pgaps if not math.isnan(p)]
            if pgaps:
                avg_primal_gap = float(np.mean(pgaps))
                print(f"\n  Average primal gap (across splits): {avg_primal_gap:.4f}")
                # Early stopping check
                if avg_primal_gap < best_metric:
                    best_metric = avg_primal_gap
                    patience_counter = 0
                    # Save best checkpoint
                    checkpoint_path = checkpoint_dir / f"{config['checkpointing']['checkpoint_prefix']}_best.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_metric': best_metric,
                        'config': model_config
                    }, checkpoint_path)
                    print(f"  ✅ Saved best checkpoint: {checkpoint_path}")
                else:
                    patience_counter += 1
                    print(f"  Patience: {patience_counter}/{train_config['patience']}")
            else:
                avg_primal_gap = float('nan')
                print("\n  Average primal gap (across splits): n/a")
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{train_config['patience']}")
            
            if patience_counter >= train_config['patience'] and epoch >= train_config['min_epochs']:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
    
    # Save last checkpoint
    checkpoint_path = checkpoint_dir / f"{config['checkpointing']['checkpoint_prefix']}_last.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metric': best_metric,
        'config': model_config
    }, checkpoint_path)
    print(f"\n✅ Saved final checkpoint: {checkpoint_path}")
    
    # Cleanup
    train_dataset.close()
    val_dataset.close()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation primal gap: {best_metric:.4f}")
    print(f"Total epochs: {epoch}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default='progressive_clean_config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--init-checkpoint', type=Path, default=None,
                        help='Optional checkpoint to initialize model weights from')
    args = parser.parse_args()
    
    train_progressive(args.config, args.init_checkpoint)


if __name__ == '__main__':
    main()
