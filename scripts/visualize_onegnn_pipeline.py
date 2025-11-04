#!/usr/bin/env python3
"""
Visualise the OneGNN architecture and simulate a forward pass with detailed step-by-step breakdown.

Outputs:
  - PNG figure illustrating the model blocks, sparse top-k refinement,
    and min-trick recovery of v.
  - Detailed step-by-step forward pass visualization
  - Console logs describing feature -> model -> dual flow for a sample matrix.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Add project root for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnn import OneGNN, compute_row_features, compute_row_features_torch  # type: ignore

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[OneGNN, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    hidden = checkpoint.get("hidden_dim")
    layers = checkpoint.get("layers")
    dropout = checkpoint.get("dropout", 0.1)
    row_feat_dim = checkpoint.get("row_feat_dim") or compute_row_features(np.zeros((1, 1))).shape[1]

    if hidden is None or layers is None:
        raise ValueError(f"Checkpoint {checkpoint_path} missing hidden/layers metadata.")

    model = OneGNN(
        in_dim=row_feat_dim,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    meta = {
        "hidden": hidden,
        "layers": layers,
        "dropout": dropout,
        "row_feat_dim": row_feat_dim,
        "topk": getattr(model, "topk", 16),
    }
    return model, meta


def simulate_forward(
    model: OneGNN,
    cost: np.ndarray,
    device: torch.device,
    use_cuda_features: bool = False,
) -> dict:
    """Simulate forward pass and capture all intermediate activations."""
    n = cost.shape[0]
    cost_tensor = torch.from_numpy(cost).float().to(device)
    mask = torch.ones((1, n), dtype=torch.bool, device=device)

    if use_cuda_features and device.type == "cuda":
        row_feat_tensor = compute_row_features_torch(cost_tensor)
    else:
        row_feat = compute_row_features(cost)
        row_feat_tensor = torch.from_numpy(row_feat).float().to(device)

    with torch.inference_mode():
        # Stage 1: input projection
        h = model.input_proj(row_feat_tensor.unsqueeze(0))
        projections = [h.squeeze(0).cpu().numpy()]
        
        # Track each residual block
        block_outputs = []
        for i, block in enumerate(model.blocks):
            h_before = h.clone()
            h = block(h)
            block_outputs.append({
                'before': h_before.squeeze(0).cpu().numpy(),
                'after': h.squeeze(0).cpu().numpy(),
                'residual': (h - h_before).squeeze(0).cpu().numpy()
            })
            projections.append(h.squeeze(0).cpu().numpy())

        # Pre-head linear (used for top-k)
        u_pre = model.pre_out(h).squeeze(-1)

        # Sparse refinement (copy logic to extract info)
        refine_info = compute_sparse_refine_info(model, h, cost_tensor.unsqueeze(0), u_pre, mask)
        h_refined = h + refine_info["message"]

        # Final row head
        h_head_input = h_refined.clone()
        u_final = model.row_out(h_refined).squeeze(-1)
        u_final_raw = u_final.clone()
        mean_u = u_final.mean(dim=-1, keepdim=True)
        u_centered = (u_final - mean_u).squeeze(0)
        v_trick = torch.min(cost_tensor - u_centered[:, None], dim=0)[0]

    return {
        "row_features": row_feat_tensor.cpu().numpy(),
        "hidden_activations": projections,
        "block_outputs": block_outputs,
        "u_pre": u_pre.squeeze(0).cpu().numpy(),
        "refine_info": {
            "indices": refine_info["indices"].squeeze(0).cpu().numpy(),
            "values": refine_info["values"].squeeze(0).cpu().numpy(),
            "weights": refine_info["weights"].squeeze(0).cpu().numpy(),
            "message": refine_info["message"].squeeze(0).cpu().numpy(),
        },
        "h_before_refine": h.squeeze(0).cpu().numpy(),
        "h_after_refine": h_refined.squeeze(0).cpu().numpy(),
        "u_final_raw": u_final_raw.squeeze(0).cpu().numpy(),
        "mean_u": mean_u.squeeze(0).cpu().numpy(),
        "u_final": u_centered.cpu().numpy(),
        "v_trick": v_trick.cpu().numpy(),
    }


def compute_sparse_refine_info(
    model: OneGNN,
    h: torch.Tensor,
    cost: torch.Tensor,
    u_pre: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> dict:
    message = model._sparse_refine(h, cost, u_pre, mask)

    # Replicate internal computations to expose top-k details
    reduced = cost - u_pre.unsqueeze(-1)
    if mask is not None:
        reduced = reduced.masked_fill(~mask.unsqueeze(-1), float("inf"))

    k = min(model.topk, cost.size(-1))
    values, indices = torch.topk(reduced, k=k, dim=-1, largest=False)
    valid = torch.isfinite(values)
    neg_vals = torch.where(valid, -values, torch.full_like(values, -float("inf")))
    weights = torch.softmax(neg_vals, dim=-1)
    weights = torch.where(valid, weights, torch.zeros_like(weights))

    return {
        "message": message,
        "indices": indices,
        "values": values,
        "weights": weights,
    }


def draw_architecture(ax: plt.Axes, meta: dict):
    """Draw comprehensive architecture diagram."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("OneGNN Architecture: Row-Only Encoder with Sparse Refinement", 
                 fontsize=14, fontweight='bold', pad=20)

    def box(x, y, w, h, text, color="lightsteelblue", edge_color="navy", linewidth=1.5):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, 
            boxstyle="round,pad=0.015", 
            linewidth=linewidth, 
            edgecolor=edge_color, 
            facecolor=color,
            alpha=0.9
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, 
                ha="center", va="center", fontsize=9, 
                color="black", weight='bold',
                wrap=True)
        return x + w / 2, y + h / 2

    def arrow(x1, y1, x2, y2, style="->", color="black", linewidth=2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), 
                   arrowprops=dict(arrowstyle=style, color=color, 
                                 linewidth=linewidth, shrinkA=0, shrinkB=0))
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.02, label, ha="center", fontsize=7, 
                   style='italic', color=color)

    # Layout parameters
    box_h = 0.11
    col1_x, col2_x, col3_x = 0.08, 0.38, 0.68
    box_w = 0.24
    
    # Row 1: Input processing
    y1 = 0.78
    box(col1_x, y1, box_w, box_h, "Cost Matrix C\n(n×n)", "#FFE5B4", "darkorange")
    box(col2_x, y1, box_w, box_h, "Row Features\n21 dims/row\n(stats + entropy)", "#B4D7FF", "darkblue")
    cx2, cy2 = box(col3_x, y1, box_w, box_h, "Input Projection\nLinear→GELU→LN", "#C7E9C0", "darkgreen")
    
    arrow(col1_x + box_w, y1 + box_h/2, col2_x, y1 + box_h/2)
    arrow(col2_x + box_w, y1 + box_h/2, col3_x, y1 + box_h/2, label="(n, F)")
    
    # Row 2: Residual blocks
    y2 = 0.60
    cx_res, cy_res = box(col2_x, y2, box_w, box_h, 
        f"{meta['layers']}× Residual Block\nLinear→GELU→Drop→Linear\n+Residual→LayerNorm", 
        "#E6B8FF", "purple")
    arrow(cx2, cy2 - box_h/2, cx_res, cy_res + box_h/2, label="h")
    
    # Row 3: Sparse refinement
    y3 = 0.42
    cx_pre, cy_pre = box(col1_x, y3, box_w, box_h, "Pre-output\nû = Linear(h)", "#FFD1DC", "darkred")
    cx_topk, cy_topk = box(col2_x, y3, box_w, box_h, 
        f"Top-{meta['topk']} Columns\nFind k cheapest\nC_ij - û_i", 
        "#FFF4B0", "goldenrod")
    cx_edge, cy_edge = box(col3_x, y3, box_w, box_h, 
        "Edge MLP\nEmbed→Softmax\nWeighted Sum", 
        "#FFB6C1", "crimson")
    
    arrow(cx_res, cy_res - box_h/2, cx_pre, cy_pre + box_h/2)
    arrow(cx_pre + box_w/2, cy_pre, cx_topk, cy_topk, label="û")
    arrow(col1_x + box_w/2, y1, col1_x + box_w/2, cy_topk + box_h/2, label="C")
    arrow(cx_topk + box_w/2, cy_topk, cx_edge, cy_edge)
    
    # Refinement feedback
    arrow(cx_edge, cy_edge + box_h/3, cx_res + box_w/2, cy_res - box_h/2, 
          style="-[", color="purple", linewidth=1.5, label="refine")
    
    # Row 4: Final output
    y4 = 0.20
    cx_head, cy_head = box(col1_x, y4, box_w, box_h, 
        "Row Head\nLinear→GELU→Drop→Linear", 
        "#C7E9C0", "darkgreen")
    cx_center, cy_center = box(col2_x, y4, box_w, box_h, 
        "Gauge Center\nu - mean(u)", 
        "#D7C7FF", "indigo")
    cx_min, cy_min = box(col3_x, y4, box_w, box_h, 
        "Min-Trick\nv_j = min_i(C_ij - u_i)", 
        "#FFE5B4", "darkorange")
    
    arrow(cx_res, cy_res - box_h, cx_head, cy_head + box_h/2, label="h'")
    arrow(cx_head + box_w/2, cy_head, cx_center, cy_center)
    arrow(cx_center + box_w/2, cy_center, cx_min, cy_min, label="u*")
    
    # Final outputs
    ax.text(col1_x + box_w/2, 0.08, "Row Duals (û)", ha="center", fontsize=11, 
           weight='bold', color="darkgreen")
    ax.text(col3_x + box_w/2, 0.08, "Column Duals (v̂)", ha="center", fontsize=11, 
           weight='bold', color="darkorange")
    arrow(cx_center, cy_center - box_h/2, col1_x + box_w/2, 0.11, color="darkgreen", linewidth=2.5)
    arrow(cx_min, cy_min - box_h/2, col3_x + box_w/2, 0.11, color="darkorange", linewidth=2.5)
    
    # Add annotations
    ax.text(0.5, 0.95, "Key Features: O(n·F) row encoder, no O(n²) edge processing", 
           ha="center", fontsize=10, style='italic', 
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))


def draw_forward_diagnostics(ax_u: plt.Axes, ax_topk: plt.Axes, sim: dict, topk_row: int):
    """Draw before/after dual comparison and top-k attention weights."""
    u_pre = sim["u_pre"]
    u_final = sim["u_final"]

    rows = np.arange(len(u_pre))
    ax_u.set_title("Row Duals: Coarse → Refined → Centered", fontsize=11, fontweight='bold')
    width = 0.25
    ax_u.bar(rows - width, u_pre, width=width, label="û (coarse)", color="#9ecae1", alpha=0.8)
    u_raw = sim["u_final_raw"]
    ax_u.bar(rows, u_raw, width=width, label="u (refined)", color="#6baed6", alpha=0.8)
    ax_u.bar(rows + width, u_final, width=width, label="u* (centered)", color="#3182bd", alpha=0.8)
    ax_u.set_xlabel("Row index", fontweight='bold')
    ax_u.set_ylabel("Dual value", fontweight='bold')
    ax_u.legend(fontsize=8, loc='best')
    ax_u.grid(True, alpha=0.3)

    indices = sim["refine_info"]["indices"][topk_row]
    weights = sim["refine_info"]["weights"][topk_row]
    values = sim["refine_info"]["values"][topk_row]

    ax_topk.set_title(f"Sparse Top-{len(indices)} Attention (Row {topk_row})", 
                      fontsize=11, fontweight='bold')
    bars = ax_topk.bar(np.arange(len(indices)), weights, color="#41ab5d", alpha=0.8, edgecolor='darkgreen')
    ax_topk.set_xticks(np.arange(len(indices)))
    ax_topk.set_xticklabels(indices, rotation=45, fontsize=7)
    ax_topk.set_xlabel("Selected column index", fontweight='bold')
    ax_topk.set_ylabel("Softmax weight", fontweight='bold')
    
    # Annotate with reduced costs
    for i, (val, w) in enumerate(zip(values, weights)):
        if w > 0.01:  # Only label significant weights
            ax_topk.text(i, w + 0.01, f"{val:.2f}", 
                        ha="center", va="bottom", fontsize=6, rotation=45)
    ax_topk.margins(y=0.2)
    ax_topk.grid(True, alpha=0.3, axis='y')


def draw_forward_pass_steps(fig, sim: dict, meta: dict):
    """Create detailed step-by-step forward pass visualization."""
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3,
                  left=0.08, right=0.98, top=0.95, bottom=0.05)
    
    # Step 1: Row features heatmap
    ax1 = fig.add_subplot(gs[0, :])
    row_feat = sim["row_features"]
    im1 = ax1.imshow(row_feat.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax1.set_title("Step 1: Row Feature Extraction (21 statistical features per row)", 
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel("Row index")
    ax1.set_ylabel("Feature dim")
    plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.1, aspect=30)
    
    # Step 2: Residual blocks evolution
    ax2a = fig.add_subplot(gs[1, 0])
    ax2b = fig.add_subplot(gs[1, 1])
    ax2c = fig.add_subplot(gs[1, 2])
    
    hidden_acts = sim["hidden_activations"]
    ax2a.set_title("Step 2a: Input Projection", fontsize=10, fontweight='bold')
    im2a = ax2a.imshow(hidden_acts[0][:10, :20].T, aspect='auto', cmap='coolwarm', interpolation='nearest')
    ax2a.set_xlabel("Row (first 10)")
    ax2a.set_ylabel("Hidden dim (first 20)")
    plt.colorbar(im2a, ax=ax2a, orientation='horizontal', pad=0.1, aspect=15)
    
    if len(sim["block_outputs"]) > 0:
        block_0 = sim["block_outputs"][0]
        ax2b.set_title(f"Step 2b: After Block 1 (of {len(sim['block_outputs'])})", 
                       fontsize=10, fontweight='bold')
        im2b = ax2b.imshow(block_0['after'][:10, :20].T, aspect='auto', cmap='coolwarm', 
                          interpolation='nearest')
        ax2b.set_xlabel("Row (first 10)")
        ax2b.set_ylabel("Hidden dim (first 20)")
        plt.colorbar(im2b, ax=ax2b, orientation='horizontal', pad=0.1, aspect=15)
        
        # Show residual connection impact
        ax2c.set_title("Step 2c: Residual Update (Block 1)", fontsize=10, fontweight='bold')
        residual_norm = np.linalg.norm(block_0['residual'], axis=1)
        ax2c.bar(range(len(residual_norm)), residual_norm, color='steelblue', alpha=0.7)
        ax2c.set_xlabel("Row index")
        ax2c.set_ylabel("||Residual||₂")
        ax2c.grid(True, alpha=0.3)
    
    # Step 3: Sparse refinement
    ax3a = fig.add_subplot(gs[2, 0])
    ax3b = fig.add_subplot(gs[2, 1])
    ax3c = fig.add_subplot(gs[2, 2])
    
    ax3a.set_title("Step 3a: Coarse Dual (û)", fontsize=10, fontweight='bold')
    ax3a.plot(sim["u_pre"], 'o-', color='orange', markersize=4, alpha=0.7)
    ax3a.set_xlabel("Row index")
    ax3a.set_ylabel("û value")
    ax3a.grid(True, alpha=0.3)
    
    ax3b.set_title(f"Step 3b: Top-{meta['topk']} Refinement Message", fontsize=10, fontweight='bold')
    message = sim["refine_info"]["message"]
    message_norm = np.linalg.norm(message, axis=1)
    ax3b.bar(range(len(message_norm)), message_norm, color='purple', alpha=0.7)
    ax3b.set_xlabel("Row index")
    ax3b.set_ylabel("||Message||₂")
    ax3b.grid(True, alpha=0.3)
    
    ax3c.set_title("Step 3c: Hidden State Change", fontsize=10, fontweight='bold')
    h_diff = sim["h_after_refine"] - sim["h_before_refine"]
    h_diff_norm = np.linalg.norm(h_diff, axis=1)
    ax3c.bar(range(len(h_diff_norm)), h_diff_norm, color='crimson', alpha=0.7)
    ax3c.set_xlabel("Row index")
    ax3c.set_ylabel("||Δh||₂")
    ax3c.grid(True, alpha=0.3)
    
    # Step 4: Final outputs
    ax4a = fig.add_subplot(gs[3, 0])
    ax4b = fig.add_subplot(gs[3, 1])
    ax4c = fig.add_subplot(gs[3, 2])
    
    ax4a.set_title("Step 4a: Final Row Duals (u*)", fontsize=10, fontweight='bold')
    ax4a.plot(sim["u_final"], 's-', color='darkgreen', markersize=5, alpha=0.8)
    ax4a.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='mean=0')
    ax4a.set_xlabel("Row index")
    ax4a.set_ylabel("u* (centered)")
    ax4a.legend(fontsize=7)
    ax4a.grid(True, alpha=0.3)
    
    ax4b.set_title("Step 4b: Column Duals via Min-Trick", fontsize=10, fontweight='bold')
    ax4b.plot(sim["v_trick"], '^-', color='darkorange', markersize=5, alpha=0.8)
    ax4b.set_xlabel("Column index")
    ax4b.set_ylabel("v̂ value")
    ax4b.grid(True, alpha=0.3)
    
    ax4c.set_title("Step 4c: Dual Statistics", fontsize=10, fontweight='bold')
    stats_labels = ['û mean', 'û std', 'u* mean', 'u* std', 'v̂ mean', 'v̂ std']
    stats_values = [
        np.mean(sim["u_pre"]), np.std(sim["u_pre"]),
        np.mean(sim["u_final"]), np.std(sim["u_final"]),
        np.mean(sim["v_trick"]), np.std(sim["v_trick"])
    ]
    bars = ax4c.barh(range(len(stats_labels)), stats_values, 
                     color=['orange', 'orange', 'green', 'green', 'brown', 'brown'],
                     alpha=0.7)
    ax4c.set_yticks(range(len(stats_labels)))
    ax4c.set_yticklabels(stats_labels, fontsize=8)
    ax4c.set_xlabel("Value")
    ax4c.grid(True, alpha=0.3, axis='x')
    
    for i, v in enumerate(stats_values):
        ax4c.text(v, i, f' {v:.3f}', va='center', fontsize=7)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to OneGNN checkpoint.")
    parser.add_argument("--output", type=Path, default=Path("results/onegnn_visualization.png"))
    parser.add_argument("--forward-output", type=Path, default=Path("results/onegnn_forward_pass.png"))
    parser.add_argument("--size", type=int, default=12, help="Synthetic matrix size if no dataset provided.")
    parser.add_argument("--dataset", type=Path, help="Optional HDF5 dataset to sample from.")
    parser.add_argument("--index", type=int, default=0, help="Dataset index when using --dataset.")
    parser.add_argument("--topk-row", type=int, default=0, help="Row to highlight in top-k plot.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sample cost matrix
    if args.dataset is not None:
        import h5py

        with h5py.File(args.dataset, "r") as f:
            n = int(f["n"][args.index])
            C = np.asarray(f["C"][args.index], dtype=np.float64).reshape(n, n)
        print(f"Loaded cost matrix from {args.dataset} (index {args.index}, size {n}).")
    else:
        rng = np.random.default_rng(42)
        C = rng.uniform(0.0, 1.0, size=(args.size, args.size)).astype(np.float64)
        print(f"Generated synthetic uniform cost matrix of size {args.size}×{args.size}.")

    model, meta = load_model(args.model, device)
    print("\n" + "="*70)
    print("MODEL CONFIGURATION")
    print("="*70)
    print(
        textwrap.indent(
            textwrap.dedent(
                f"""
                Hidden dimension: {meta['hidden']}
                Residual layers: {meta['layers']}
                Dropout rate: {meta['dropout']:.2f}
                Row feature dim: {meta['row_feat_dim']}
                Top-k refinement: {meta['topk']}
                Device: {device}
                """
            ).strip(),
            prefix="  ",
        )
    )
    print("="*70)

    sim = simulate_forward(model, C, device, use_cuda_features=(device.type == "cuda"))

    # Print comprehensive forward pass summary
    print("\n" + "="*70)
    print("FORWARD PASS SUMMARY")
    print("="*70)
    print(f"  Input cost matrix: {C.shape}")
    print(f"  Row features shape: {sim['row_features'].shape}")
    print(f"  Row feature stats: mean={sim['row_features'].mean():.4f}, std={sim['row_features'].std():.4f}")
    print(f"\n  Hidden activations captured at {len(sim['hidden_activations'])} stages")
    for i, h in enumerate(sim['hidden_activations']):
        print(f"    Stage {i}: shape={h.shape}, mean={h.mean():.4f}, std={h.std():.4f}")
    
    print(f"\n  Coarse dual (û):")
    print(f"    Range: [{sim['u_pre'].min():.4f}, {sim['u_pre'].max():.4f}]")
    print(f"    Mean: {sim['u_pre'].mean():.4f}, Std: {sim['u_pre'].std():.4f}")
    
    print(f"\n  Sparse refinement (top-{meta['topk']}):")
    message_norms = np.linalg.norm(sim['refine_info']['message'], axis=1)
    print(f"    Message norm per row: mean={message_norms.mean():.4f}, max={message_norms.max():.4f}")
    
    print(f"\n  Final centered dual (u*):")
    print(f"    Range: [{sim['u_final'].min():.4f}, {sim['u_final'].max():.4f}]")
    print(f"    Mean: {sim['u_final'].mean():.6f} (should be ≈0), Std: {sim['u_final'].std():.4f}")
    
    print(f"\n  Column duals via min-trick (v̂):")
    print(f"    Range: [{sim['v_trick'].min():.4f}, {sim['v_trick'].max():.4f}]")
    print(f"    Mean: {sim['v_trick'].mean():.4f}, Std: {sim['v_trick'].std():.4f}")

    row = args.topk_row
    indices = sim["refine_info"]["indices"][row]
    weights = sim["refine_info"]["weights"][row]
    values = sim["refine_info"]["values"][row]
    print(f"\n  Sparse refinement details for row {row}:")
    print(f"  {'Rank':<6} {'Column':<8} {'Reduced Cost':<15} {'Weight':<10}")
    print("  " + "-"*50)
    for idx, (col, val, w) in enumerate(zip(indices, values, weights)):
        print(f"  {idx+1:<6} {col:<8} {val:<15.4f} {w:<10.4f}")
    print("="*70)

    # Create architecture visualization
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig1 = plt.figure(figsize=(14, 10))
    gs1 = fig1.add_gridspec(2, 2, height_ratios=[3, 2], hspace=0.3)

    ax_arch = fig1.add_subplot(gs1[0, :])
    draw_architecture(ax_arch, meta)

    ax_duals = fig1.add_subplot(gs1[1, 0])
    ax_topk = fig1.add_subplot(gs1[1, 1])
    draw_forward_diagnostics(ax_duals, ax_topk, sim, row)

    fig1.tight_layout()
    fig1.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f"\n✓ Architecture visualization saved to {args.output}")

    # Create detailed forward pass visualization
    args.forward_output.parent.mkdir(parents=True, exist_ok=True)
    fig2 = plt.figure(figsize=(16, 14))
    draw_forward_pass_steps(fig2, sim, meta)
    fig2.suptitle("OneGNN Forward Pass: Step-by-Step Breakdown", 
                  fontsize=14, fontweight='bold', y=0.99)
    fig2.savefig(args.forward_output, dpi=200, bbox_inches='tight')
    print(f"✓ Forward pass visualization saved to {args.forward_output}")
    
    print(f"\n{'='*70}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
