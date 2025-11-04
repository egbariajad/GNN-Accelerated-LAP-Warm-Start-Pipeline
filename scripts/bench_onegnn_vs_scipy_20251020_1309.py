#!/usr/bin/env python3
import argparse, time, math, random, json
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np

import torch
from torch import nn

try:
    from features import compute_row_features, ROW_FEATURE_DIM
except Exception as e:
    raise RuntimeError("Could not import features.py (compute_row_features, ROW_FEATURE_DIM).") from e

try:
    from one_gnn import OneGNN
except Exception as e:
    raise RuntimeError("Could not import one_gnn.py (OneGNN).") from e

try:
    from scipy.optimize import linear_sum_assignment
except Exception as e:
    raise RuntimeError("SciPy is required: from scipy.optimize import linear_sum_assignment") from e


def load_h5_indices(h5_path: Path, size: int, max_k: int):
    with h5py.File(h5_path, "r") as f:
        n_all = f["n"][:]
        idxs = np.where(n_all == size)[0]
        if len(idxs) == 0:
            return []
        idxs = idxs.tolist()
        random.shuffle(idxs)
        return [(h5_path, i) for i in idxs[:max_k]]

def find_samples(data_root: Path, size: int, per_size: int):
    cand = []
    if data_root.is_file() and data_root.suffix == ".h5":
        cand += load_h5_indices(data_root, size, per_size)
    else:
        for p in sorted(data_root.rglob("*.h5")):
            cand += load_h5_indices(p, size, per_size - len(cand))
            if len(cand) >= per_size:
                break
    return cand[:per_size]

def read_cost(h5_path: Path, idx: int) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        n = int(f["n"][idx])
        C = np.array(f["C"][idx]).reshape(n, n).astype(np.float32, copy=False)
    return C

@torch.no_grad()
def predict_u(model: OneGNN, C: np.ndarray, device: torch.device):
    t0 = time.perf_counter()
    row_feat = compute_row_features(C)  # [n, F]
    x = torch.from_numpy(row_feat).to(device=device, dtype=torch.float32).unsqueeze(0)  # [1, n, F]
    out = model(x)["u"].squeeze(0).detach().cpu().numpy()  # [n]
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000.0

def project_v(C: np.ndarray, u: np.ndarray) -> np.ndarray:
    # v_j = min_i (C_ij - u_i)
    return np.min(C - u[:, None], axis=0)

def prune_by_topk(C: np.ndarray, u: np.ndarray, v: np.ndarray, k: int) -> np.ndarray:
    """Keep the k best columns per row in reduced costs; others set to large."""
    n = C.shape[0]
    big = np.float32(1e6)
    R = C - u[:, None] - v[None, :]
    if k >= n:
        return C  # nothing to prune
    # indices of k smallest per row
    idx = np.argpartition(R, kth=k-1, axis=1)[:, :k]
    mask = np.full_like(C, True, dtype=bool)
    rows = np.arange(n)[:, None]
    mask[rows, idx] = False  # False = keep
    C_masked = C.copy()
    C_masked[mask] = big
    return C_masked

def run_scipy(C: np.ndarray) -> tuple[float, float]:
    t0 = time.perf_counter()
    r, c = linear_sum_assignment(C)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, float(C[r, c].sum())

def load_checkpoint_to(model: nn.Module, ckpt_path: Path):
    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj.get("model_state_dict", None)
    if state is None:
        state = obj
    model.load_state_dict(state, strict=False)

def main():
    ap = argparse.ArgumentParser(description="OneGNN vs SciPy benchmark (per-size summaries + CSV).")
    ap.add_argument("--data", type=Path, required=True, help="Root dir or a single .h5")
    ap.add_argument("--ckpt", type=Path, required=True, help="OneGNN checkpoint (.pt)")
    ap.add_argument("--sizes", type=int, nargs="+", required=True, help="Matrix sizes to evaluate")
    ap.add_argument("--per-size", type=int, default=5, help="Instances per size")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=16, help="Pruning k per row (reduced-cost top-k)")
    ap.add_argument("--out", type=Path, default=None, help="CSV path (default under logs/performance)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = OneGNN(in_dim=int(ROW_FEATURE_DIM), hidden=args.hidden, layers=args.layers, dropout=args.dropout).to(device)
    load_checkpoint_to(model, args.ckpt)
    model.eval()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = args.out or Path("logs/performance") / f"onegnn_speed_{ts}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w") as f:
        f.write("timestamp,size_n,trial,T_pred_ms,T_refine_ms,T_total_ms,scipylap_ms,speedup_x,prune_k,ckpt,device,source\n")

        for size in args.sizes:
            pairs = find_samples(args.data, size, args.per_size)
            if len(pairs) == 0:
                print(f"== {size}x{size} Problems ==")
                print("No instances of this size found.")
                continue

            print(f"== {size}x{size} Problems ==")
            pred_times, refine_times, total_times, scipys, speeds = [], [], [], [], []

            for t, (h5p, idx) in enumerate(pairs, start=1):
                C = read_cost(h5p, idx)

                # Baseline SciPy on full C
                sc_ms, _ = run_scipy(C)

                # OneGNN prediction
                u, t_pred = predict_u(model, C, device)
                v = project_v(C, u)

                # Prune and refine with SciPy
                C_pruned = prune_by_topk(C, u, v, k=args.topk)
                t_refine, _ = run_scipy(C_pruned)
                t_total = t_pred + t_refine

                speed = sc_ms / t_total if t_total > 0 else float("nan")

                pred_times.append(t_pred)
                refine_times.append(t_refine)
                total_times.append(t_total)
                scipys.append(sc_ms)
                speeds.append(speed)

                print(f"Instance {t}/{len(pairs)}")
                print(f"  SciPy baseline... {sc_ms:8.2f} ms")
                print(f"  GNN prediction... {t_pred:8.2f} ms + refine {t_refine:8.2f} ms = {t_total:8.2f} ms total")
                print(f"  Speedup vs SciPy: {speed:0.2f}x")

                f.write(
                    f"{ts},{size},{t},{t_pred:.4f},{t_refine:.4f},{t_total:.4f},{sc_ms:.4f},{speed:.4f},{args.topk},{args.ckpt.name},{device.type},{h5p.name}\n"
                )

            # per-size summary
            def s(v):
                v = np.array(v, dtype=float)
                return np.mean(v), np.min(v), np.max(v), np.std(v)

            m_pred, a_pred, b_pred, sd_pred = s(pred_times)
            m_ref, a_ref, b_ref, sd_ref = s(refine_times)
            m_tot, a_tot, b_tot, sd_tot = s(total_times)
            m_spd, a_spd, b_spd, sd_spd = s(speeds)
            m_sci, a_sci, b_sci, sd_sci = s(scipys)

            print()
            print(f"{size}x{size} Summary: {len(pairs)} instances")
            print(f"  GNN inference: mean {m_pred:0.2f} ms, range {a_pred:0.2f}-{b_pred:0.2f} ms")
            print(f"  Refine (SciPy on pruned): mean {m_ref:0.2f} ms, range {a_ref:0.2f}-{b_ref:0.2f} ms")
            print(f"  Total pipeline: mean {m_tot:0.2f} ms, range {a_tot:0.2f}-{b_tot:0.2f} ms")
            print(f"  SciPy baseline: mean {m_sci:0.2f} ms, range {a_sci:0.2f}-{b_sci:0.2f} ms")
            print(f"  Speedup vs SciPy: mean {m_spd:0.2f}x, range {a_spd:0.2f}x–{b_spd:0.2f}x")
            print()

    print(f"Saved CSV to {out_csv}")

if __name__ == "__main__":
    main()
