#!/usr/bin/env python3
import argparse, time, random, sys
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

# Ensure project root is on sys.path so we can import features.py and one_gnn.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import compute_row_features, ROW_FEATURE_DIM
from one_gnn import OneGNN

def load_h5_indices(h5_path: Path, size: int, max_k: int):
    with h5py.File(h5_path, "r") as f:
        n_all = f["n"][:]
        idxs = np.where(n_all == size)[0].tolist()
    random.shuffle(idxs)
    return [(h5_path, i) for i in idxs[:max_k]]

def find_samples(data_root: Path, size: int, per_size: int):
    pairs = []
    if data_root.is_file() and data_root.suffix == ".h5":
        pairs += load_h5_indices(data_root, size, per_size)
    else:
        for p in sorted(data_root.rglob("*.h5")):
            need = per_size - len(pairs)
            if need <= 0: break
            pairs += load_h5_indices(p, size, need)
    return pairs[:per_size]

def read_cost(h5_path: Path, idx: int) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        n = int(f["n"][idx])
        C = np.array(f["C"][idx]).reshape(n, n).astype(np.float32, copy=False)
    return C

@torch.no_grad()
def predict_u(model: OneGNN, C: np.ndarray, device: torch.device):
    t0 = time.perf_counter()
    feats = compute_row_features(C)  # [n, F]
    x = torch.from_numpy(feats).to(device=device, dtype=torch.float32).unsqueeze(0)
    u = model(x)["u"].squeeze(0).detach().cpu().numpy()
    t1 = time.perf_counter()
    return u, (t1 - t0) * 1000.0

def project_v(C: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.min(C - u[:, None], axis=0)

def prune_by_topk(C: np.ndarray, u: np.ndarray, v: np.ndarray, k: int) -> np.ndarray:
    n = C.shape[0]
    if k >= n: return C
    big = np.float32(1e6)
    R = C - u[:, None] - v[None, :]
    idx = np.argpartition(R, kth=k-1, axis=1)[:, :k]
    keep = np.full_like(C, False, dtype=bool)
    keep[np.arange(n)[:, None], idx] = True
    Cp = np.full_like(C, big)
    Cp[keep] = C[keep]
    return Cp

def run_scipy(C: np.ndarray) -> tuple[float, float]:
    t0 = time.perf_counter()
    r, c = linear_sum_assignment(C)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, float(C[r, c].sum())

def load_checkpoint_to(model: nn.Module, ckpt_path: Path):
    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj.get("model_state_dict", None)
    if state is None: state = obj
    model.load_state_dict(state, strict=False)

def main():
    ap = argparse.ArgumentParser(description="OneGNN vs SciPy benchmark")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    ap.add_argument("--per-size", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = OneGNN(in_dim=int(ROW_FEATURE_DIM), hidden=args.hidden, layers=args.layers, dropout=args.dropout).to(device)
    load_checkpoint_to(model, args.ckpt)
    model.eval()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = args.out or Path("logs/performance") / f"onegnn_speed_{ts}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write(f"{ts},{n},{t},{tp:.4f},{tr:.4f},{tt:.4f},{sc_ms:.4f},{s:.4f},{args.topk},{args.ckpt.name},{device.type},{h5p.name}\n")
")
        for n in args.sizes:
            pairs = find_samples(args.data, n, args.per_size)
            print(f"== {n}x{n} ==")
            if not pairs:
                print("No instances found for this size."); continue
            t_pred=[], t_ref=[], t_tot=[], t_sci=[], sp=[]
            for t,(h5p,idx) in enumerate(pairs,1):
                C = read_cost(h5p, idx)
                sc_ms,_ = run_scipy(C)
                u,tp = predict_u(model, C, device)
                v = project_v(C, u)
                Cp = prune_by_topk(C, u, v, args.topk)
                tr,_ = run_scipy(Cp)
                tt = tp + tr
                s = sc_ms/tt if tt>0 else float("nan")
                print(f"  [{t}/{len(pairs)}] SciPy={sc_ms:8.2f} ms | pred={tp:6.2f} ms + refine={tr:7.2f} ms => total={tt:7.2f} ms | speedup={s:0.2f}x")
        f.write(f"{ts},{n},{t},{tp:.4f},{tr:.4f},{tt:.4f},{sc_ms:.4f},{s:.4f},{args.topk},{args.ckpt.name},{device.type},{h5p.name}\n")
")
        t_pred.append(tp); t_ref.append(tr); t_tot.append(tt); t_sci.append(sc_ms); sp.append(s)
            def stats(a):
                a = np.asarray(a, float); 
                return a.mean(), a.min(), a.max()
            mp,ap,bp = stats(t_pred)
            mr,ar,br = stats(t_ref)
            mt,at,bt = stats(t_tot)
            ms,as_,bs = stats(sp)
            msci, asci, bsci = stats(t_sci)
            print(f"Summary {n}: pred mean {mp:0.2f} ms, refine mean {mr:0.2f} ms, total mean {mt:0.2f} ms, SciPy mean {msci:0.2f} ms, speedup mean {ms:0.2f}x")
            print()
    print(f"Saved CSV to {out_csv}")

if __name__ == "__main__":
    main()
