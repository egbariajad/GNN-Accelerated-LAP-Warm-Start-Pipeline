#!/usr/bin/env python3
import argparse, glob, sys, math, random
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# --- column normalization ---
SIZE_CANDIDATES = ["size_n","n","size","N"]
RENAME = {
    "Tpred_ms":"T_pred_ms","T_pred":"T_pred_ms",
    "Ttransfer_ms":"T_xfer_ms","T_xfer":"T_xfer_ms","Txfer_ms":"T_xfer_ms",
    "Trefine_ms":"T_refine_ms","Ttotal_ms":"T_total_ms","T_total":"T_total_ms",
    "scipy_ms":"scipylap_ms","SciPy_ms":"scipylap_ms",
}
NUMERIC_HINT = {"T_pred_ms","T_xfer_ms","T_refine_ms","T_total_ms","scipylap_ms"}

def ts(): return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_all_csv(patterns):
    frames = []
    for pat in patterns:
        for p in glob.glob(pat):
            try:
                df = pd.read_csv(p)
                df["__source"] = Path(p).name
                frames.append(df)
            except Exception as e:
                print(f"[warn] read {p} failed: {e}", file=sys.stderr)
    if not frames: return pd.DataFrame()
    big = pd.concat(frames, ignore_index=True)

    # rename headers
    big.columns = [RENAME.get(c.strip(), c.strip()) for c in big.columns]

    # detect size column
    size_col = next((c for c in SIZE_CANDIDATES if c in big.columns), None)
    if size_col is None:
        raise RuntimeError(f"No size column. Expected one of {SIZE_CANDIDATES}")
    big[size_col] = pd.to_numeric(big[size_col], errors="coerce")
    big = big.rename(columns={size_col:"n"})

    # coerce numerics
    for c in list(big.columns):
        if c in NUMERIC_HINT or c.endswith("_ms"):
            big[c] = pd.to_numeric(big[c], errors="coerce")

    # keep only rows having both SciPy and total time
    if "scipylap_ms" not in big.columns or "T_total_ms" not in big.columns:
        raise RuntimeError("Missing 'scipylap_ms' or 'T_total_ms' in inputs.")
    big = big.dropna(subset=["n","scipylap_ms","T_total_ms"]).reset_index(drop=True)

    # speedup ratio
    big["speedup_x"] = big["scipylap_ms"] / big["T_total_ms"]
    big["win"] = (big["T_total_ms"] < big["scipylap_ms"]).astype(int)
    return big

def bootstrap_ci(x, iters=5000, alpha=0.05, seed=123):
    rng = random.Random(seed)
    vals = [float(v) for v in x if math.isfinite(v)]
    if not vals: return (math.nan, math.nan, math.nan)
    n = len(vals)
    boots = []
    for _ in range(iters):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        boots.append(np.median(sample))
    boots.sort()
    lo = boots[int((alpha/2)*iters)]
    hi = boots[int((1-alpha/2)*iters)]
    return (float(np.median(vals)), float(lo), float(hi))

def summarize_global(df):
    m, lo, hi = bootstrap_ci(df["speedup_x"])
    win_rate = df["win"].mean() if len(df)>0 else float("nan")
    return {
        "rows": len(df),
        "median_speedup_x": m,
        "median_speedup_x_CI95_lo": lo,
        "median_speedup_x_CI95_hi": hi,
        "mean_speedup_x": df["speedup_x"].mean(),
        "win_rate": win_rate,
        "median_T_total_ms": df["T_total_ms"].median(),
        "median_scipy_ms": df["scipylap_ms"].median(),
    }

def summarize_by_n(df):
    rows = []
    for n, g in df.groupby("n"):
        m, lo, hi = bootstrap_ci(g["speedup_x"])
        rows.append({
            "n": int(n),
            "rows": len(g),
            "median_speedup_x": m,
            "median_speedup_x_CI95_lo": lo,
            "median_speedup_x_CI95_hi": hi,
            "mean_speedup_x": g["speedup_x"].mean(),
            "win_rate": g["win"].mean(),
            "median_T_total_ms": g["T_total_ms"].median(),
            "median_scipy_ms": g["scipylap_ms"].median(),
        })
    out = pd.DataFrame(rows).sort_values("n").reset_index(drop=True)
    return out

def optional_subgroups(df, keys):
    keys = [k for k in keys if k in df.columns]
    if not keys: return []
    summaries = []
    for grp_vals, g in df.groupby(keys, dropna=False):
        grp_vals = grp_vals if isinstance(grp_vals, tuple) else (grp_vals,)
        label = {k: v for k, v in zip(keys, grp_vals)}
        glob = summarize_global(g)
        byn = summarize_by_n(g)
        summaries.append((label, glob, byn))
    return summaries

def main():
    ap = argparse.ArgumentParser(description="Make speed-claim tables vs SciPy.")
    ap.add_argument("--inputs", nargs="+", required=True, help="CSV globs")
    ap.add_argument("--outdir", default="artifacts/reports", help="Output directory")
    ap.add_argument("--by", nargs="*", default=[], help="Optional subgroup columns (e.g., dtype pinned impl gpu)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = load_all_csv(args.inputs)

    # Global claim
    global_summary = summarize_global(df)
    byn = summarize_by_n(df)
    stamp = ts()
    pd.DataFrame([global_summary]).to_csv(outdir / f"claim_global_{stamp}.csv", index=False)
    byn.to_csv(outdir / f"claim_by_n_{stamp}.csv", index=False)
    df.to_csv(outdir / f"all_rows_{stamp}.csv", index=False)

    # Text narrative
    lines = []
    lines.append("Speed Claim vs SciPy (lower is better for time; higher is better for speedup).")
    lines.append(f"rows={global_summary['rows']}")
    lines.append(f"median speedup: {global_summary['median_speedup_x']:.3f}x "
                 f"(95% CI [{global_summary['median_speedup_x_CI95_lo']:.3f}, "
                 f"{global_summary['median_speedup_x_CI95_hi']:.3f}])")
    lines.append(f"mean speedup: {global_summary['mean_speedup_x']:.3f}x")
    lines.append(f"win-rate (T_total < SciPy): {100.0*global_summary['win_rate']:.1f}%")
    lines.append(f"median T_total: {global_summary['median_T_total_ms']:.2f} ms | "
                 f"median SciPy: {global_summary['median_scipy_ms']:.2f} ms")
    lines.append("")
    lines.append("By size (n):")
    for _, r in byn.iterrows():
        lines.append(
            f" n={int(r['n'])}: med speedup {r['median_speedup_x']:.3f}x "
            f"(CI [{r['median_speedup_x_CI95_lo']:.3f},{r['median_speedup_x_CI95_hi']:.3f}]), "
            f"wins {100.0*r['win_rate']:.1f}%, "
            f"T_total_med {r['median_T_total_ms']:.2f} ms vs SciPy_med {r['median_scipy_ms']:.2f} ms"
        )

    # Subgroups
    subs = optional_subgroups(df, args.by)
    for label, gsum, byn_sub in subs:
        tag = "_".join(f"{k}={label[k]}" for k in label.keys())
        byn_sub.to_csv(outdir / f"claim_by_n_{tag}_{stamp}.csv", index=False)
        lines.append("")
        lines.append(f"Subgroup: {tag}")
        lines.append(f" rows={gsum['rows']}, median speedup {gsum['median_speedup_x']:.3f}x "
                     f"(CI [{gsum['median_speedup_x_CI95_lo']:.3f},{gsum['median_speedup_x_CI95_hi']:.3f}]), "
                     f"wins {100.0*gsum['win_rate']:.1f}%")

    (outdir / f"claim_text_{stamp}.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved CSVs + narrative under: {outdir}")

if __name__ == "__main__":
    main()
