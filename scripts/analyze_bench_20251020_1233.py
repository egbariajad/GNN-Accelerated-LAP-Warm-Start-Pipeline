#!/usr/bin/env python3
import argparse, glob, sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ---------- helpers ----------
def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

SIZE_CANDIDATES = ["size_n","n","size","N"]

RENAME = {
    "Tpred_ms":"T_pred_ms",
    "T_pred":"T_pred_ms",
    "Ttransfer_ms":"T_xfer_ms",
    "T_xfer":"T_xfer_ms",
    "Txfer_ms":"T_xfer_ms",
    "Trefine_ms":"T_refine_ms",
    "Ttotal_ms":"T_total_ms",
    "T_total":"T_total_ms",
    "scipy_ms":"scipylap_ms",
    "SciPy_ms":"scipylap_ms",
    "T_HtoD_ms":"T_H2D_ms",
    "T_DtoH_ms":"T_D2H_ms",
    "BW_HtoD_GBs":"BW_H2D_GBs",
    "BW_DtoH_GBs":"BW_D2H_GBs",
}

NUMERIC_WHITELIST = {
    "T_pred_ms","T_xfer_ms","T_refine_ms","T_total_ms","scipylap_ms",
    "T_H2D_ms","T_D2H_ms","BW_H2D_GBs","BW_D2H_GBs",
    "bytes_H2D","bytes_D2H"
}

def load_all_csv(patterns):
    frames = []
    for pat in patterns:
        for p in glob.glob(pat):
            try:
                df = pd.read_csv(p)
                df["__source"] = Path(p).name
                frames.append(df)
            except Exception as e:
                print(f"[warn] failed to read {p}: {e}", file=sys.stderr)
    if not frames:
        return pd.DataFrame()
    big = pd.concat(frames, ignore_index=True)

    # normalize columns
    cols = [c.strip() for c in big.columns]
    cols = [RENAME.get(c, c) for c in cols]
    big.columns = cols

    # choose size column
    size_col = None
    for c in SIZE_CANDIDATES:
        if c in big.columns:
            size_col = c
            break
    if size_col is None:
        raise RuntimeError("No size column found. Expected one of: " + ",".join(SIZE_CANDIDATES))

    # coerce numeric columns
    for c in big.columns:
        if c == size_col or c in NUMERIC_WHITELIST or c.startswith("T_") or c.endswith("_ms") or c.endswith("_GBs"):
            big[c] = pd.to_numeric(big[c], errors="coerce")

    # ensure integer size
    big[size_col] = pd.to_numeric(big[size_col], errors="coerce").astype("Int64")

    # drop rows without size
    big = big.dropna(subset=[size_col])
    return big.rename(columns={size_col:"n"})

def pretty_table(df, float_cols=(), int_cols=(), title=None):
    if df.empty:
        return ""
    tmp = df.copy()
    for c in float_cols:
        if c in tmp.columns:
            tmp[c] = tmp[c].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
    for c in int_cols:
        if c in tmp.columns:
            tmp[c] = tmp[c].map(lambda x: f"{int(x)}" if pd.notnull(x) else "")
    s = []
    if title:
        s.append(title)
    s.append(tmp.to_string(index=False))
    return "\n".join(s)

def summarize_by_n(df, cols):
    g = df.groupby("n")
    out = pd.DataFrame({
        "count": g.size()
    })
    for c in cols:
        if c not in df.columns: 
            continue
        out[f"{c}_median"] = g[c].median()
        out[f"{c}_mean"]   = g[c].mean()
        out[f"{c}_p10"]    = g[c].quantile(0.10)
        out[f"{c}_p90"]    = g[c].quantile(0.90)
    return out.reset_index().sort_values("n")

def compute_speedup(df):
    if "scipylap_ms" in df.columns and "T_total_ms" in df.columns:
        df = df.copy()
        df["speedup_x"] = df["scipylap_ms"] / df["T_total_ms"]
    return df

def main():
    ap = argparse.ArgumentParser(description="Aggregate and print benchmark summaries by matrix size.")
    ap.add_argument("--inputs", nargs="+", required=True, help="CSV globs, e.g. logs/performance/*.csv")
    ap.add_argument("--outdir", default="artifacts/reports", help="Where to write aggregated CSVs")
    args = ap.parse_args()

    df = load_all_csv(args.inputs)
    if df.empty:
        print("No rows loaded. Check your --inputs globs.")
        sys.exit(1)

    df = compute_speedup(df)

    # what columns exist?
    pick_cols = [
        "T_pred_ms","T_xfer_ms","T_refine_ms","T_total_ms","scipylap_ms",
        "T_H2D_ms","T_D2H_ms","BW_H2D_GBs","BW_D2H_GBs","speedup_x"
    ]
    exist_cols = [c for c in pick_cols if c in df.columns]

    # per-n summaries
    agg = summarize_by_n(df, exist_cols)

    # niceties for printing
    print_sections = []

    # 1) Transfer-only if available
    tx_cols = [c for c in exist_cols if c.startswith("T_H2D") or c.startswith("T_D2H") or c.startswith("BW_")]
    if tx_cols:
        tx = agg[["n"] + [c for c in agg.columns if any(k in c for k in ["T_H2D","T_D2H","BW_"])]].copy()
        print_sections.append(pretty_table(
            tx,
            float_cols=[c for c in tx.columns if c!="n"],
            int_cols=["n"],
            title="=== Host<->Device Transfer (by n) ==="
        ))

    # 2) Solver timings if available
    solv_cols = [c for c in exist_cols if c in ["T_pred_ms","T_xfer_ms","T_refine_ms","T_total_ms","scipylap_ms","speedup_x"]]
    if solv_cols:
        sv = agg[["n"] + [c for c in agg.columns if any(k in c for k in ["T_pred_ms","T_xfer_ms","T_refine_ms","T_total_ms","scipylap_ms","speedup_x"])]].copy()
        print_sections.append(pretty_table(
            sv,
            float_cols=[c for c in sv.columns if c!="n"],
            int_cols=["n"],
            title="=== Solver Timings & Speedup (by n) ==="
        ))

    # 3) Raw counts per n
    cnt = agg[["n","count"]].copy()
    print_sections.append(pretty_table(
        cnt, int_cols=["n","count"], title="=== Sample Count (by n) ==="
    ))

    # 4) Optional per-source medians (useful to compare different runs)
    if "__source" in df.columns:
        gsrc = df.groupby(["__source","n"]).agg({
            c:"median" for c in exist_cols if c!="speedup_x"
        })
        gsrc = gsrc.reset_index().sort_values(["__source","n"])
        print_sections.append(pretty_table(
            gsrc, float_cols=[c for c in gsrc.columns if c not in ["__source","n"]], int_cols=["n"],
            title="=== Per-File Medians (source x n) ==="
        ))

    # print all sections
    print("\n\n".join(s for s in print_sections if s))

    # write aggregated CSVs with timestamp
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = timestamp()
    agg_path = outdir / f"bench_agg_by_n_{ts}.csv"
    df_path  = outdir / f"bench_all_rows_{ts}.csv"
    agg.to_csv(agg_path, index=False)
    df.to_csv(df_path, index=False)
    print(f"\nSaved: {agg_path}")
    print(f"Saved: {df_path}")

if __name__ == "__main__":
    main()
