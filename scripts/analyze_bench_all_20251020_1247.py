#!/usr/bin/env python3
import argparse, glob, re
from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------- IO ---------------------------

def load_all_csv(patterns):
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            df["__source"] = Path(fp).name
            frames.append(df)
        except Exception:
            continue
    if not frames:
        raise SystemExit("No CSV files matched the given patterns.")
    big = pd.concat(frames, ignore_index=True)
    return big

# ---------------------- Normalization ---------------------

def extract_size_col(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series 'n' for size.
    Supports either:
      - 'size_n' (int) or
      - 'problem_size' (like 'uniform_100x100' or just 100)
    """
    if "size_n" in df.columns:
        return pd.to_numeric(df["size_n"], errors="coerce")

    if "problem_size" in df.columns:
        # can be 100 or 'uniform_100x100'
        def parse_v(x):
            try:
                return int(x)
            except Exception:
                if isinstance(x, str):
                    m = re.search(r"(\d+)[xX](\d+)", x)
                    if m:
                        a, b = int(m.group(1)), int(m.group(2))
                        return a if a == b else max(a, b)
            return np.nan
        return df["problem_size"].map(parse_v)

    return pd.Series(np.nan, index=df.index)

def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a unified schema:
      ['__source','n','solver','time_ms','kind']
    kind ∈ {'algo','txfer'}
    """
    out = []

    # Case A: modular logs: solver_name + time_ms
    if {"solver_name","time_ms"}.issubset(df.columns):
        tmp = pd.DataFrame({
            "__source": df["__source"],
            "n": extract_size_col(df),
            "solver": df["solver_name"].astype(str),
            "time_ms": pd.to_numeric(df["time_ms"], errors="coerce"),
            "kind": "algo",
        })
        out.append(tmp)

    # Case B: baseline with T_total_ms / scipylap_ms
    # We materialize two solver rows: 'seeded_total' and 'scipy'
    has_seed = "T_total_ms" in df.columns
    has_scipy = "scipylap_ms" in df.columns
    if has_seed or has_scipy:
        base = pd.DataFrame({
            "__source": df["__source"],
            "n": extract_size_col(df),
        })
        if has_seed:
            seeded = base.copy()
            seeded["solver"] = "seeded_total"
            seeded["time_ms"] = pd.to_numeric(df["T_total_ms"], errors="coerce")
            seeded["kind"] = "algo"
            out.append(seeded)
        if has_scipy:
            sc = base.copy()
            sc["solver"] = "scipy"
            sc["time_ms"] = pd.to_numeric(df["scipylap_ms"], errors="coerce")
            sc["kind"] = "algo"
            out.append(sc)

    # Case C: transfer logs (T_H2D_ms / T_D2H_ms)
    if {"T_H2D_ms","T_D2H_ms"}.issubset(df.columns):
        n = pd.to_numeric(df.get("size_n", np.nan), errors="coerce")
        if n.isna().all():
            # try device-bytes schemas; size may be in 'size_n' or absent
            n = extract_size_col(df)
        for col, tag in (("T_H2D_ms","H2D"), ("T_D2H_ms","D2H")):
            tmp = pd.DataFrame({
                "__source": df["__source"],
                "n": n,
                "solver": f"txfer_{tag}",
                "time_ms": pd.to_numeric(df[col], errors="coerce"),
                "kind": "txfer",
            })
            out.append(tmp)

    if not out:
        # Nothing matched; return empty with target columns
        return pd.DataFrame(columns=["__source","n","solver","time_ms","kind"])

    return pd.concat(out, ignore_index=True)

# ---------------------- Aggregations ----------------------

def agg_algo(df_u: pd.DataFrame) -> dict:
    algo = df_u[df_u["kind"]=="algo"].dropna(subset=["n","time_ms"])
    results = {}

    # Per-file, per-n medians
    per_file = (
        algo.groupby(["__source","n","solver"], as_index=False)["time_ms"]
            .median()
            .rename(columns={"time_ms":"time_ms_median"})
            .sort_values(["__source","n","solver"])
    )
    results["per_file_medians"] = per_file

    # Per-n medians across all sources
    per_n = (
        algo.groupby(["n","solver"], as_index=False)["time_ms"]
            .median()
            .rename(columns={"time_ms":"time_ms_median"})
            .sort_values(["n","solver"])
    )
    results["per_n_medians"] = per_n

    # Speedup vs SciPy (only where we have both)
    wide = per_n.pivot(index="n", columns="solver", values="time_ms_median")
    if "scipy" in wide.columns:
        for s in wide.columns:
            if s == "scipy": 
                continue
            wide[f"speedup_{s}_vs_scipy"] = wide["scipy"] / wide[s]
    results["speedup_vs_scipy"] = wide.reset_index()

    # Global speedup distribution using raw rows where both are present within same source+n (more robust)
    pairs = (
        algo.pivot_table(index=["__source","n"], columns="solver", values="time_ms", aggfunc="median")
        .reset_index()
    )
    if {"scipy","seeded_total"}.issubset(pairs.columns):
        pairs["speedup_seeded_total_vs_scipy"] = pairs["scipy"] / pairs["seeded_total"]
        results["global_speedup_seeded"] = pairs[["__source","n","speedup_seeded_total_vs_scipy"]]

    return results

def agg_txfer(df_u: pd.DataFrame) -> dict:
    tx = df_u[df_u["kind"]=="txfer"].dropna(subset=["n","time_ms"])
    if tx.empty:
        return {"per_n_medians": pd.DataFrame()}
    per_n = (
        tx.groupby(["n","solver"], as_index=False)["time_ms"]
          .agg(median="median", mean="mean", p10=lambda x: np.percentile(x,10), p90=lambda x: np.percentile(x,90), count="count")
          .sort_values(["n","solver"])
    )
    return {"per_n_medians": per_n}

# ------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Unify heterogeneous benchmark CSVs and summarize results.")
    ap.add_argument("--inputs", nargs="+", default=["logs/performance/*.csv"], help="Glob patterns")
    ap.add_argument("--outdir", default="artifacts/reports", help="Where to save CSV summaries")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    raw = load_all_csv(args.inputs)
    uni = normalize_schema(raw)

    # Save all unified rows
    all_rows_path = f"{args.outdir}/bench_all_rows.csv"
    uni.to_csv(all_rows_path, index=False)

    # Algo summaries
    algo_res = agg_algo(uni)
    algo_res["per_file_medians"].to_csv(f"{args.outdir}/bench_algo_per_file_medians.csv", index=False)
    algo_res["per_n_medians"].to_csv(f"{args.outdir}/bench_algo_per_n_medians.csv", index=False)
    algo_res["speedup_vs_scipy"].to_csv(f"{args.outdir}/bench_algo_speedup_vs_scipy.csv", index=False)
    if "global_speedup_seeded" in algo_res:
        algo_res["global_speedup_seeded"].to_csv(f"{args.outdir}/bench_algo_global_speedup_seeded.csv", index=False)

    # Txfer summaries
    tx_res = agg_txfer(uni)
    if not tx_res["per_n_medians"].empty:
        tx_res["per_n_medians"].to_csv(f"{args.outdir}/bench_txfer_per_n_medians.csv", index=False)

    # Pretty prints
    print("\n=== Rows unified sample ===")
    print(uni.head(12).to_string(index=False))

    print("\n=== Algo per-n medians (ms) ===")
    print(algo_res["per_n_medians"].to_string(index=False))

    print("\n=== Speedup vs SciPy (per-n medians) ===")
    print(algo_res["speedup_vs_scipy"].fillna(np.nan).to_string(index=False))

    if "global_speedup_seeded" in algo_res:
        g = algo_res["global_speedup_seeded"]
        if not g.empty:
            gm = g["speedup_seeded_total_vs_scipy"].median()
            print(f"\nGlobal median speedup (seeded_total vs SciPy): {gm:.3f}x over matched (source,n) pairs.")

    if not tx_res["per_n_medians"].empty:
        print("\n=== Transfer medians (ms) ===")
        print(tx_res["per_n_medians"].to_string(index=False))

    print(f"\nSaved: {all_rows_path}")
    print(f"Saved: {args.outdir}/bench_algo_per_file_medians.csv")
    print(f"Saved: {args.outdir}/bench_algo_per_n_medians.csv")
    print(f"Saved: {args.outdir}/bench_algo_speedup_vs_scipy.csv")
    if (Path(args.outdir)/'bench_algo_global_speedup_seeded.csv').exists():
        print(f"Saved: {args.outdir}/bench_algo_global_speedup_seeded.csv")
    if (Path(args.outdir)/'bench_txfer_per_n_medians.csv').exists():
        print(f"Saved: {args.outdir}/bench_txfer_per_n_medians.csv")

