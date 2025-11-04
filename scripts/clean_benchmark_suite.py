#!/usr/bin/env python3
"""
Clean Benchmark Suite

Runs pairwise comparisons for specified OneGNN checkpoints on the CLEAN test
datasets and writes results to files. Also compares progressive models against
size-equivalent OneGNN baselines.

Outputs per-pair JSON files and a summary text file with winners.
"""

import os
import sys
import json
import time
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Optional

# Project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Thread limits for reproducibility/fairness
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np
import h5py

# Solvers
from solvers import SciPySolver, SeededLAPSolver, LAPSolver, time_solver_rigorous

# Reuse predictor from the standard benchmark to support full/reduced features
import importlib.util
_bench_spec = importlib.util.spec_from_file_location(
    "gnn_benchmark_mod", str(PROJECT_ROOT / "scripts/gnn_benchmark.py")
)
_bench_mod = importlib.util.module_from_spec(_bench_spec)
assert _bench_spec and _bench_spec.loader, "Failed to locate gnn_benchmark.py"
_bench_spec.loader.exec_module(_bench_mod)  # type: ignore
GNNPredictor = _bench_mod.GNNPredictor


@dataclass
class InstanceResult:
    n: int
    success: bool
    scipy_time: float | None = None
    lap_time: float | None = None
    gnn_time: float | None = None
    seeded_time: float | None = None
    pipeline_time: float | None = None
    speedup: float | None = None
    lap_speedup: float | None = None
    u_mae: float | None = None
    v_mae: float | None = None
    error: str | None = None


def load_clean_instances(problem_sizes: Sequence[int], max_instances_per_size: int) -> Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Load instances from CLEAN datasets for the requested sizes.

    Returns a dict: size -> list of (C, u_true, v_true)
    """
    data_dir = PROJECT_ROOT / "data" / "generated" / "processed_clean"
    paths = {
        512: data_dir / "small" / "full" / "test.h5",
        1024: data_dir / "small" / "full" / "test.h5",
        1536: data_dir / "mid_1536" / "full" / "test.h5",
        2048: data_dir / "mid_2048" / "full" / "test.h5",
        3072: data_dir / "mid_3072" / "full" / "test.h5",
        4096: data_dir / "large_4096" / "full" / "test.h5",
    }

    instances: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {s: [] for s in problem_sizes}
    for size in problem_sizes:
        h5_path = paths.get(size)
        if not h5_path or not h5_path.exists():
            print(f"⚠️ Clean test file not found for size {size}: {h5_path}")
            continue

        with h5py.File(h5_path, "r") as f:
            n_total = len(f["n"]) if "n" in f else len(f["C"])  # fallback
            count = 0
            for i in range(n_total):
                n = int(f["n"][i]) if "n" in f else size
                if n != size:
                    continue
                C_flat = f["C"][i]
                u_true = f["u"][i]
                v_true = f["v"][i]
                C = C_flat.reshape(n, n)
                instances[size].append((C, u_true, v_true))
                count += 1
                if count >= max_instances_per_size:
                    break
        print(f"Loaded {len(instances[size])} clean instances for n={size} from {h5_path}")
    return instances


def eval_model_on_instances(model_path: Path, instances: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]) -> List[InstanceResult]:
    predictor = GNNPredictor(str(model_path))
    scipy_solver = SciPySolver()
    lap_solver = LAPSolver()
    seeded_solver = SeededLAPSolver()

    results: List[InstanceResult] = []

    for size in sorted(instances.keys()):
        for (C, u_true, v_true) in instances[size]:
            r = InstanceResult(n=size, success=True)
            try:
                t_sc = time_solver_rigorous(lambda: scipy_solver.solve(C))
                if not t_sc["success"]:
                    r.success = False
                    r.error = f"SciPy failed: {t_sc.get('error')}"
                    results.append(r)
                    continue
                r.scipy_time = float(t_sc["median"])  # seconds

                t_lap = time_solver_rigorous(lambda: lap_solver.solve(C))
                if not t_lap["success"]:
                    r.success = False
                    r.error = f"LAP failed: {t_lap.get('error')}"
                    results.append(r)
                    continue
                r.lap_time = float(t_lap["median"])  # seconds

                t0 = time.time()
                u_pred, v_pred = predictor.predict(C)
                r.gnn_time = time.time() - t0

                t_seed = time_solver_rigorous(lambda: seeded_solver.solve(C, u_pred, v_pred))
                if not t_seed["success"]:
                    r.success = False
                    r.error = f"Seeded LAP failed: {t_seed.get('error')}"
                    results.append(r)
                    continue
                r.seeded_time = float(t_seed["median"])  # seconds
                r.pipeline_time = r.gnn_time + r.seeded_time
                r.speedup = (r.scipy_time or 0.0) / (r.pipeline_time or 1e-12)
                r.lap_speedup = (r.lap_time or 0.0) / (r.pipeline_time or 1e-12)
                r.u_mae = float(np.mean(np.abs(u_pred - u_true)))
                r.v_mae = float(np.mean(np.abs(v_pred - v_true)))
            except Exception as e:  # noqa: BLE001
                r.success = False
                r.error = str(e)
            results.append(r)

    return results


def summarize_results(results: List[InstanceResult]) -> Dict[str, float]:
    ok = [r for r in results if r.success and r.speedup is not None]
    if not ok:
        return {
            "count": 0,
            "mean_speedup": float("nan"),
            "lap_mean_speedup": float("nan"),
            "success_rate": 0.0,
        }
    speedups = [r.speedup for r in ok if r.speedup is not None]
    lap_speedups = [r.lap_speedup for r in ok if r.lap_speedup is not None]
    return {
        "count": float(len(ok)),
        "mean_speedup": float(statistics.mean(speedups)) if speedups else float("nan"),
        "lap_mean_speedup": float(statistics.mean(lap_speedups)) if lap_speedups else float("nan"),
        "success_rate": float(100.0 * len(ok) / len(results)) if results else 0.0,
    }


def run_pair(
    name: str,
    model_a: str,
    model_b: str,
    sizes: Sequence[int],
    max_instances: int,
    checkpoint_dirs: Sequence[Path],
    out_dir: Path,
) -> Dict[str, object]:
    print(f"\n=== Running pair: {name} ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model paths
    def resolve_model(m: str) -> Optional[Path]:
        p = Path(m)
        if p.exists():
            return p
        for d in checkpoint_dirs:
            cand = d / m
            if cand.exists():
                return cand
        return None

    mA = resolve_model(model_a)
    mB = resolve_model(model_b)
    if not mA or not mB:
        raise FileNotFoundError(f"Model not found. A={model_a} -> {mA}, B={model_b} -> {mB}")

    instances = load_clean_instances(sizes, max_instances)
    results_a = eval_model_on_instances(mA, instances)
    results_b = eval_model_on_instances(mB, instances)

    summary_a = summarize_results(results_a)
    summary_b = summarize_results(results_b)

    # Winner by mean_speedup (higher is better)
    winner = model_a
    if (summary_b["mean_speedup"] or float("nan")) > (summary_a["mean_speedup"] or float("nan")):
        winner = model_b

    payload = {
        "pair": name,
        "sizes": list(sizes),
        "max_instances": max_instances,
        "models": {
            model_a: {
                "path": str(mA),
                "summary": summary_a,
                "results": [asdict(r) for r in results_a],
            },
            model_b: {
                "path": str(mB),
                "summary": summary_b,
                "results": [asdict(r) for r in results_b],
            },
        },
        "winner": winner,
    }

    out_path = out_dir / f"{name}.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved pair results -> {out_path}")

    return payload


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run clean benchmark suite pairs")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["all"],
        help="Which pairs to run: all | small | mid1536 | mid2048 | mid3072 | progressive",
    )
    args = parser.parse_args()
    # Where checkpoints may live
    checkpoint_dirs = [
        PROJECT_ROOT / "gnn" / "checkpoints_clean",
        PROJECT_ROOT / "gnn" / "checkpoints",
        PROJECT_ROOT / "checkpoints",
    ]

    out_dir = PROJECT_ROOT / "logs" / "clean_benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    max_instances = int(os.environ.get("CLEAN_BENCH_MAX_INSTANCES", "1000000"))

    # Pairs to compare (key -> tuple)
    PAIRS: dict[str, tuple[str, str, str, list[int]]] = {
        "mid1536": (
            "mid1536_full_clean_h192L4_vs_default",
            "one_gnn_mid1536_full_clean_h192L4.pt",
            "one_gnn_mid1536_full_clean.pt",
            [1536],
        ),
        "small": (
            "small_full_clean_h192L4_vs_default",
            "one_gnn_small_full_clean_h192L4.pt",
            "one_gnn_small_full_clean.pt",
            [512, 1024],
        ),
        "mid2048": (
            "mid2048_full_clean_h192L4_vs_default",
            "one_gnn_mid2048_full_clean_h192L4.pt",
            "one_gnn_mid2048_full_clean.pt",
            [2048],
        ),
        "mid3072": (
            "mid3072_full_clean_h192L4_vs_default",
            "one_gnn_mid3072_full_clean_h192L4.pt",
            "one_gnn_mid3072_full_clean.pt",
            [3072],
        ),
        "progressive": (
            "progressive_best_vs_last",
            "progressive_clean_best.pt",
            "progressive_clean_last.pt",
            [512, 1024, 1536, 2048, 3072],
        ),
    }

    selected_keys = set([k.lower() for k in args.pairs])
    if "all" in selected_keys:
        selected = list(PAIRS.values())
    else:
        unknown = [k for k in selected_keys if k not in PAIRS]
        if unknown:
            raise SystemExit(f"Unknown pair keys: {unknown}. Valid keys: {list(PAIRS.keys())} or 'all'")
        selected = [PAIRS[k] for k in selected_keys]

    # Run all pairs and collect winners
    results_by_pair: Dict[str, Dict[str, object]] = {}
    winners: Dict[str, str] = {}
    for name, mA, mB, sizes in selected:
        payload = run_pair(name, mA, mB, sizes, max_instances, checkpoint_dirs, out_dir)
        results_by_pair[name] = payload
        winners[name] = str(payload["winner"])

    # Progressive vs size-equivalent baselines comparison
    # Compare progressive_best (winner of progressive pair) against winners of size-specific OneGNN pairs per size
    summary_lines: List[str] = []
    summary_lines.append("CLEAN BENCHMARK SUITE SUMMARY")
    summary_lines.append("")
    for name, winner in winners.items():
        summary_lines.append(f"Pair {name}: winner = {winner}")

    # Map size -> baseline winner from the 4 size-specific pairs
    baseline_by_size: Dict[int, str] = {}
    size_pair_map = {
        1536: "mid1536_full_clean_h192L4_vs_default",
        512: "small_full_clean_h192L4_vs_default",
        1024: "small_full_clean_h192L4_vs_default",
        2048: "mid2048_full_clean_h192L4_vs_default",
        3072: "mid3072_full_clean_h192L4_vs_default",
    }
    for sz, pair_key in size_pair_map.items():
        baseline_by_size[sz] = winners.get(pair_key, "")

    # Progressive pair payload
    prog_payload = results_by_pair.get("progressive_best_vs_last")
    if prog_payload:
        # Identify progressive "best" model name (winner of the pair)
        prog_best_name = winners.get("progressive_best_vs_last", "progressive_clean_best.pt")
        prog_results = prog_payload["models"][prog_best_name]["results"]  # type: ignore[index]
        # Aggregate by size
        per_size: Dict[int, List[float]] = {}
        for r in prog_results:  # type: ignore[assignment]
            if r.get("success") and r.get("n"):
                per_size.setdefault(int(r["n"]), []).append(float(r.get("speedup") or 0.0))

        summary_lines.append("")
        summary_lines.append("Progressive vs Baseline (mean speedup by size):")
        for sz in sorted(per_size.keys()):
            prog_mean = statistics.mean(per_size[sz]) if per_size[sz] else float("nan")
            baseline_model = baseline_by_size.get(sz, "")
            base_mean = float("nan")
            if baseline_model:
                # Look up baseline results: find pair containing this model
                for pair_key, payload in results_by_pair.items():
                    models = payload.get("models", {})  # type: ignore[assignment]
                    if baseline_model in models:
                        # collect only size sz
                        res = [rr for rr in models[baseline_model]["results"] if rr.get("success") and rr.get("n") == sz]  # type: ignore[index]
                        vals = [float(rr.get("speedup") or 0.0) for rr in res]
                        if vals:
                            base_mean = statistics.mean(vals)
                        break
            delta = prog_mean - base_mean if (not np.isnan(prog_mean) and not np.isnan(base_mean)) else float("nan")
            summary_lines.append(f"  n={sz}: progressive={prog_mean:.3f}x, baseline={base_mean:.3f}x, diff={delta:.3f}x")

    # Write summary
    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"\nWrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
