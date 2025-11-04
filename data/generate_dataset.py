"""CLI for generating HDF5 datasets used by the DualGNN pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import scipy.optimize

if __package__ is None or __package__ == "":  # Allow running as script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.generators import SYNTHETIC_FAMILIES, SyntheticInstance, generate_synthetic_instance
from data.processors import RealInstance, iter_real_instances
from data.splits import plan_bucket
from solvers.advanced_dual import make_feasible_duals, project_feasible
from solvers.dual_computation import dual_from_matching_diff_constraints


try:  # pragma: no cover - optional dependency for import time
    import h5py
except (ImportError, ValueError) as exc:  # pragma: no cover
    raise ImportError("Install a NumPy-compatible build of h5py to use the dataset generator") from exc


@dataclass
class DatasetStats:
    split: str
    count: int
    families: Dict[str, int] = field(default_factory=dict)


class H5Writer:
    """Append-only writer that stores ragged matrices in HDF5."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "w")
        vfloat = h5py.vlen_dtype(np.float64)
        vint = h5py.vlen_dtype(np.int32)
        str_dtype = h5py.string_dtype("utf-8")

        self.datasets = {
            "C": self.file.create_dataset("C", shape=(0,), maxshape=(None,), dtype=vfloat),
            "u": self.file.create_dataset("u", shape=(0,), maxshape=(None,), dtype=vfloat),
            "v": self.file.create_dataset("v", shape=(0,), maxshape=(None,), dtype=vfloat),
            "rows": self.file.create_dataset("rows", shape=(0,), maxshape=(None,), dtype=vint),
            "cols": self.file.create_dataset("cols", shape=(0,), maxshape=(None,), dtype=vint),
            "cost": self.file.create_dataset("cost", shape=(0,), maxshape=(None,), dtype=np.float64),
            "n": self.file.create_dataset("n", shape=(0,), maxshape=(None,), dtype=np.int32),
            "family": self.file.create_dataset("family", shape=(0,), maxshape=(None,), dtype=str_dtype),
            "noise_std": self.file.create_dataset(
                "noise_std", shape=(0,), maxshape=(None,), dtype=np.float32
            ),
            "tag": self.file.create_dataset("tag", shape=(0,), maxshape=(None,), dtype=str_dtype),
        }
        self.count = 0

    def append(self, instance: SyntheticInstance) -> None:
        idx = self.count
        self.count += 1
        for dset in self.datasets.values():
            dset.resize((self.count,))

        cost = float(instance.cost[instance.rows, instance.cols].sum())

        self.datasets["C"][idx] = instance.cost.ravel()
        self.datasets["u"][idx] = instance.u
        self.datasets["v"][idx] = instance.v
        self.datasets["rows"][idx] = instance.rows
        self.datasets["cols"][idx] = instance.cols
        self.datasets["cost"][idx] = cost
        self.datasets["n"][idx] = instance.size
        self.datasets["family"][idx] = instance.family
        self.datasets["noise_std"][idx] = instance.noise_std
        self.datasets["tag"][idx] = instance.tag or ""

    def close(self) -> DatasetStats:
        self.file.attrs["records"] = self.count
        self.file.flush()
        self.file.close()
        return DatasetStats(split=self.path.stem, count=self.count, families={})


def build_real_instance(
    real: RealInstance,
    rng: np.random.Generator,
    noise_probability: float,
    noise_std: float,
) -> SyntheticInstance:
    cost = np.asarray(real.cost, dtype=np.float64)
    if cost.shape[0] != cost.shape[1]:
        raise ValueError(f"Real instance '{real.name}' is not square: {cost.shape}")

    rows, cols = scipy.optimize.linear_sum_assignment(cost)
    try:
        u, v, _ = dual_from_matching_diff_constraints(cost, rows, cols)
    except (AssertionError, RuntimeError):
        u, v = make_feasible_duals(cost)

    applied_noise = 0.0
    if noise_probability > 0.0 and rng.random() < noise_probability:
        applied_noise = noise_std
        u_noisy = u + rng.normal(0.0, noise_std, size=u.shape)
        v_noisy = v + rng.normal(0.0, noise_std, size=v.shape)
        u, v = project_feasible(cost, u_noisy, v_noisy, max_rounds=75)

    return SyntheticInstance(
        cost=cost,
        rows=rows.astype(np.int32),
        cols=cols.astype(np.int32),
        u=u.astype(np.float64),
        v=v.astype(np.float64),
        family=f"real_{real.source}",
        size=int(cost.shape[0]),
        noise_std=float(applied_noise),
        tag=real.name,
    )


def generate_dataset(
    *,
    output_dir: Path,
    sizes: Sequence[int],
    families: Sequence[str],
    instances_per_family: Dict[int, int],
    split_ratios: Dict[str, float],
    noise_probability: float,
    noise_std: float,
    dual_noise_prob: float,
    seed: int,
    or_library: Sequence[str] | None = None,
    suite_sparse: Sequence[str] | None = None,
) -> List[DatasetStats]:
    rng = np.random.default_rng(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    writers: Dict[str, H5Writer] = {
        split: H5Writer(output_dir / f"{split}.h5") for split in split_ratios
    }
    stats: Dict[str, Dict[str, int]] = {split: {} for split in writers}

    for n in sizes:
        count = instances_per_family.get(n, 0)
        if count <= 0:
            continue
        for family in families:
            if family not in SYNTHETIC_FAMILIES:
                raise KeyError(f"Unknown synthetic family '{family}'. Available: {sorted(SYNTHETIC_FAMILIES)}")
            plan = plan_bucket(count, split_ratios, rng)
            for local_idx in range(count):
                instance = generate_synthetic_instance(
                    family,
                    n,
                    rng=rng,
                    noise_probability=noise_probability,
                    noise_std=noise_std,
                    dual_noise_prob=dual_noise_prob,
                )
                split = plan[local_idx]
                writers[split].append(instance)
                stats[split][family] = stats[split].get(family, 0) + 1

    if or_library or suite_sparse:
        real_sources = list(
            iter_real_instances(or_library_paths=or_library, suite_sparse_names=suite_sparse)
        )
        if real_sources:
            plan = plan_bucket(len(real_sources), split_ratios, rng)
            for idx, real in enumerate(real_sources):
                instance = build_real_instance(
                    real,
                    rng=rng,
                    noise_probability=noise_probability,
                    noise_std=noise_std,
                )
                split = plan[idx]
                writers[split].append(instance)
                stats[split][instance.family] = stats[split].get(instance.family, 0) + 1

    summary: List[DatasetStats] = []
    for split, writer in writers.items():
        result = writer.close()
        result.families = stats[split]
        summary.append(result)
    return summary


def parse_instances_per_family(values: Sequence[str]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for item in values:
        if ":" not in item:
            raise ValueError(f"Expected 'size:count' format, got '{item}'")
        size_str, count_str = item.split(":", 1)
        mapping[int(size_str)] = int(count_str)
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[512, 1024, 1536, 2048, 3072, 4096],
        help="Problem sizes to generate",
    )
    parser.add_argument(
        "--families",
        type=str,
        nargs="+",
        default=list(SYNTHETIC_FAMILIES.keys()),
        help="Synthetic families to sample",
    )
    parser.add_argument(
        "--instances-per-family",
        type=int,
        default=32,
        help="Default number of instances per size per family",
    )
    parser.add_argument(
        "--instances-map",
        type=str,
        nargs="*",
        default=(),
        help="Optional overrides in 'size:count' form",
    )
    parser.add_argument("--noise-prob", type=float, default=0.2, help="Legacy parameter (reserved for future cost noise)")
    parser.add_argument("--noise-std", type=float, default=0.15, help="Standard deviation for dual noise")
    parser.add_argument("--dual-noise-prob", type=float, default=0.0, help="Probability of dual noise injection (0.0 = clean duals)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--or-library", type=str, nargs="*", default=None)
    parser.add_argument("--suite-sparse", type=str, nargs="*", default=None)
    parser.add_argument("--summary", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    per_family = {size: args.instances_per_family for size in args.sizes}
    sizes = list(args.sizes)
    if args.instances_map:
        overrides = parse_instances_per_family(args.instances_map)
        for size, count in overrides.items():
            per_family[size] = count
        sizes = sorted(set(sizes) | set(overrides.keys()))

    ratios = {"train": args.train, "val": args.val, "test": args.test}

    summary = generate_dataset(
        output_dir=args.output_dir,
        sizes=sizes,
        families=args.families,
        instances_per_family=per_family,
        split_ratios=ratios,
        noise_probability=args.noise_prob,
        noise_std=args.noise_std,
        dual_noise_prob=args.dual_noise_prob,
        seed=args.seed,
        or_library=args.or_library,
        suite_sparse=args.suite_sparse,
    )

    if args.summary:
        as_json = [asdict(item) for item in summary]
        print(json.dumps(as_json, indent=2))
    else:
        for item in summary:
            target = args.output_dir / f"{item.split}.h5"
            fam_desc = ", ".join(f"{k}:{v}" for k, v in sorted(item.families.items()))
            print(f"[{item.split}] {item.count} instances -> {target} ({fam_desc})")


if __name__ == "__main__":
    main()
