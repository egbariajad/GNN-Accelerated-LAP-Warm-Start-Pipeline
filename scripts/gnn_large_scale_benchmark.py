#!/usr/bin/env python3
"""
GNN Large-Scale Benchmark

Tests GNN models on very large problem sizes (4096-16384) to evaluate scaling behavior.
Generates synthetic test instances on-the-fly since pre-generated datasets may not exist.
"""

import os
import sys
import argparse
from pathlib import Path
import time
import statistics
from typing import Dict, List, Tuple, Optional, Sequence
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set thread limits for fair comparison
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1") 
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np

# Import torch and GNN components
try:
    import torch
    from torch import nn
    from torch.amp import autocast
    torch_available = True
except ImportError:
    torch_available = False
    print("❌ PyTorch not available - cannot run GNN inference")

# Import solvers
from solvers import SciPySolver, SeededLAPSolver, LAPSolver, time_solver_rigorous

# Import GNN components
if torch_available:
    from gnn import DualGNN, OneGNN, compute_features, compute_row_features, compute_row_features_torch

warnings.filterwarnings('ignore')


class GNNPredictor:
    """Wrapper for loading and running GNN inference."""
    
    def __init__(self, model_path: str, device: str = None):
        self.model_path = model_path
        # Auto-detect best device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model = None
        self.model_info = {}
        self.row_only = False
        self.use_cuda = 'cuda' in self.device
        self._load_model()
    
    def _load_model(self):
        """Load the GNN model from checkpoint."""
        if not torch_available:
            raise RuntimeError("PyTorch not available")
            
        print(f"Loading model from: {Path(self.model_path).name}")
        print(f"  Device: {self.device} {'(GPU)' if 'cuda' in self.device else '(CPU)'}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            architecture = checkpoint.get('architecture', 'dual_gnn')
            row_feat_dim = checkpoint.get('row_feat_dim')
            
            if 'features' in checkpoint:
                features = checkpoint['features']
            elif row_feat_dim == 17:
                features = 'reduced'
            elif row_feat_dim == 21:
                features = 'full'
            else:
                features = 'full'
            
            self.model_info = {
                'architecture': architecture,
                'hidden_dim': checkpoint.get('hidden_dim', 128),
                'layers': checkpoint.get('layers', 4),
                'dropout': checkpoint.get('dropout', 0.1),
                'heads': checkpoint.get('heads', 4),
                'row_feat_dim': row_feat_dim,
                'features': features,
            }

            nested_cfg = checkpoint.get('config')
            if isinstance(nested_cfg, dict):
                self.model_info['architecture'] = nested_cfg.get('architecture', self.model_info['architecture'])
                self.model_info['hidden_dim'] = nested_cfg.get('hidden_dim', self.model_info['hidden_dim'])
                self.model_info['layers'] = nested_cfg.get('layers', self.model_info['layers'])
                self.model_info['dropout'] = nested_cfg.get('dropout', self.model_info['dropout'])
                self.model_info['heads'] = nested_cfg.get('heads', self.model_info['heads'])
                self.model_info['row_feat_dim'] = nested_cfg.get('row_feat_dim', self.model_info['row_feat_dim'])
                if nested_cfg.get('row_feat_dim') == 21:
                    self.model_info['features'] = 'full'
        else:
            state_dict = checkpoint
            self.model_info = {
                'architecture': 'dual_gnn',
                'hidden_dim': 128,
                'layers': 4,
                'dropout': 0.1,
                'heads': 4,
                'row_feat_dim': None,
            }

        architecture = self.model_info['architecture']
        if architecture == 'one_gnn':
            in_dim = self.model_info.get('row_feat_dim') or compute_row_features(np.zeros((1, 1))).shape[1]
            self.model = OneGNN(
                in_dim=in_dim,
                hidden=self.model_info['hidden_dim'],
                layers=self.model_info['layers'],
                dropout=self.model_info['dropout'],
            )
            self.row_only = True
        else:
            self.model = DualGNN(
                hidden_dim=self.model_info['hidden_dim'],
                layers=self.model_info['layers'],
                heads=self.model_info['heads'],
                dropout=self.model_info['dropout'],
            )
            self.row_only = False

        self.amp_enabled = self.use_cuda and not self.row_only

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        print(f"  Config: {self.model_info}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        if 'cuda' in self.device:
            self._gpu_warmup()
    
    def _gpu_warmup(self):
        """Perform GPU warmup to avoid first-inference overhead."""
        print("  GPU warmup...", end=' ')
        
        dummy_size = 64
        dummy_C = np.random.rand(dummy_size, dummy_size).astype(np.float64)
        
        for _ in range(3):
            with torch.inference_mode():
                if self.row_only:
                    if 'cuda' in self.device:
                        cost_tensor = torch.from_numpy(dummy_C).float().to(self.device)
                        row_feat_tensor = compute_row_features_torch(cost_tensor)
                        row_tensor = row_feat_tensor.unsqueeze(0)
                        cost_batch = cost_tensor.unsqueeze(0)
                    else:
                        row_feat = compute_row_features(dummy_C)
                        row_tensor = torch.from_numpy(row_feat).float().unsqueeze(0).to(self.device)
                        cost_batch = torch.from_numpy(dummy_C).float().unsqueeze(0).to(self.device)
                    
                    mask = torch.ones((1, dummy_size), dtype=torch.bool, device=self.device)
                    with autocast('cuda', enabled=self.amp_enabled):
                        _ = self.model(row_tensor, cost=cost_batch, mask=mask)
                else:
                    features = compute_features(dummy_C)
                    edge_feat = torch.from_numpy(features.edge_feat).float().unsqueeze(0).to(self.device)
                    row_feat = torch.from_numpy(features.row_feat).float().unsqueeze(0).to(self.device)
                    col_feat = torch.from_numpy(features.col_feat).float().unsqueeze(0).to(self.device)
                    mask = torch.ones((1, dummy_size), dtype=torch.bool, device=self.device)
                    with autocast('cuda', enabled=self.use_cuda):
                        _ = self.model(edge_feat, row_feat, col_feat, mask=mask)
        
        if 'cuda' in self.device:
            torch.cuda.synchronize()
        
        print("done")
    
    def predict(self, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict dual potentials for a cost matrix."""
        if not torch_available or self.model is None:
            raise RuntimeError("Model not loaded")
        
        C = np.asarray(C, dtype=np.float64)
        n = C.shape[0]
        
        cost_tensor_on_gpu = None  # Track if we have C on GPU for V computation reuse

        with torch.inference_mode():
            if self.row_only:
                if 'cuda' in self.device:
                    cost_tensor = torch.from_numpy(C).float().to(self.device)
                    cost_tensor_on_gpu = cost_tensor  # Save reference for V computation
                    row_feat_tensor = compute_row_features_torch(cost_tensor)
                    row_tensor = row_feat_tensor.unsqueeze(0)
                    cost_batch = cost_tensor.unsqueeze(0)
                else:
                    row_feat = compute_row_features(C)
                    row_tensor = torch.from_numpy(row_feat).float().unsqueeze(0).to(self.device)
                    cost_batch = torch.from_numpy(C).float().unsqueeze(0).to(self.device)
                    
                mask = torch.ones((1, n), dtype=torch.bool, device=self.device)
                with autocast('cuda', enabled=self.amp_enabled):
                    outputs = self.model(row_tensor, cost=cost_batch, mask=mask)
                u_pred = outputs['u'].squeeze(0).cpu().numpy()[:n]
                
                # GPU-accelerated V computation: Reuse cost_tensor_on_gpu to avoid redundant transfer
                u_pred_tensor = torch.from_numpy(u_pred).float().to(self.device)
                
                if self.use_cuda and cost_tensor_on_gpu is not None:
                    # GPU path: Parallel min reduction using existing tensor
                    v_pred_tensor = torch.min(cost_tensor_on_gpu - u_pred_tensor[:, None], dim=0)[0]
                    v_pred = v_pred_tensor.cpu().numpy()
                else:
                    # CPU fallback path
                    v_pred = np.min(C - u_pred[:, None], axis=0)
            else:
                features = compute_features(C)
                edge_feat = torch.from_numpy(features.edge_feat).float().unsqueeze(0).to(self.device)
                row_feat = torch.from_numpy(features.row_feat).float().unsqueeze(0).to(self.device)
                col_feat = torch.from_numpy(features.col_feat).float().unsqueeze(0).to(self.device)
                mask = torch.ones((1, n), dtype=torch.bool, device=self.device)
                with autocast('cuda', enabled=self.use_cuda):
                    outputs = self.model(edge_feat, row_feat, col_feat, mask=mask)
                u_pred = outputs['u'].cpu().numpy().flatten()[:n]
                v_pred = np.min(C - u_pred[:, None], axis=0)

        return u_pred.astype(np.float64), v_pred.astype(np.float64)


def generate_test_instance(n: int, seed: int) -> np.ndarray:
    """
    Generate a synthetic test instance similar to training distribution.
    
    Uses uniform random costs in [0, 1] range.
    """
    rng = np.random.RandomState(seed)
    C = rng.uniform(0, 1, size=(n, n))
    return C


def benchmark_instance(C: np.ndarray,
                      gnn_predictor: GNNPredictor, 
                      scipy_solver: SciPySolver,
                      lap_solver: LAPSolver,
                      seeded_solver: SeededLAPSolver,
                      compute_baselines: bool = True) -> Dict:
    """Benchmark a single instance with timing."""
    
    n = C.shape[0]
    results = {'n': n, 'success': True}
    
    try:
        # === SciPy Baseline (optional for very large problems) ===
        if compute_baselines:
            print(f"    SciPy baseline...", end=' ', flush=True)
            scipy_timing = time_solver_rigorous(lambda: scipy_solver.solve(C))
            if scipy_timing['success']:
                scipy_time = scipy_timing['median']
                results['scipy_time'] = scipy_time
                print(f"{scipy_time*1000:.2f} ms")
            else:
                print("FAILED")
                results['scipy_time'] = float('nan')
        
            # === Unseeded LAP baseline ===
            print(f"    LAP (unseeded) baseline...", end=' ', flush=True)
            lap_timing = time_solver_rigorous(lambda: lap_solver.solve(C))
            if lap_timing['success']:
                lap_time = lap_timing['median']
                results['lap_time'] = lap_time
                print(f"{lap_time*1000:.2f} ms")
            else:
                print("FAILED")
                results['lap_time'] = float('nan')

        # === GNN + Seeded LAP Pipeline ===  
        print(f"    GNN prediction...", end=' ', flush=True)
        
        # Time GNN inference
        gnn_start = time.time()
        u_pred, v_pred = gnn_predictor.predict(C)
        gnn_time = time.time() - gnn_start
        print(f"{gnn_time*1000:.2f} ms", end=' ')
        
        # Time seeded LAP solve
        print(f"+ Seeded LAP...", end=' ', flush=True)
        seeded_timing = time_solver_rigorous(lambda: seeded_solver.solve(C, u_pred, v_pred))

        if seeded_timing['success']:
            seeded_time = seeded_timing['median'] 
            total_pipeline_time = gnn_time + seeded_time
            results.update({
                'gnn_time': gnn_time,
                'seeded_time': seeded_time, 
                'pipeline_time': total_pipeline_time,
            })
            print(f"{seeded_time*1000:.2f} ms = {total_pipeline_time*1000:.2f} ms total")
        else:
            print("FAILED")
            results['success'] = False
            return results
        
        # === Analysis ===
        if compute_baselines and 'scipy_time' in results and not np.isnan(results['scipy_time']):
            speedup = results['scipy_time'] / total_pipeline_time
            results['speedup'] = speedup
            print(f"    Speedup vs SciPy: {speedup:.2f}x")
        
        if compute_baselines and 'lap_time' in results and not np.isnan(results['lap_time']):
            lap_speedup = results['lap_time'] / total_pipeline_time
            results['lap_speedup'] = lap_speedup
            print(f"    Speedup vs LAP (unseeded): {lap_speedup:.2f}x")
        
    except Exception as e:
        print(f"    ERROR: {e}")
        results['success'] = False
        results['error'] = str(e)
    
    return results


def run_large_scale_benchmark(
    model_files: Sequence[str] | None = None,
    checkpoint_dirs: Sequence[Path] | None = None,
    problem_sizes: Sequence[int] = (4096, 8192, 10240, 16384),
    num_instances: int = 5,
    compute_baselines: bool = True,
):
    """Run the large-scale GNN benchmark suite."""
    
    print("=" * 80)
    print("GNN LARGE-SCALE BENCHMARK")
    print("=" * 80)
    print(f"Thread settings: OMP={os.environ.get('OMP_NUM_THREADS')}, MKL={os.environ.get('MKL_NUM_THREADS')}")
    print(f"Problem sizes: {list(problem_sizes)}")
    print(f"Instances per size: {num_instances}")
    print(f"Compute baselines: {compute_baselines}")
    
    if not torch_available:
        print("❌ PyTorch not available - exiting")
        return
    
    if model_files is None or len(model_files) == 0:
        model_files = [
            "one_gnn_small_full_clean.pt",
            "one_gnn_small_full_clean_h192L4.pt",
            "one_gnn_mid1536_full_clean.pt",
            "one_gnn_mid2048_clean.pt",
            "one_gnn_mid3072_clean.pt",
        ]

    if checkpoint_dirs is None or len(checkpoint_dirs) == 0:
        checkpoint_dirs = [
            project_root / "checkpoints", 
            project_root / "gnn/checkpoints",
            project_root / "gnn/checkpoints_clean"
        ]
    else:
        checkpoint_dirs = [Path(d) if not isinstance(d, Path) else d for d in checkpoint_dirs]
    
    scipy_solver = SciPySolver()
    lap_solver = LAPSolver()
    seeded_solver = SeededLAPSolver()
    
    # Test each model
    all_results = {}
    
    for model_file in model_files:
        model_path: Optional[Path] = None
        candidate = Path(model_file)
        if candidate.is_file():
            model_path = candidate
        else:
            for base_dir in checkpoint_dirs:
                cand = base_dir / model_file
                if cand.exists():
                    model_path = cand
                    break
        if model_path is None or not model_path.exists():
            print(f"⚠️ Model not found in checkpoints: {model_file}")
            continue
        
        print(f"\n{'='*60}")
        print(f"TESTING MODEL: {model_file}")
        print("="*60)
        
        try:
            # Load model
            gnn_predictor = GNNPredictor(str(model_path))
            model_results = []
            
            # Test on each problem size
            for size in problem_sizes:
                print(f"\n--- {size}x{size} Problems ---")
                print(f"Generating {num_instances} synthetic test instances...")
                size_results = []
                
                for i in range(num_instances):
                    print(f"  Instance {i+1}/{num_instances}:")
                    
                    # Generate test instance
                    C = generate_test_instance(size, seed=42 + i)
                    
                    result = benchmark_instance(
                        C,
                        gnn_predictor,
                        scipy_solver,
                        lap_solver,
                        seeded_solver,
                        compute_baselines=compute_baselines,
                    )
                    if result['success']:
                        size_results.append(result)
                        model_results.append(result)
                
                # Size summary
                if size_results:
                    pipeline_times = [r['pipeline_time'] for r in size_results]
                    gnn_times = [r['gnn_time'] for r in size_results]
                    seeded_times = [r['seeded_time'] for r in size_results]
                    
                    print(f"\n  {size}x{size} Summary: {len(size_results)} instances")
                    print(f"    GNN inference: mean {statistics.mean(gnn_times)*1000:.2f} ms, range {min(gnn_times)*1000:.2f}-{max(gnn_times)*1000:.2f} ms")
                    print(f"    Seeded LAP: mean {statistics.mean(seeded_times)*1000:.2f} ms, range {min(seeded_times)*1000:.2f}-{max(seeded_times)*1000:.2f} ms")
                    print(f"    Total pipeline: mean {statistics.mean(pipeline_times)*1000:.2f} ms, range {min(pipeline_times)*1000:.2f}-{max(pipeline_times)*1000:.2f} ms")
                    
                    if compute_baselines:
                        speedups = [r.get('speedup') for r in size_results if 'speedup' in r]
                        lap_speedups = [r.get('lap_speedup') for r in size_results if 'lap_speedup' in r]
                        
                        if speedups:
                            print(f"    Speedup vs SciPy: mean {statistics.mean(speedups):.2f}x, range {min(speedups):.2f}x-{max(speedups):.2f}x")
                        if lap_speedups:
                            print(f"    Speedup vs LAP (unseeded): mean {statistics.mean(lap_speedups):.2f}x, range {min(lap_speedups):.2f}x-{max(lap_speedups):.2f}x")
            
            all_results[model_file] = model_results
            
        except Exception as e:
            print(f"❌ Failed to test model {model_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # === COMPREHENSIVE ANALYSIS ===
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS") 
    print("="*80)
    
    if not all_results:
        print("❌ No successful results!")
        return
    
    # Summary table
    header = f"\n{'Model':<30} {'Size':<10} {'Instances':<10} {'GNN(ms)':<12} {'Seeded(ms)':<12} {'Total(ms)':<12}"
    if compute_baselines:
        header += f" {'SciPySpeedup':<14} {'LAPSpeedup':<12}"
    print(header)
    print("-" * len(header))
    
    for model_name, results in all_results.items():
        if not results:
            continue
        
        # Group by size
        by_size = {}
        for r in results:
            size = r['n']
            if size not in by_size:
                by_size[size] = []
            by_size[size].append(r)
        
        for size, size_results in sorted(by_size.items()):
            gnn_times = [r['gnn_time']*1000 for r in size_results]
            seeded_times = [r['seeded_time']*1000 for r in size_results]
            total_times = [r['pipeline_time']*1000 for r in size_results]
            
            line = f"{model_name[:29]:<30} {size:<10} {len(size_results):<10} "
            line += f"{statistics.mean(gnn_times):<12.1f} {statistics.mean(seeded_times):<12.1f} {statistics.mean(total_times):<12.1f}"
            
            if compute_baselines:
                speedups = [r.get('speedup') for r in size_results if 'speedup' in r]
                lap_speedups = [r.get('lap_speedup') for r in size_results if 'lap_speedup' in r]
                
                scipy_str = f"{statistics.mean(speedups):.2f}x" if speedups else "N/A"
                lap_str = f"{statistics.mean(lap_speedups):.2f}x" if lap_speedups else "N/A"
                line += f" {scipy_str:<14} {lap_str:<12}"
            
            print(line)
    
    print(f"\n🎯 SCALING ANALYSIS:")
    print("   For each model, observe how GNN inference and Seeded LAP times grow with problem size.")
    print("   Ideally: GNN time grows ~O(n) or O(n log n), Seeded LAP benefits from good initialization.")
    
    # Calculate scaling factors
    for model_name, results in all_results.items():
        by_size = {}
        for r in results:
            size = r['n']
            if size not in by_size:
                by_size[size] = []
            by_size[size].append(r)
        
        if len(by_size) < 2:
            continue
        
        sorted_sizes = sorted(by_size.keys())
        print(f"\n   Model: {model_name}")
        for i in range(1, len(sorted_sizes)):
            prev_size = sorted_sizes[i-1]
            curr_size = sorted_sizes[i]
            
            prev_time = statistics.mean([r['pipeline_time'] for r in by_size[prev_size]])
            curr_time = statistics.mean([r['pipeline_time'] for r in by_size[curr_size]])
            
            size_ratio = curr_size / prev_size
            time_ratio = curr_time / prev_time
            
            print(f"     {prev_size} → {curr_size}: size ratio {size_ratio:.2f}x, time ratio {time_ratio:.2f}x")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Large-scale benchmark for GNN models on 4096-16384 problems")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model filenames or paths.",
    )
    parser.add_argument(
        "--checkpoint-dirs",
        nargs="+",
        default=[
            str(project_root / "checkpoints"), 
            str(project_root / "gnn/checkpoints"),
            str(project_root / "gnn/checkpoints_clean")
        ],
        help="Directories to search for model checkpoints.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[4096, 8192, 10240, 16384],
        help="Problem sizes (n) to benchmark.",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=5,
        help="Number of instances per size.",
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip SciPy/LAP baselines (faster, only measures GNN+Seeded pipeline).",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_large_scale_benchmark(
        model_files=args.models,
        checkpoint_dirs=[Path(d) for d in args.checkpoint_dirs],
        problem_sizes=args.sizes,
        num_instances=args.instances,
        compute_baselines=not args.no_baselines,
    )
