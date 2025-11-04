#!/usr/bin/env python3
"""
GNN + Seeded LAP Benchmark

Comprehensive benchmark comparing:
1. GNN inference + Seeded LAP (end-to-end pipeline)  
2. SciPy baseline (linear_sum_assignment)

Tests all 4 trained models on different problem sizes from the dataset.
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
import h5py

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
            # Determine feature type based on input dimension if not explicitly set
            if 'features' in checkpoint:
                features = checkpoint['features']
            elif row_feat_dim == 17:
                features = 'reduced'
            elif row_feat_dim == 21:
                features = 'full'
            else:
                features = 'full'  # Default for unknown dimensions
            
            self.model_info = {
                'architecture': architecture,
                'hidden_dim': checkpoint.get('hidden_dim', 128),
                'layers': checkpoint.get('layers', 4),
                'dropout': checkpoint.get('dropout', 0.1),
                'heads': checkpoint.get('heads', 4),
                'row_feat_dim': row_feat_dim,
                'features': features,
            }

            # Some checkpoints (e.g., progressive training) store metadata under
            # a nested "config" dictionary rather than top-level keys. Merge
            # those values if present.
            nested_cfg = checkpoint.get('config')
            if isinstance(nested_cfg, dict):
                self.model_info['architecture'] = nested_cfg.get('architecture', self.model_info['architecture'])
                self.model_info['hidden_dim'] = nested_cfg.get('hidden_dim', self.model_info['hidden_dim'])
                self.model_info['layers'] = nested_cfg.get('layers', self.model_info['layers'])
                self.model_info['dropout'] = nested_cfg.get('dropout', self.model_info['dropout'])
                self.model_info['heads'] = nested_cfg.get('heads', self.model_info['heads'])
                self.model_info['row_feat_dim'] = nested_cfg.get('row_feat_dim', self.model_info['row_feat_dim'])
                # Treat progressive OneGNN checkpoints as using full features by default
                if nested_cfg.get('row_feat_dim') == 21:
                    self.model_info['features'] = 'full'
        else:
            # Direct state dict - treat as legacy DualGNN checkpoint
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
            if in_dim != compute_row_features(np.zeros((1, 1))).shape[1]:
                raise ValueError(
                    f"Checkpoint expects {in_dim} row features but the pipeline now uses "
                    f"{compute_row_features(np.zeros((1, 1))).shape[1]}."
                )
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

        # AMP + sparse top-k refinement can generate NaNs for row-only models on GPU.
        # Keep FP32 math for that configuration while still running on the GPU.
        self.amp_enabled = self.use_cuda and not self.row_only

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        print(f"  Config: {self.model_info}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # GPU warmup to avoid first-inference overhead
        if 'cuda' in self.device:
            self._gpu_warmup()
    
    def _gpu_warmup(self):
        """Perform GPU warmup to avoid first-inference overhead."""
        print("  GPU warmup...", end=' ')
        
        # Create a small dummy problem for warmup
        dummy_size = 64
        dummy_C = np.random.rand(dummy_size, dummy_size).astype(np.float64)
        
        # Run a few warmup inferences
        for _ in range(3):
            with torch.inference_mode():
                if self.row_only:
                    # OneGNN warmup - use correct feature function
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
                    # DualGNN warmup
                    features = compute_features(dummy_C)
                    edge_feat = torch.from_numpy(features.edge_feat).float().unsqueeze(0).to(self.device)
                    row_feat = torch.from_numpy(features.row_feat).float().unsqueeze(0).to(self.device)
                    col_feat = torch.from_numpy(features.col_feat).float().unsqueeze(0).to(self.device)
                    mask = torch.ones((1, dummy_size), dtype=torch.bool, device=self.device)
                    with autocast('cuda', enabled=self.use_cuda):
                        _ = self.model(edge_feat, row_feat, col_feat, mask=mask)
        
        # Synchronize GPU to ensure warmup completes
        if 'cuda' in self.device:
            torch.cuda.synchronize()
        
        print("done")
    
    def predict(self, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict dual potentials for a cost matrix.
        
        Args:
            C: Cost matrix (n x n)
            
        Returns:
            u, v: Predicted dual potentials
        """
        if not torch_available or self.model is None:
            raise RuntimeError("Model not loaded")
        
        C = np.asarray(C, dtype=np.float64)
        n = C.shape[0]
        
        cost_tensor_on_gpu = None  # Track if we have C on GPU for V computation reuse

        with torch.inference_mode():
            if self.row_only:
                # Use CUDA-optimized feature computation for OneGNN
                if 'cuda' in self.device:
                    cost_tensor = torch.from_numpy(C).float().to(self.device)
                    cost_tensor_on_gpu = cost_tensor  # Save reference for V computation
                    row_feat_tensor = compute_row_features_torch(cost_tensor)
                    row_tensor = row_feat_tensor.unsqueeze(0)  # [1, n, feat_dim]
                    cost_batch = cost_tensor.unsqueeze(0)
                else:
                    # Fallback to CPU version
                    row_feat = compute_row_features(C)
                    row_tensor = torch.from_numpy(row_feat).float().unsqueeze(0).to(self.device)
                    cost_batch = torch.from_numpy(C).float().unsqueeze(0).to(self.device)
                    
                mask = torch.ones((1, n), dtype=torch.bool, device=self.device)
                with autocast('cuda', enabled=self.amp_enabled):
                    outputs = self.model(row_tensor, cost=cost_batch, mask=mask)
                
                # GPU-accelerated V computation: Reuse cost_tensor_on_gpu to avoid redundant transfer
                u_pred_tensor = outputs['u'].squeeze(0)[:n]  # Keep on GPU
                
                if self.use_cuda and cost_tensor_on_gpu is not None:
                    # GPU path: Use PyTorch for parallel min reduction with tensor reuse
                    v_pred_tensor = torch.min(cost_tensor_on_gpu - u_pred_tensor[:, None], dim=0)[0]
                    torch.cuda.synchronize()
                    v_pred = v_pred_tensor.cpu().numpy()
                    u_pred = u_pred_tensor.cpu().numpy()
                else:
                    # CPU path: Use NumPy as before
                    u_pred = u_pred_tensor.cpu().numpy()
                    v_pred = np.min(C - u_pred[:, None], axis=0)
                    
            else:
                # DualGNN: uses graph features  
                features = compute_features(C)
                edge_feat = torch.from_numpy(features.edge_feat).float().unsqueeze(0).to(self.device)
                row_feat = torch.from_numpy(features.row_feat).float().unsqueeze(0).to(self.device)
                col_feat = torch.from_numpy(features.col_feat).float().unsqueeze(0).to(self.device)
                mask = torch.ones((1, n), dtype=torch.bool, device=self.device)
                with autocast('cuda', enabled=self.use_cuda):
                    outputs = self.model(edge_feat, row_feat, col_feat, mask=mask)
                
                # GPU-accelerated V computation for DualGNN
                u_pred_tensor = outputs['u'].squeeze(0)[:n]  # Keep on GPU
                
                if self.use_cuda:
                    # Transfer C to GPU for V computation (DualGNN doesn't have C on GPU yet)
                    C_tensor = torch.from_numpy(C).float().to(self.device)
                    v_pred_tensor = torch.min(C_tensor - u_pred_tensor[:, None], dim=0)[0]
                    torch.cuda.synchronize()
                    v_pred = v_pred_tensor.cpu().numpy()
                    u_pred = u_pred_tensor.cpu().numpy().flatten()
                else:
                    # CPU path: Use NumPy
                    u_pred = u_pred_tensor.cpu().numpy().flatten()
                    v_pred = np.min(C - u_pred[:, None], axis=0)

        return u_pred.astype(np.float64), v_pred.astype(np.float64)


# removed sparse-pruned helpers


class DatasetLoader:
    """Load test instances from HDF5 datasets."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
    
    def load_instances(self, problem_sizes: List[int], max_instances_per_size: int = 50) -> Dict:
        """
        Load test instances for specified problem sizes.
        
        Returns:
            Dict mapping size -> list of (C, u_true, v_true) tuples
        """
        instances = {}
        
        for size in problem_sizes:
            instances[size] = []
            
            # Find appropriate dataset file
            test_file = None
            if size <= 512:
                # Use small dataset
                test_file = self.data_dir / "generated/processed/small/full/test.h5"
            elif size <= 1024:
                # Use small dataset (contains both 512 and 1024)
                test_file = self.data_dir / "generated/processed/small/full/test.h5"  
            elif size <= 1536:
                test_file = self.data_dir / "generated/processed/mid_1536/full/test.h5"
            elif size <= 2048:
                test_file = self.data_dir / "generated/processed/mid_2048/full/test.h5"
            elif size <= 3072:
                test_file = self.data_dir / "generated/processed/mid_3072/full/test.h5"
            elif size <= 4096:
                test_file = self.data_dir / "generated/processed/large_4096/full/test.h5"
            else:
                print(f"⚠️ No dataset found for size {size}")
                continue
                
            if not test_file.exists():
                print(f"⚠️ Test file not found: {test_file}")
                continue
            
            print(f"Loading {size}x{size} instances from: {test_file.name}")
            
            # Load instances of the target size
            with h5py.File(test_file, 'r') as f:
                n_total = len(f['n'])
                size_indices = []
                
                # Find instances with target size
                for i in range(n_total):
                    if f['n'][i] == size:
                        size_indices.append(i)
                        if len(size_indices) >= max_instances_per_size:
                            break
                
                print(f"  Found {len(size_indices)} instances of size {size}x{size}")
                
                # Load the instances
                for i in size_indices:
                    C_flat = f['C'][i]
                    u_true = f['u'][i] 
                    v_true = f['v'][i]
                    n = f['n'][i]
                    
                    # Reshape cost matrix
                    C = C_flat.reshape(n, n)
                    
                    instances[size].append((C, u_true, v_true))
        
        return instances


def benchmark_instance(C: np.ndarray, u_true: np.ndarray, v_true: np.ndarray, 
                      gnn_predictor: GNNPredictor, 
                      scipy_solver: SciPySolver,
                      lap_solver: LAPSolver,
                      seeded_solver: SeededLAPSolver) -> Dict:
    """Benchmark a single instance with timing."""
    
    n = C.shape[0]
    results = {'n': n, 'success': True}
    
    try:
        # === SciPy Baseline ===
        print(f"    SciPy baseline...", end=' ')
        scipy_timing = time_solver_rigorous(lambda: scipy_solver.solve(C))
        if scipy_timing['success']:
            scipy_time = scipy_timing['median']
            results['scipy_time'] = scipy_time
            print(f"{scipy_time*1000:.2f} ms")
        else:
            print("FAILED")
            results['success'] = False
            return results
        
        # === Unseeded LAP baseline ===
        print(f"    LAP (unseeded) baseline...", end=' ')
        lap_timing = time_solver_rigorous(lambda: lap_solver.solve(C))
        if lap_timing['success']:
            lap_time = lap_timing['median']
            results['lap_time'] = lap_time
            print(f"{lap_time*1000:.2f} ms")
        else:
            print("FAILED")
            results['success'] = False
            results['lap_error'] = lap_timing.get('error')
            return results

        # === GNN + Seeded LAP Pipeline ===  
        print(f"    GNN prediction...", end=' ')
        
        # Time GNN inference
        gnn_start = time.time()
        u_pred, v_pred = gnn_predictor.predict(C)
        gnn_time = time.time() - gnn_start
        print(f"{gnn_time*1000:.2f} ms", end=' ')
        
        # Time seeded LAP solve
        print(f"+ Seeded LAP...", end=' ')
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
        speedup = scipy_time / total_pipeline_time
        results['speedup'] = speedup
        lap_speedup = results['lap_time'] / total_pipeline_time if total_pipeline_time > 0 else float('nan')
        results['lap_speedup'] = lap_speedup
        
        # Quality analysis
        u_error = np.mean(np.abs(u_pred - u_true))
        v_error = np.mean(np.abs(v_pred - v_true))
        results.update({
            'u_mae': u_error,
            'v_mae': v_error,
        })
        
        print(f"    Speedup vs SciPy: {speedup:.2f}x")
        print(f"    Speedup vs LAP (unseeded): {lap_speedup:.2f}x, Dual MAE: u={u_error:.4f}, v={v_error:.4f}")
        
    except Exception as e:
        print(f"    ERROR: {e}")
        results['success'] = False
        results['error'] = str(e)
    
    return results


def run_gnn_benchmark(
    model_files: Sequence[str] | None = None,
    checkpoint_dirs: Sequence[Path] | None = None,
    problem_sizes: Sequence[int] = (512, 1024, 2048),
    max_instances: int = 5,
):
    """Run the complete GNN benchmark suite."""
    
    print("=" * 80)
    print("GNN + SEEDED LAP vs SCIPY BENCHMARK")
    print("=" * 80)
    print(f"Thread settings: OMP={os.environ.get('OMP_NUM_THREADS')}, MKL={os.environ.get('MKL_NUM_THREADS')}")
    print("Pipeline: SciPy vs unseeded LAP vs GNN-seeded LAP")
    
    if not torch_available:
        print("❌ PyTorch not available - exiting")
        return
    
    # Configuration
    if model_files is None or len(model_files) == 0:
        model_files = [
            # DualGNN models (in checkpoints/)
            "debug_model_proper.pt",    # Latest DualGNN Sep 24
            "debug_model.pt",          # Latest DualGNN Sep 24 
            "small_bucket_model.pt",   # Older DualGNN
            "validation_model.pt",     # Older DualGNN
            # DualGNN models from progressive training (in gnn/checkpoints/)
            "small_512.pt",            # DualGNN Progressive Sep 26
            "small_1024.pt",           # DualGNN Progressive Sep 26
            # OneGNN models (in gnn/checkpoints/)
            "one_gnn_debug.pt",        # OneGNN Sep 26
            "one_gnn_small_full.pt",   # OneGNN Sep 26
        ]

    if checkpoint_dirs is None or len(checkpoint_dirs) == 0:
        checkpoint_dirs = [
            project_root / "checkpoints", 
            project_root / "gnn/checkpoints",
            project_root / "gnn/checkpoints_clean"
        ]
    else:
        checkpoint_dirs = [Path(d) if not isinstance(d, Path) else d for d in checkpoint_dirs]

    data_dir = project_root / "data"
    
    scipy_solver = SciPySolver()
    lap_solver = LAPSolver()
    seeded_solver = SeededLAPSolver()
    
    # Load test data
    print(f"\n{'='*60}")
    print("LOADING TEST DATA")
    print("="*60)
    
    dataset_loader = DatasetLoader(data_dir)
    instances = dataset_loader.load_instances(problem_sizes, max_instances)
    
    total_instances = sum(len(inst_list) for inst_list in instances.values())
    print(f"Total test instances loaded: {total_instances}")
    
    if total_instances == 0:
        print("❌ No test instances loaded - exiting")
        return
    
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
                if size not in instances or not instances[size]:
                    continue
                
                print(f"\n--- {size}x{size} Problems ---")
                size_results = []
                
                for i, (C, u_true, v_true) in enumerate(instances[size]):
                    print(f"  Instance {i+1}/{len(instances[size])}:")
                    
                    result = benchmark_instance(
                        C, u_true, v_true,
                        gnn_predictor,
                        scipy_solver,
                        lap_solver,
                        seeded_solver,
                    )
                    if result['success']:
                        size_results.append(result)
                        model_results.append(result)
                
                # Size summary
                if size_results:
                    speedups = [r['speedup'] for r in size_results]
                    lap_speedups = [r['lap_speedup'] for r in size_results]
                    avg_speedup = statistics.mean(speedups)
                    lap_avg_speedup = statistics.mean(lap_speedups)
                    
                    print(f"  {size}x{size} Summary: {len(size_results)} instances")
                    print(f"    GNN prediction... + Seeded LAP...")
                    print(f"      Speedup vs SciPy: mean {avg_speedup:.2f}x, range {min(speedups):.2f}x - {max(speedups):.2f}x")
                    print(f"      Speedup vs LAP (unseeded): mean {lap_avg_speedup:.2f}x, range {min(lap_speedups):.2f}x - {max(lap_speedups):.2f}x")
            
            all_results[model_file] = model_results
            
        except Exception as e:
            print(f"❌ Failed to test model {model_file}: {e}")
    
    # === COMPREHENSIVE ANALYSIS ===
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS") 
    print("="*80)
    
    if not all_results:
        print("❌ No successful results!")
        return
    
    # Summary table
    header = f"\n{'Model':<25} {'Instances':<10} {'SciPyMean':<12} {'SciPyMin':<12} {'SciPyMax':<12} {'VsLAPMean':<12} {'Success':<8}"
    print(header)
    print("-" * (len(header) - 1))
    
    best_model = None
    best_speedup = 0
    
    for model_name, results in all_results.items():
        if not results:
            continue
        
        speedups = [r['speedup'] for r in results if r['success']]
        lap_speedups = [r['lap_speedup'] for r in results if r['success']]
        if not speedups:
            continue
        
        mean_speedup = statistics.mean(speedups)
        min_speedup = min(speedups)
        max_speedup = max(speedups)
        lap_mean = statistics.mean(lap_speedups) if lap_speedups else float('nan')
        success_rate = len(speedups) / len(results) * 100
        
        print(f"{model_name[:24]:<25} {len(results):<10} {mean_speedup:<12.2f} {min_speedup:<12.2f} {max_speedup:<12.2f} {lap_mean:<12.2f} {success_rate:<7.0f}%")
        
        if mean_speedup > best_speedup:
            best_speedup = mean_speedup
            best_model = model_name
    
    # Overall statistics
    all_speedups = []
    all_lap_speedups = []
    for results in all_results.values():
        all_speedups.extend([r['speedup'] for r in results if r['success']])
        all_lap_speedups.extend([r['lap_speedup'] for r in results if r['success']])
    
    if all_speedups:
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Total successful tests: {len(all_speedups)}")
        print(f"   Full pipeline vs SciPy: mean {statistics.mean(all_speedups):.2f}x, range {min(all_speedups):.2f}x - {max(all_speedups):.2f}x")
        if all_lap_speedups:
            print(f"   Full pipeline vs LAP (unseeded): mean {statistics.mean(all_lap_speedups):.2f}x, range {min(all_lap_speedups):.2f}x - {max(all_lap_speedups):.2f}x")
        
        excellent_count = sum(1 for s in all_speedups if s >= 2.0)
        good_count = sum(1 for s in all_speedups if s >= 1.0) 
        
        print(f"   Excellent cases (≥2.0x): {excellent_count}/{len(all_speedups)} ({excellent_count/len(all_speedups)*100:.1f}%)")
        print(f"   Speedup cases (≥1.0x): {good_count}/{len(all_speedups)} ({good_count/len(all_speedups)*100:.1f}%)")
    
    if best_model:
        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"   Best mean speedup vs SciPy: {best_speedup:.2f}x")
    
    print(f"\n🎯 CONCLUSION:")
    if all_speedups and statistics.mean(all_speedups) > 1.0:
        print("   ✅ GNN + Seeded LAP pipeline shows significant promise!")
        print("   📈 End-to-end pipeline outperforms SciPy baseline")
    else:
        print("   ⚠️ Pipeline needs optimization - GNN inference overhead?")
    
    print("   💡 Consider: Model optimization, inference acceleration, batch processing")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark DualGNN/OneGNN checkpoints against SciPy/LAP")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model filenames or paths. Defaults to the legacy DualGNN checkpoints.",
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
        default=[512, 1024, 2048],
        help="Problem sizes (n) to benchmark.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=5,
        help="Maximum number of instances per size.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_gnn_benchmark(
        model_files=args.models,
        checkpoint_dirs=[Path(d) for d in args.checkpoint_dirs],
        problem_sizes=args.sizes,
        max_instances=args.max_instances,
    )
