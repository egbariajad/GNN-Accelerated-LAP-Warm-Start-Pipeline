#!/usr/bin/env python3
"""
GNN Benchmark By Problem Type

Tests GNN models on different problem types to identify strengths and weaknesses:
- 'block', 'clustered', 'low_rank', 'metric', 'noisy_linear', 'sparse', 'tie', 'uniform'

Analyzes performance breakdown by category.
"""

import os
import sys
import argparse
from pathlib import Path
import time
import statistics
from typing import Dict, List, Tuple, Optional, Sequence
from collections import defaultdict, Counter
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
                # DualGNN: uses graph features  
                features = compute_features(C)
                edge_feat = torch.from_numpy(features.edge_feat).float().unsqueeze(0).to(self.device)
                row_feat = torch.from_numpy(features.row_feat).float().unsqueeze(0).to(self.device)
                col_feat = torch.from_numpy(features.col_feat).float().unsqueeze(0).to(self.device)
                mask = torch.ones((1, n), dtype=torch.bool, device=self.device)
                with autocast('cuda', enabled=self.use_cuda):
                    outputs = self.model(edge_feat, row_feat, col_feat, mask=mask)
                u_pred = outputs['u'].cpu().numpy().flatten()[:n]
                # For DualGNN, also compute v using dual feasibility
                v_pred = np.min(C - u_pred[:, None], axis=0)

        return u_pred.astype(np.float64), v_pred.astype(np.float64)


class DatasetLoaderByType:
    """Load test instances from HDF5 datasets, grouped by problem type."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
    
    def load_instances_by_type(self, problem_sizes: List[int], instances_per_type: int = 5, 
                              filter_types: List[str] | None = None) -> Dict:
        """
        Load test instances for specified problem sizes, grouped by type.
        
        Args:
            problem_sizes: List of problem sizes to load
            instances_per_type: Number of instances to load per type
            filter_types: Optional list of problem types to load (e.g., ['block', 'tie'])
        
        Returns:
            Dict mapping size -> type -> list of (C, u_true, v_true, index) tuples
        """
        instances = {}
        
        for size in problem_sizes:
            instances[size] = defaultdict(list)
            
            # Find appropriate dataset file
            test_file = None
            if size <= 512:
                test_file = self.data_dir / "generated/processed/small/full/test.h5"
            elif size <= 1024:
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
            
            # Load instances, grouping by type
            with h5py.File(test_file, 'r') as f:
                n_total = len(f['n'])
                
                # Check if 'family' field exists (problem type)
                if 'family' not in f:
                    print(f"  ⚠️ No 'family' field in dataset - cannot group by type")
                    continue
                
                # Count instances by type for this size
                type_counts = defaultdict(int)
                
                # First pass: find all instances of target size and count by type
                for i in range(n_total):
                    if f['n'][i] == size:
                        prob_type = f['family'][i].decode('utf-8') if isinstance(f['family'][i], bytes) else f['family'][i]
                        type_counts[prob_type] += 1
                
                print(f"  Available types for {size}x{size}: {dict(type_counts)}")
                
                # Second pass: load instances_per_type of each type
                type_loaded = defaultdict(int)
                for i in range(n_total):
                    if f['n'][i] == size:
                        prob_type = f['family'][i].decode('utf-8') if isinstance(f['family'][i], bytes) else f['family'][i]
                        
                        # Skip if filtering and this type is not in the filter
                        if filter_types and prob_type not in filter_types:
                            continue
                        
                        # Load up to instances_per_type of each type
                        if type_loaded[prob_type] < instances_per_type:
                            C_flat = f['C'][i]
                            u_true = f['u'][i] 
                            v_true = f['v'][i]
                            n = f['n'][i]
                            
                            # Reshape cost matrix
                            C = C_flat.reshape(n, n)
                            
                            instances[size][prob_type].append((C, u_true, v_true, i))
                            type_loaded[prob_type] += 1
                
                print(f"  Loaded: {dict(type_loaded)}")
                total_loaded = sum(type_loaded.values())
                print(f"  Total instances: {total_loaded}")
        
        return instances


def benchmark_instance(C: np.ndarray, u_true: np.ndarray, v_true: np.ndarray, 
                      gnn_predictor: GNNPredictor, 
                      scipy_solver: SciPySolver,
                      lap_solver: LAPSolver,
                      seeded_solver: SeededLAPSolver,
                      verbose: bool = True) -> Dict:
    """Benchmark a single instance with timing."""
    
    n = C.shape[0]
    results = {'n': n, 'success': True}
    
    try:
        # === SciPy Baseline ===
        if verbose:
            print(f"    SciPy...", end=' ')
        scipy_timing = time_solver_rigorous(lambda: scipy_solver.solve(C))
        if scipy_timing['success']:
            scipy_time = scipy_timing['median']
            results['scipy_time'] = scipy_time
            if verbose:
                print(f"{scipy_time*1000:.2f}ms", end=' ')
        else:
            if verbose:
                print("FAIL")
            results['success'] = False
            return results
        
        # === Unseeded LAP baseline ===
        if verbose:
            print(f"LAP...", end=' ')
        lap_timing = time_solver_rigorous(lambda: lap_solver.solve(C))
        if lap_timing['success']:
            lap_time = lap_timing['median']
            results['lap_time'] = lap_time
            if verbose:
                print(f"{lap_time*1000:.2f}ms", end=' ')
        else:
            if verbose:
                print("FAIL")
            results['success'] = False
            return results

        # === GNN + Seeded LAP Pipeline ===  
        if verbose:
            print(f"GNN...", end=' ')
        
        # Time GNN inference
        gnn_start = time.time()
        u_pred, v_pred = gnn_predictor.predict(C)
        gnn_time = time.time() - gnn_start
        if verbose:
            print(f"{gnn_time*1000:.2f}ms", end=' ')
        
        # Time seeded LAP solve
        if verbose:
            print(f"+Seeded...", end=' ')
        seeded_timing = time_solver_rigorous(lambda: seeded_solver.solve(C, u_pred, v_pred))

        if seeded_timing['success']:
            seeded_time = seeded_timing['median'] 
            total_pipeline_time = gnn_time + seeded_time
            results.update({
                'gnn_time': gnn_time,
                'seeded_time': seeded_time, 
                'pipeline_time': total_pipeline_time,
            })
            if verbose:
                print(f"{seeded_time*1000:.2f}ms", end=' ')
        else:
            if verbose:
                print("FAIL")
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
        
        if verbose:
            print(f"→ SciPy:{speedup:.2f}x LAP:{lap_speedup:.2f}x MAE:{u_error:.4f}")
        
    except Exception as e:
        if verbose:
            print(f"ERROR: {e}")
        results['success'] = False
        results['error'] = str(e)
    
    return results


def run_benchmark_by_type(
    model_files: Sequence[str] | None = None,
    checkpoint_dirs: Sequence[Path] | None = None,
    problem_sizes: Sequence[int] = (512, 1024),
    instances_per_type: int = 5,
    problem_types: Sequence[str] | None = None,
):
    """Run the GNN benchmark grouped by problem type."""
    
    print("=" * 80)
    print("GNN BENCHMARK BY PROBLEM TYPE")
    print("=" * 80)
    print(f"Thread settings: OMP={os.environ.get('OMP_NUM_THREADS')}, MKL={os.environ.get('MKL_NUM_THREADS')}")
    print(f"Testing {instances_per_type} instances per problem type")
    if problem_types:
        print(f"Filtering to types: {', '.join(problem_types)}")
    
    if not torch_available:
        print("❌ PyTorch not available - exiting")
        return
    
    # Configuration
    if model_files is None or len(model_files) == 0:
        model_files = ["progressive_clean_best.pt", "one_gnn_mid1536_full_clean_h192L4.pt"]

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
    
    # Load test data by type
    print(f"\n{'='*60}")
    print("LOADING TEST DATA BY TYPE")
    print("="*60)
    
    dataset_loader = DatasetLoaderByType(data_dir)
    instances = dataset_loader.load_instances_by_type(problem_sizes, instances_per_type, 
                                                      filter_types=problem_types)
    
    total_instances = sum(
        sum(len(inst_list) for inst_list in type_dict.values())
        for type_dict in instances.values()
    )
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
            print(f"⚠️ Model not found: {model_file}")
            continue
        
        print(f"\n{'='*60}")
        print(f"TESTING MODEL: {model_file}")
        print("="*60)
        
        try:
            # Load model
            gnn_predictor = GNNPredictor(str(model_path))
            
            # Results grouped by type
            results_by_type = defaultdict(list)
            
            # Test on each problem size and type
            for size in problem_sizes:
                if size not in instances or not instances[size]:
                    continue
                
                print(f"\n--- {size}x{size} Problems ---")
                
                for prob_type in sorted(instances[size].keys()):
                    type_instances = instances[size][prob_type]
                    print(f"\n  Type: {prob_type} ({len(type_instances)} instances)")
                    
                    for idx, (C, u_true, v_true, data_idx) in enumerate(type_instances):
                        print(f"    [{idx+1}/{len(type_instances)}]", end=' ')
                        
                        result = benchmark_instance(
                            C, u_true, v_true,
                            gnn_predictor,
                            scipy_solver,
                            lap_solver,
                            seeded_solver,
                            verbose=True,
                        )
                        
                        if result['success']:
                            result['type'] = prob_type
                            result['size'] = size
                            result['data_index'] = data_idx
                            results_by_type[prob_type].append(result)
                    
                    # Type summary for this size
                    type_results = [r for r in results_by_type[prob_type] if r['size'] == size]
                    if type_results:
                        speedups = [r['speedup'] for r in type_results]
                        lap_speedups = [r['lap_speedup'] for r in type_results]
                        print(f"    Summary: SciPy {statistics.mean(speedups):.2f}x [{min(speedups):.2f}-{max(speedups):.2f}], "
                              f"LAP {statistics.mean(lap_speedups):.2f}x [{min(lap_speedups):.2f}-{max(lap_speedups):.2f}]")
            
            all_results[model_file] = results_by_type
            
        except Exception as e:
            print(f"❌ Failed to test model {model_file}: {e}")
    
    # === COMPREHENSIVE ANALYSIS BY TYPE ===
    print(f"\n{'='*80}")
    print("PERFORMANCE BREAKDOWN BY PROBLEM TYPE") 
    print("="*80)
    
    if not all_results:
        print("❌ No successful results!")
        return
    
    for model_name, results_by_type in all_results.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print("="*60)
        
        # Summary table by type
        print(f"\n{'Type':<15} {'Count':<8} {'SciPy':<12} {'LAP':<12} {'MAE':<12}")
        print("-" * 60)
        
        type_performance = []
        
        for prob_type in sorted(results_by_type.keys()):
            results = results_by_type[prob_type]
            if not results:
                continue
            
            speedups = [r['speedup'] for r in results if r['success']]
            lap_speedups = [r['lap_speedup'] for r in results if r['success']]
            maes = [r['u_mae'] for r in results if r['success']]
            
            if not speedups:
                continue
            
            mean_speedup = statistics.mean(speedups)
            lap_mean = statistics.mean(lap_speedups) if lap_speedups else float('nan')
            mean_mae = statistics.mean(maes)
            
            type_performance.append({
                'type': prob_type,
                'speedup': mean_speedup,
                'lap_speedup': lap_mean,
                'mae': mean_mae,
                'count': len(results)
            })
            
            print(f"{prob_type:<15} {len(results):<8} "
                  f"{mean_speedup:>5.2f}x [{min(speedups):>4.2f}-{max(speedups):>4.2f}] "
                  f"{lap_mean:>5.2f}x [{min(lap_speedups):>4.2f}-{max(lap_speedups):>4.2f}] "
                  f"{mean_mae:>8.4f}")
        
        # Rank by performance
        if type_performance:
            print(f"\n{'='*60}")
            print("STRENGTHS AND WEAKNESSES")
            print("="*60)
            
            # Sort by speedup vs SciPy
            by_scipy = sorted(type_performance, key=lambda x: x['speedup'], reverse=True)
            print(f"\n🏆 Best Performance (vs SciPy):")
            for i, perf in enumerate(by_scipy[:3], 1):
                print(f"  {i}. {perf['type']:<15} {perf['speedup']:.2f}x speedup")
            
            print(f"\n⚠️  Worst Performance (vs SciPy):")
            for i, perf in enumerate(reversed(by_scipy[-3:]), 1):
                print(f"  {i}. {perf['type']:<15} {perf['speedup']:.2f}x speedup")
            
            # Sort by speedup vs LAP
            by_lap = sorted(type_performance, key=lambda x: x['lap_speedup'], reverse=True)
            print(f"\n🚀 Best Performance (vs LAP):")
            for i, perf in enumerate(by_lap[:3], 1):
                print(f"  {i}. {perf['type']:<15} {perf['lap_speedup']:.2f}x speedup")
            
            print(f"\n⛔ Worst Performance (vs LAP):")
            for i, perf in enumerate(reversed(by_lap[-3:]), 1):
                print(f"  {i}. {perf['type']:<15} {perf['lap_speedup']:.2f}x speedup")
            
            # Sort by accuracy
            by_mae = sorted(type_performance, key=lambda x: x['mae'])
            print(f"\n🎯 Most Accurate Predictions:")
            for i, perf in enumerate(by_mae[:3], 1):
                print(f"  {i}. {perf['type']:<15} MAE={perf['mae']:.4f}")
            
            print(f"\n📊 Least Accurate Predictions:")
            for i, perf in enumerate(reversed(by_mae[-3:]), 1):
                print(f"  {i}. {perf['type']:<15} MAE={perf['mae']:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark GNN models by problem type")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["progressive_clean_best.pt", "one_gnn_mid1536_full_clean_h192L4.pt"],
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
        default=[512, 1024],
        help="Problem sizes (n) to benchmark.",
    )
    parser.add_argument(
        "--instances-per-type",
        type=int,
        default=5,
        help="Number of instances to test per problem type.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=None,
        help="Problem types to test (e.g., block tie sparse). If not specified, tests all available types.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_benchmark_by_type(
        model_files=args.models,
        checkpoint_dirs=[Path(d) for d in args.checkpoint_dirs],
        problem_sizes=args.sizes,
        instances_per_type=args.instances_per_type,
        problem_types=args.types,
    )
