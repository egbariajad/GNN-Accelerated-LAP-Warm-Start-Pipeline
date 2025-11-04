#!/usr/bin/env python3
"""
Comprehensive Pipeline Analysis - All Problem Types

Analyzes pipeline performance across all problem types:
- UNIFORM
- SPARSE  
- BLOCK
- TIE
- METRIC
- LOW_RANK

For each type, breaks down timing of every stage to identify:
1. Which types benefit from GNN acceleration
2. Which types should use fallback (unseeded LAP)
3. Where bottlenecks occur in each type
"""

import os
import sys
from pathlib import Path
import time
import numpy as np
import h5py
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set thread limits
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import torch
    from torch.amp import autocast
    torch_available = True
except ImportError:
    torch_available = False
    print("❌ PyTorch not available")
    sys.exit(1)

from solvers import SciPySolver, SeededLAPSolver, LAPSolver
from gnn import OneGNN, compute_row_features, compute_row_features_torch


class ComprehensivePipelineProfiler:
    """Profile pipeline performance across all problem types."""
    
    def __init__(self, model_path: str, data_dir: str, *, device: str = None):
        self.data_dir = Path(data_dir)
        self.device = self._resolve_device(device)
        self.use_cuda = 'cuda' in self.device
        self.row_only = True
        
        print(f"Using device: {self.device} {'(GPU)' if self.use_cuda else '(CPU)'}")
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Initialize solvers
        self.scipy_solver = SciPySolver()
        self.lap_solver = LAPSolver()
        self.seeded_solver = SeededLAPSolver()
    
    def _resolve_device(self, device_request: str | None) -> str:
        """Resolve which torch device to use."""
        if device_request is None or device_request == "auto":
            if torch.cuda.is_available():
                return "cuda"
            print("⚠️ CUDA not available, falling back to CPU")
            return "cpu"
        
        if device_request == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            print("⚠️ Requested CUDA but it is unavailable, falling back to CPU")
            return "cpu"
        
        return "cpu"
    
    def _load_model(self, model_path: str):
        """Load GNN model with optimized GPU settings."""
        print(f"\nLoading model: {Path(model_path).name}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            hidden_dim = checkpoint.get('hidden_dim', 192)
            layers = checkpoint.get('layers', 4)
            dropout = checkpoint.get('dropout', 0.1)
            
            nested_cfg = checkpoint.get('config')
            if isinstance(nested_cfg, dict):
                hidden_dim = nested_cfg.get('hidden_dim', hidden_dim)
                layers = nested_cfg.get('layers', layers)
                dropout = nested_cfg.get('dropout', dropout)
        else:
            state_dict = checkpoint
            hidden_dim, layers, dropout = 192, 4, 0.1
        
        in_dim = compute_row_features(np.zeros((1, 1))).shape[1]
        model = OneGNN(in_dim=in_dim, hidden=hidden_dim, layers=layers, dropout=dropout)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        # Defensive AMP
        self.amp_enabled = self.use_cuda and not self.row_only
        
        params = sum(p.numel() for p in model.parameters())
        print(f"  Hidden: {hidden_dim}, Layers: {layers}, Features: {in_dim}")
        print(f"  Parameters: {params:,}")
        print(f"  AMP enabled: {self.amp_enabled}")
        
        # Set model first, then warmup
        self.model = model
        
        if self.use_cuda:
            self._gpu_warmup()
        
        return model
    
    def _gpu_warmup(self):
        """GPU warmup to avoid first-inference overhead."""
        print("  GPU warmup...", end=' ', flush=True)
        
        dummy_size = 64
        dummy_C = np.random.rand(dummy_size, dummy_size).astype(np.float64)
        
        for _ in range(3):
            with torch.inference_mode():
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
        
        if self.use_cuda:
            torch.cuda.synchronize()
        
        print("done ✓")
    
    def profile_pipeline(self, C: np.ndarray) -> Dict:
        """Profile all stages of the pipeline with detailed timing."""
        C = np.asarray(C, dtype=np.float64)
        n = C.shape[0]
        
        timings = {}
        cost_tensor_on_gpu = None  # Track if we have C on GPU already
        
        # ============================================================
        # Stage 1: Feature Computation
        # ============================================================
        if 'cuda' in self.device:
            # GPU path
            t0 = time.perf_counter()
            cost_tensor = torch.from_numpy(C).float().to(self.device)
            cost_tensor_on_gpu = cost_tensor  # Save reference for V computation
            timings['data_to_gpu_ms'] = (time.perf_counter() - t0) * 1000
            
            t0 = time.perf_counter()
            row_feat_tensor = compute_row_features_torch(cost_tensor)
            if self.use_cuda:
                torch.cuda.synchronize()
            timings['feature_compute_gpu_ms'] = (time.perf_counter() - t0) * 1000
            
            t0 = time.perf_counter()
            row_tensor = row_feat_tensor.unsqueeze(0)
            cost_batch = cost_tensor.unsqueeze(0)
            timings['tensor_prep_ms'] = (time.perf_counter() - t0) * 1000
        else:
            # CPU path
            t0 = time.perf_counter()
            row_feat = compute_row_features(C)
            timings['feature_compute_cpu_ms'] = (time.perf_counter() - t0) * 1000
            
            t0 = time.perf_counter()
            row_tensor = torch.from_numpy(row_feat).float().unsqueeze(0).to(self.device)
            cost_batch = torch.from_numpy(C).float().unsqueeze(0).to(self.device)
            timings['tensor_prep_ms'] = (time.perf_counter() - t0) * 1000
        
        # ============================================================
        # Stage 2: GNN Inference
        # ============================================================
        with torch.inference_mode():
            mask = torch.ones((1, n), dtype=torch.bool, device=self.device)
            
            t0 = time.perf_counter()
            with autocast('cuda', enabled=self.amp_enabled):
                outputs = self.model(row_tensor, cost=cost_batch, mask=mask)
            
            if self.use_cuda:
                torch.cuda.synchronize()
            timings['gnn_inference_ms'] = (time.perf_counter() - t0) * 1000
        
        # ============================================================
        # Stage 3: Dual Potential Extraction (GPU-accelerated)
        # ============================================================
        t0 = time.perf_counter()
        u_pred_tensor = outputs['u'].squeeze(0)[:n]  # Keep on GPU
        timings['u_extract_ms'] = (time.perf_counter() - t0) * 1000
        
        t0 = time.perf_counter()
        if self.use_cuda:
            # GPU path: Reuse cost_tensor_on_gpu to avoid redundant transfer
            if cost_tensor_on_gpu is not None:
                # C is already on GPU from Stage 1 - reuse it!
                v_pred_tensor = torch.min(cost_tensor_on_gpu - u_pred_tensor[:, None], dim=0)[0]
            else:
                # Fallback: transfer C to GPU (shouldn't happen in GPU path)
                C_tensor = torch.from_numpy(C).to(self.device, dtype=torch.float32)
                v_pred_tensor = torch.min(C_tensor - u_pred_tensor[:, None], dim=0)[0]
            
            torch.cuda.synchronize()
            v_pred = v_pred_tensor.cpu().numpy()
            u_pred = u_pred_tensor.cpu().numpy()
        else:
            # CPU path: Use NumPy as before
            u_pred = u_pred_tensor.cpu().numpy()
            v_pred = np.min(C - u_pred[:, None], axis=0)
        timings['v_compute_ms'] = (time.perf_counter() - t0) * 1000
        
        u_pred = u_pred.astype(np.float64)
        v_pred = v_pred.astype(np.float64)
        
        # ============================================================
        # Stage 4: Seeded LAP Solve
        # ============================================================
        t0 = time.perf_counter()
        result = self.seeded_solver.solve(C, u=u_pred, v=v_pred)
        timings['seeded_lap_ms'] = (time.perf_counter() - t0) * 1000
        
        if isinstance(result, dict):
            row_ind = result['row_ind']
            col_ind = result['col_ind']
            cost = result['cost']
        else:
            row_ind, col_ind, cost = result
        
        # ============================================================
        # Stage 5: Solution Validation (optional)
        # ============================================================
        t0 = time.perf_counter()
        assignment_cost = C[row_ind, col_ind].sum()
        timings['validation_ms'] = (time.perf_counter() - t0) * 1000
        
        # ============================================================
        # Aggregate timings
        # ============================================================
        # GNN overhead (everything except seeded LAP)
        timings['gnn_overhead_ms'] = sum([
            timings.get('data_to_gpu_ms', 0),
            timings.get('feature_compute_gpu_ms', 0),
            timings.get('feature_compute_cpu_ms', 0),
            timings.get('tensor_prep_ms', 0),
            timings.get('gnn_inference_ms', 0),
            timings.get('u_to_cpu_ms', 0),
            timings.get('v_compute_ms', 0),
        ])
        
        # Total pipeline time
        timings['total_pipeline_ms'] = sum([
            timings.get('data_to_gpu_ms', 0),
            timings.get('feature_compute_gpu_ms', 0),
            timings.get('feature_compute_cpu_ms', 0),
            timings.get('tensor_prep_ms', 0),
            timings.get('gnn_inference_ms', 0),
            timings.get('u_to_cpu_ms', 0),
            timings.get('v_compute_ms', 0),
            timings.get('seeded_lap_ms', 0),
            timings.get('validation_ms', 0)
        ])
        
        return {
            'timings': timings,
            'cost': cost,
            'assignment_cost': assignment_cost,
            'u_pred': u_pred,
            'v_pred': v_pred
        }
    
    def profile_baselines(self, C: np.ndarray) -> Dict:
        """Profile baseline solvers for comparison."""
        baselines = {}
        
        # SciPy
        t0 = time.perf_counter()
        scipy_result = self.scipy_solver.solve(C)
        baselines['scipy_ms'] = (time.perf_counter() - t0) * 1000
        
        if isinstance(scipy_result, dict):
            baselines['scipy_cost'] = scipy_result['cost']
        else:
            baselines['scipy_cost'] = scipy_result[2]
        
        # LAP (unseeded)
        t0 = time.perf_counter()
        lap_result = self.lap_solver.solve(C)
        baselines['lap_ms'] = (time.perf_counter() - t0) * 1000
        
        if isinstance(lap_result, dict):
            baselines['lap_cost'] = lap_result['cost']
        else:
            baselines['lap_cost'] = lap_result[2]
        
        return baselines
    
    def analyze_dual_quality(self, C: np.ndarray, u: np.ndarray, v: np.ndarray,
                            u_true: np.ndarray, v_true: np.ndarray) -> Dict:
        """Analyze quality of predicted dual potentials."""
        n = C.shape[0]
        
        # Reduced costs: should be non-negative for dual feasibility
        reduced_costs = C - u[:, None] - v[None, :]
        
        # Analysis metrics
        metrics = {
            # Dual feasibility violations
            'min_reduced_cost': np.min(reduced_costs),
            'negative_count': np.sum(reduced_costs < -1e-10),
            'negative_fraction': np.sum(reduced_costs < -1e-10) / (n * n),
            'max_violation': np.abs(np.min([0, np.min(reduced_costs)])),
            
            # Reduced cost distribution
            'mean_reduced_cost': np.mean(reduced_costs),
            'std_reduced_cost': np.std(reduced_costs),
            'median_reduced_cost': np.median(reduced_costs),
            
            # Complementary slackness quality
            'min_per_row': np.min(reduced_costs, axis=1),
            'min_per_col': np.min(reduced_costs, axis=0),
            'rows_with_zero': np.sum(np.min(reduced_costs, axis=1) < 1e-10),
            'cols_with_zero': np.sum(np.min(reduced_costs, axis=0) < 1e-10),
            
            # Prediction errors
            'u_mae': np.mean(np.abs(u - u_true)),
            'v_mae': np.mean(np.abs(v - v_true)),
            'u_max_error': np.max(np.abs(u - u_true)),
            'v_max_error': np.max(np.abs(v - v_true)),
            
            # Cost statistics
            'cost_mean': np.mean(C),
            'cost_std': np.std(C),
            'cost_min': np.min(C),
            'cost_max': np.max(C),
            'cost_range': np.max(C) - np.min(C),
        }
        
        # Count how many zeros/ties in reduced costs (indicates optimal assignments)
        tight_threshold = 1e-8
        metrics['tight_edges'] = np.sum(reduced_costs < tight_threshold)
        metrics['tight_fraction'] = metrics['tight_edges'] / (n * n)
        
        return metrics
    
    def analyze_initial_assignment(self, C: np.ndarray, u: np.ndarray, v: np.ndarray) -> Dict:
        """Analyze the initial assignment that would be derived from predicted duals."""
        n = C.shape[0]
        reduced_costs = C - u[:, None] - v[None, :]
        
        # Greedy assignment based on reduced costs
        assignment = np.argmin(reduced_costs, axis=1)
        
        # Check for conflicts (multiple rows assigned to same column)
        unique_assignments = len(np.unique(assignment))
        conflicts = n - unique_assignments
        
        # Cost of this assignment
        assignment_cost = np.sum(C[np.arange(n), assignment])
        
        # Check how many assignments are on tight edges
        tight_assignments = np.sum(reduced_costs[np.arange(n), assignment] < 1e-8)
        
        return {
            'conflicts': conflicts,
            'conflict_rate': conflicts / n,
            'unique_assignments': unique_assignments,
            'assignment_cost': assignment_cost,
            'tight_assignments': tight_assignments,
            'tight_rate': tight_assignments / n,
        }
    
    def visualize_reduced_costs(self, problem: Dict, save_path: Optional[str] = None):
        """Visualize reduced cost matrix to identify patterns."""
        try:
            import matplotlib.pyplot as plt
            matplotlib_available = True
        except ImportError:
            print("⚠️ Matplotlib not available - skipping visualization")
            return
        
        C = problem['C']
        u_true = problem['u_true']
        v_true = problem['v_true']
        
        # Predict duals
        with torch.inference_mode():
            if self.use_cuda:
                cost_tensor = torch.from_numpy(C).float().to(self.device)
                row_feat_tensor = compute_row_features_torch(cost_tensor)
                row_tensor = row_feat_tensor.unsqueeze(0)
                cost_batch = cost_tensor.unsqueeze(0)
            else:
                row_feat = compute_row_features(C)
                row_tensor = torch.from_numpy(row_feat).float().unsqueeze(0).to(self.device)
                cost_batch = torch.from_numpy(C).float().unsqueeze(0).to(self.device)
            
            mask = torch.ones((1, C.shape[0]), dtype=torch.bool, device=self.device)
            with autocast('cuda', enabled=self.amp_enabled):
                u_pred, _ = self.model(row_tensor, cost=cost_batch, mask=mask)
            
            u_pred = u_pred.squeeze(0).cpu().numpy().astype(np.float64)
        
        # Compute V
        v_pred = np.min(C - u_pred[:, None], axis=0)
        reduced_costs = C - u_pred[:, None] - v_pred[None, :]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Original cost matrix
        im1 = axes[0].imshow(C, cmap='viridis', aspect='auto')
        axes[0].set_title(f"Cost Matrix ({problem.get('type', 'unknown')})")
        axes[0].set_xlabel('Column')
        axes[0].set_ylabel('Row')
        plt.colorbar(im1, ax=axes[0])
        
        # Reduced costs
        vmax = np.abs(reduced_costs).max()
        im2 = axes[1].imshow(reduced_costs, cmap='RdBu_r', aspect='auto', 
                            vmin=-vmax, vmax=vmax)
        axes[1].set_title('Reduced Costs (should be ≥ 0)')
        axes[1].set_xlabel('Column')
        axes[1].set_ylabel('Row')
        plt.colorbar(im2, ax=axes[1])
        
        # Histogram of reduced costs
        axes[2].hist(reduced_costs.flatten(), bins=100, edgecolor='black', alpha=0.7)
        axes[2].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero (optimal)')
        axes[2].set_xlabel('Reduced Cost')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Reduced Cost Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved visualization to: {save_path}")
        else:
            default_path = f'reduced_costs_{problem.get("type", "unknown")}_n{C.shape[0]}.png'
            plt.savefig(default_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {default_path}")
        
        plt.close()
    
    def load_problems_by_type(self, size: int, problem_types: List[str], max_per_type: int = 10) -> Dict[str, List]:
        """Load instances for multiple problem types."""
        problems = defaultdict(list)
        
        # Find appropriate dataset (check clean datasets first)
        if size <= 1024:
            test_file = self.data_dir / "generated/processed_clean/small/full/test.h5"
            if not test_file.exists():
                test_file = self.data_dir / "generated/processed/small/full/test.h5"
        elif size <= 1536:
            test_file = self.data_dir / "generated/processed_clean/mid_1536/full/test.h5"
            if not test_file.exists():
                test_file = self.data_dir / "generated/processed/mid_1536/full/test.h5"
        elif size <= 2048:
            test_file = self.data_dir / "generated/processed_clean/mid_2048/full/test.h5"
            if not test_file.exists():
                test_file = self.data_dir / "generated/processed/mid_2048/full/test.h5"
        elif size <= 3072:
            test_file = self.data_dir / "generated/processed_clean/mid_3072/full/test.h5"
            if not test_file.exists():
                test_file = self.data_dir / "generated/processed/mid_3072/full/test.h5"
        elif size <= 4096:
            test_file = self.data_dir / "generated/processed_clean/large_4096/full/test.h5"
            if not test_file.exists():
                test_file = self.data_dir / "generated/processed/large_4096/full/test.h5"
        else:
            print(f"⚠️ Unsupported size: {size}")
            return problems
        
        if not test_file.exists():
            print(f"⚠️ File not found: {test_file}")
            return problems
        
        print(f"\nLoading problems from: {test_file.name}")
        print(f"  Target size: {size}, Max per type: {max_per_type}")
        print(f"  Types: {', '.join(problem_types)}")
        
        with h5py.File(test_file, 'r') as f:
            type_counts = defaultdict(int)
            
            for i in range(len(f['n'])):
                if f['n'][i] == size:
                    prob_type = f['family'][i].decode('utf-8') if isinstance(f['family'][i], bytes) else f['family'][i]
                    
                    if prob_type in problem_types and type_counts[prob_type] < max_per_type:
                        C_flat = f['C'][i]
                        u_true = f['u'][i]
                        v_true = f['v'][i]
                        n = f['n'][i]
                        
                        C = C_flat.reshape(n, n)
                        problems[prob_type].append({
                            'C': C,
                            'u_true': u_true,
                            'v_true': v_true,
                            'index': i,
                            'type': prob_type,
                            'size': n
                        })
                        type_counts[prob_type] += 1
        
        print(f"  Loaded: {dict(type_counts)}")
        return problems
    
    def analyze_all_types(self, size: int, problem_types: List[str], max_per_type: int = 10):
        """Comprehensive analysis of all problem types."""
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE PIPELINE ANALYSIS - ALL PROBLEM TYPES")
        print("=" * 80)
        print(f"Problem size: {size}×{size}")
        print(f"Instances per type: {max_per_type}")
        
        problems_by_type = self.load_problems_by_type(size, problem_types, max_per_type)
        
        if not problems_by_type:
            print("❌ No problems found!")
            return
        
        # Collect results for each type
        type_results = {}
        
        for prob_type in sorted(problems_by_type.keys()):
            print(f"\n{'='*80}")
            print(f"ANALYZING: {prob_type.upper()}")
            print(f"{'='*80}")
            
            problems = problems_by_type[prob_type]
            
            all_timings = defaultdict(list)
            all_baselines = defaultdict(list)
            all_predictions = defaultdict(list)
            
            for idx, problem in enumerate(problems, 1):
                C = problem['C']
                
                # Profile baselines
                baselines = self.profile_baselines(C)
                for key, val in baselines.items():
                    all_baselines[key].append(val)
                
                # Profile full pipeline
                pipeline = self.profile_pipeline(C)
                timings = pipeline['timings']
                
                for key, val in timings.items():
                    all_timings[key].append(val)
                
                # Prediction quality
                u_mae = np.mean(np.abs(pipeline['u_pred'] - problem['u_true']))
                v_mae = np.mean(np.abs(pipeline['v_pred'] - problem['v_true']))
                all_predictions['u_mae'].append(u_mae)
                all_predictions['v_mae'].append(v_mae)
                
                # Print detailed breakdown for EVERY instance
                print(f"\n{'='*80}")
                print(f"Instance {idx}/{len(problems)} (index={problem['index']})")
                print(f"{'='*80}")
                
                # Detailed timing breakdown
                print(f"\n📊 TIMING BREAKDOWN:")
                print(f"{'Stage':<30} {'Time (ms)':<12} {'% of Total':<12}")
                print("-" * 54)
                
                total = timings['total_pipeline_ms']
                
                if 'cuda' in self.device:
                    stages = [
                        ('Data transfer to GPU', timings.get('data_to_gpu_ms', 0)),
                        ('Feature computation (GPU)', timings.get('feature_compute_gpu_ms', 0)),
                        ('Tensor preparation', timings.get('tensor_prep_ms', 0)),
                        ('GNN inference', timings.get('gnn_inference_ms', 0)),
                        ('U extraction', timings.get('u_extract_ms', 0)),
                        ('V computation (GPU)', timings.get('v_compute_ms', 0)),
                        ('Seeded LAP solve', timings.get('seeded_lap_ms', 0)),
                        ('Validation', timings.get('validation_ms', 0)),
                    ]
                else:
                    stages = [
                        ('Feature computation (CPU)', timings.get('feature_compute_cpu_ms', 0)),
                        ('Tensor preparation', timings.get('tensor_prep_ms', 0)),
                        ('GNN inference', timings.get('gnn_inference_ms', 0)),
                        ('U extraction', timings.get('u_extract_ms', 0)),
                        ('V computation (CPU)', timings.get('v_compute_ms', 0)),
                        ('Seeded LAP solve', timings.get('seeded_lap_ms', 0)),
                        ('Validation', timings.get('validation_ms', 0)),
                    ]
                
                for stage_name, stage_time in stages:
                    pct = (stage_time / total * 100) if total > 0 else 0
                    icon = "🔴" if pct > 30 else ("🟡" if pct > 10 else "🟢")
                    print(f"{icon} {stage_name:<28} {stage_time:>10.3f}   {pct:>10.1f}%")
                
                print("-" * 54)
                print(f"{'TOTAL PIPELINE':<30} {total:>10.3f}   {'100.0%':>10}")
                
                print(f"\n🏁 COMPARISON:")
                print(f"  SciPy:        {baselines['scipy_ms']:>8.2f} ms (cost={baselines['scipy_cost']:.6f})")
                print(f"  LAP unseeded: {baselines['lap_ms']:>8.2f} ms (cost={baselines['lap_cost']:.6f})")
                print(f"  GNN+Seeded:   {total:>8.2f} ms (cost={pipeline['cost']:.6f})")
                
                speedup_scipy = baselines['scipy_ms'] / total
                speedup_lap = baselines['lap_ms'] / total
                
                print(f"\n📈 SPEEDUP:")
                print(f"  vs SciPy: {speedup_scipy:.2f}x {'✅' if speedup_scipy > 1 else '❌'}")
                print(f"  vs LAP:   {speedup_lap:.2f}x {'✅' if speedup_lap > 1 else '❌'}")
                
                # Analyze bottleneck
                max_stage = max(stages, key=lambda x: x[1])
                print(f"\n⚠️  BOTTLENECK: {max_stage[0]} ({max_stage[1]:.2f}ms, {max_stage[1]/total*100:.1f}%)")
                
                # Prediction quality
                print(f"\n🎯 PREDICTION QUALITY:")
                print(f"  U MAE: {u_mae:.6f}")
                print(f"  V MAE: {v_mae:.6f}")
            
            # Compute statistics
            avg_pipeline = statistics.mean(all_timings['total_pipeline_ms'])
            avg_lap = statistics.mean(all_baselines['lap_ms'])
            avg_scipy = statistics.mean(all_baselines['scipy_ms'])
            avg_gnn_overhead = statistics.mean(all_timings['gnn_overhead_ms'])
            avg_seeded_lap = statistics.mean(all_timings['seeded_lap_ms'])
            avg_u_mae = statistics.mean(all_predictions['u_mae'])
            avg_v_mae = statistics.mean(all_predictions['v_mae'])
            
            speedup_vs_lap = avg_lap / avg_pipeline
            speedup_vs_scipy = avg_scipy / avg_pipeline
            
            type_results[prob_type] = {
                'count': len(problems),
                'avg_pipeline_ms': avg_pipeline,
                'avg_lap_ms': avg_lap,
                'avg_scipy_ms': avg_scipy,
                'avg_gnn_overhead_ms': avg_gnn_overhead,
                'avg_seeded_lap_ms': avg_seeded_lap,
                'speedup_vs_lap': speedup_vs_lap,
                'speedup_vs_scipy': speedup_vs_scipy,
                'avg_u_mae': avg_u_mae,
                'avg_v_mae': avg_v_mae,
                'timings': all_timings,
                'baselines': all_baselines,
            }
            
            # Print summary
            print(f"\n📊 SUMMARY for {prob_type.upper()}:")
            print(f"  Instances: {len(problems)}")
            print(f"  Pipeline: {avg_pipeline:.2f}ms (GNN overhead: {avg_gnn_overhead:.2f}ms, Seeded LAP: {avg_seeded_lap:.2f}ms)")
            print(f"  LAP unseeded: {avg_lap:.2f}ms")
            print(f"  SciPy: {avg_scipy:.2f}ms")
            print(f"  Speedup vs LAP: {speedup_vs_lap:.2f}x {'✅' if speedup_vs_lap > 1 else '❌'}")
            print(f"  Speedup vs SciPy: {speedup_vs_scipy:.2f}x {'✅' if speedup_vs_scipy > 1 else '❌'}")
            print(f"  Prediction quality: U MAE={avg_u_mae:.6f}, V MAE={avg_v_mae:.6f}")
            
            # Verdict
            if speedup_vs_lap < 0.9:
                overhead_vs_benefit = avg_gnn_overhead / avg_lap
                print(f"  ⚠️  VERDICT: GNN HURTS performance ({1/speedup_vs_lap:.2f}x slower)")
                print(f"      → GNN overhead ({avg_gnn_overhead:.2f}ms) is {overhead_vs_benefit:.1f}x the LAP solve time")
                print(f"      → RECOMMENDATION: Skip GNN, use unseeded LAP")
            elif speedup_vs_lap > 1.1:
                time_saved = avg_lap - avg_pipeline
                print(f"  ✅ VERDICT: GNN HELPS performance ({speedup_vs_lap:.2f}x faster)")
                print(f"      → Saves {time_saved:.2f}ms per instance")
                print(f"      → RECOMMENDATION: Keep using GNN")
            else:
                print(f"  ⚖️  VERDICT: GNN is NEUTRAL (within 10% of LAP)")
                print(f"      → RECOMMENDATION: Either approach acceptable")
        
        # Overall comparison
        print(f"\n{'='*80}")
        print("OVERALL COMPARISON - ALL TYPES")
        print(f"{'='*80}\n")
        
        print(f"{'Type':<12} {'Count':<6} {'Pipeline':<10} {'LAP':<10} {'Speedup':<10} {'U MAE':<12} {'Verdict':<15}")
        print("-" * 90)
        
        for prob_type in sorted(type_results.keys()):
            r = type_results[prob_type]
            speedup = r['speedup_vs_lap']
            
            if speedup < 0.9:
                verdict = "❌ Skip GNN"
            elif speedup > 1.1:
                verdict = "✅ Use GNN"
            else:
                verdict = "⚖️ Neutral"
            
            print(f"{prob_type:<12} {r['count']:<6} {r['avg_pipeline_ms']:>8.2f}ms {r['avg_lap_ms']:>8.2f}ms "
                  f"{speedup:>8.2f}x {r['avg_u_mae']:>10.6f}  {verdict}")
        
        print("-" * 90)
        
        # Calculate weighted average
        total_instances = sum(r['count'] for r in type_results.values())
        weighted_speedup = sum(r['speedup_vs_lap'] * r['count'] for r in type_results.values()) / total_instances
        
        print(f"{'OVERALL':<12} {total_instances:<6} {'':<10} {'':<10} {weighted_speedup:>8.2f}x (weighted)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive pipeline analysis for all problem types",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model', type=str,
                       default='one_gnn_mid1536_full_clean_h192L4.pt',
                       help='Model checkpoint to test')
    parser.add_argument('--size', type=int, default=1024,
                       help='Problem size to test')
    parser.add_argument('--instances', type=int, default=10,
                       help='Number of instances per type')
    parser.add_argument('--types', type=str, nargs='+',
                       default=['uniform', 'sparse', 'block', 'tie', 'metric'],
                       help='Problem types to analyze')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to run the GNN on')
    
    args = parser.parse_args()
    
    # Find model path
    model_path = None
    checkpoint_dirs = [
        project_root / "gnn" / "checkpoints_clean",
        project_root / "checkpoints",
        project_root,
    ]
    
    for checkpoint_dir in checkpoint_dirs:
        candidate = checkpoint_dir / args.model
        if candidate.exists():
            model_path = str(candidate)
            break
    
    if model_path is None:
        print(f"❌ Model not found: {args.model}")
        print(f"Searched in: {[str(d) for d in checkpoint_dirs]}")
        sys.exit(1)
    
    data_dir = project_root / "data"
    
    profiler = ComprehensivePipelineProfiler(model_path, data_dir, device=args.device)
    profiler.analyze_all_types(args.size, args.types, args.instances)


if __name__ == "__main__":
    main()
