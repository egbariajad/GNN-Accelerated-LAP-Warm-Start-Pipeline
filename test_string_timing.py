#!/usr/bin/env python3
"""
STRING Dataset Timing Benchmark - Focus on Speed, Not Quality

Since LAP solvers always find optimal solutions, we measure:
1. Runtime (GNN+Seeded vs Cold LAP)
2. Iteration counts (if available)
3. Speedup factors
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import torch
import time
from tqdm import tqdm

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from solvers import SeededLAPSolver, LAPSolver, SciPySolver
from gnn import OneGNN, compute_row_features, compute_row_features_torch

def time_solver_multiple_runs(solve_fn, n_runs=5):
    """Time a solver with multiple runs for accuracy."""
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        result = solve_fn()
        t1 = time.time()
        times.append(t1 - t0)
    
    return {
        'min': min(times),
        'max': max(times),
        'mean': np.mean(times),
        'median': np.median(times),
        'std': np.std(times),
    }

class STRINGTimingBenchmark:
    """Benchmark focusing on runtime performance."""
    
    def __init__(self, model_path: str, device='cuda'):
        self.device = device
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()
        
        # Initialize solvers
        self.seeded_solver = SeededLAPSolver()
        self.unseeded_solver = LAPSolver()
        self.scipy_solver = SciPySolver()
    
    def _load_model(self):
        """Load GNN model."""
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        nested_cfg = checkpoint.get('config', {})
        model_cfg = nested_cfg.get('model', {}) if nested_cfg else checkpoint
        
        in_dim = model_cfg.get('row_feat_dim', 21)
        
        self.model = OneGNN(
            in_dim=in_dim,
            hidden=model_cfg.get('hidden_dim', 192),
            layers=model_cfg.get('layers', 4),
            dropout=model_cfg.get('dropout', 0.1),
        )
        
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # GPU warmup
        if 'cuda' in self.device:
            dummy_C = np.random.rand(64, 64).astype(np.float64)
            for _ in range(3):
                self._predict_duals(dummy_C)
            torch.cuda.synchronize()
    
    def _predict_duals(self, C: np.ndarray):
        """Predict dual variables."""
        n = C.shape[0]
        
        with torch.inference_mode():
            if 'cuda' in self.device:
                cost_tensor = torch.from_numpy(C).float().to(self.device)
                row_feat_tensor = compute_row_features_torch(cost_tensor)
                row_tensor = row_feat_tensor.unsqueeze(0)
                cost_batch = cost_tensor.unsqueeze(0)
            else:
                row_feat = compute_row_features(C)
                row_tensor = torch.from_numpy(row_feat).float().unsqueeze(0).to(self.device)
                cost_batch = torch.from_numpy(C).float().unsqueeze(0).to(self.device)
            
            mask = torch.ones((1, n), dtype=torch.bool, device=self.device)
            outputs = self.model(row_tensor, cost=cost_batch, mask=mask)
            
            u_pred_tensor = outputs['u'].squeeze(0)[:n]
            
            if 'cuda' in self.device:
                C_tensor = torch.from_numpy(C).float().to(self.device)
                v_pred_tensor = torch.min(C_tensor - u_pred_tensor[:, None], dim=0)[0]
                torch.cuda.synchronize()
                v_pred = v_pred_tensor.cpu().numpy()
                u_pred = u_pred_tensor.cpu().numpy()
            else:
                u_pred = u_pred_tensor.cpu().numpy()
                v_pred = np.min(C - u_pred[:, None], axis=0)
        
        return u_pred.astype(np.float64), v_pred.astype(np.float64)
    
    def benchmark_instance(self, C: np.ndarray, n_runs=5):
        """Benchmark single instance with timing focus."""
        n = C.shape[0]
        
        results = {'n': n}
        
        # === 1. SciPy Baseline (Optimal) ===
        scipy_timing = time_solver_multiple_runs(lambda: self.scipy_solver.solve(C), n_runs)
        _, _, cost_optimal = self.scipy_solver.solve(C)
        results['scipy_time'] = scipy_timing['median']
        results['cost_optimal'] = cost_optimal
        
        # === 2. Unseeded LAP (Cold Start) ===
        cold_timing = time_solver_multiple_runs(lambda: self.unseeded_solver.solve(C), n_runs)
        _, _, cost_cold = self.unseeded_solver.solve(C)
        results['cold_time'] = cold_timing['median']
        results['cost_cold'] = cost_cold
        
        # === 3. GNN Prediction ===
        gnn_times = []
        for _ in range(n_runs):
            t0 = time.time()
            u_pred, v_pred = self._predict_duals(C)
            gnn_times.append(time.time() - t0)
        results['gnn_time'] = np.median(gnn_times)
        
        # === 4. Seeded LAP (Warm Start) ===
        seeded_timing = time_solver_multiple_runs(
            lambda: self.seeded_solver.solve(C, u_pred, v_pred), n_runs
        )
        _, _, cost_seeded = self.seeded_solver.solve(C, u_pred, v_pred)
        results['seeded_time'] = seeded_timing['median']
        results['cost_seeded'] = cost_seeded
        
        # === Total pipeline time ===
        results['pipeline_time'] = results['gnn_time'] + results['seeded_time']
        
        # === Speedups ===
        results['speedup_vs_cold'] = results['cold_time'] / results['pipeline_time']
        results['speedup_vs_scipy'] = results['scipy_time'] / results['pipeline_time']
        
        # === Quality (should all be optimal) ===
        results['gap_seeded'] = (cost_seeded - cost_optimal) / abs(cost_optimal) * 100
        results['gap_cold'] = (cost_cold - cost_optimal) / abs(cost_optimal) * 100
        
        return results
    
    def run_benchmark(self, data_path: str, max_instances=None):
        """Run benchmark on dataset."""
        print(f"\n{'='*80}")
        print(f"STRING Dataset Timing Benchmark")
        print(f"{'='*80}")
        print(f"Model: {self.model_path.name}")
        print(f"Device: {self.device}")
        print(f"Data: {data_path}")
        
        # Load instances
        instances = []
        with h5py.File(data_path, 'r') as f:
            n_inst = len(f.keys())
            if max_instances:
                n_inst = min(n_inst, max_instances)
            
            for i in range(n_inst):
                inst = f[f'instance_{i}']
                C = inst['cost_matrix'][:]
                instances.append(C)
        
        print(f"Instances: {len(instances)}")
        print(f"{'='*80}\n")
        
        # Benchmark each instance
        all_results = []
        
        for idx, C in enumerate(tqdm(instances, desc="Benchmarking")):
            print(f"\nInstance {idx} (size {C.shape[0]}x{C.shape[1]}):")
            results = self.benchmark_instance(C, n_runs=3)
            
            print(f"  SciPy:          {results['scipy_time']*1000:6.2f} ms")
            print(f"  Cold LAP:       {results['cold_time']*1000:6.2f} ms")
            print(f"  GNN:            {results['gnn_time']*1000:6.2f} ms")
            print(f"  Seeded LAP:     {results['seeded_time']*1000:6.2f} ms")
            print(f"  Pipeline Total: {results['pipeline_time']*1000:6.2f} ms")
            print(f"  Speedup vs Cold: {results['speedup_vs_cold']:.2f}x")
            print(f"  Speedup vs SciPy: {results['speedup_vs_scipy']:.2f}x")
            
            all_results.append(results)
        
        # Aggregate statistics
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results):
        """Print summary statistics."""
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")
        
        # Size info
        sizes = [r['n'] for r in results]
        print(f"\nProblem Sizes:")
        print(f"  Range: {min(sizes)} - {max(sizes)}")
        print(f"  Mean:  {np.mean(sizes):.1f}")
        
        # Timing stats
        print(f"\nRuntime (median across instances):")
        print(f"  SciPy:      {np.median([r['scipy_time'] for r in results])*1000:7.2f} ms")
        print(f"  Cold LAP:   {np.median([r['cold_time'] for r in results])*1000:7.2f} ms")
        print(f"  GNN:        {np.median([r['gnn_time'] for r in results])*1000:7.2f} ms")
        print(f"  Seeded LAP: {np.median([r['seeded_time'] for r in results])*1000:7.2f} ms")
        print(f"  Pipeline:   {np.median([r['pipeline_time'] for r in results])*1000:7.2f} ms")
        
        # Speedups
        speedup_cold = [r['speedup_vs_cold'] for r in results]
        speedup_scipy = [r['speedup_vs_scipy'] for r in results]
        
        print(f"\nSpeedup vs Cold LAP:")
        print(f"  Mean:   {np.mean(speedup_cold):.2f}x")
        print(f"  Median: {np.median(speedup_cold):.2f}x")
        print(f"  Min:    {np.min(speedup_cold):.2f}x")
        print(f"  Max:    {np.max(speedup_cold):.2f}x")
        
        print(f"\nSpeedup vs SciPy:")
        print(f"  Mean:   {np.mean(speedup_scipy):.2f}x")
        print(f"  Median: {np.median(speedup_scipy):.2f}x")
        
        # Quality (should be ~0%)
        gaps = [r['gap_seeded'] for r in results]
        print(f"\nPrimal Gap (% above optimal):")
        print(f"  Mean:   {np.mean(gaps):.6f}%")
        print(f"  Max:    {np.max(gaps):.6f}%")
        print(f"  (All should be ~0% since LAP finds exact optimal)")
        
        print(f"\n{'='*80}")
        
        # Interpretation
        median_speedup_cold = np.median(speedup_cold)
        median_speedup_scipy = np.median(speedup_scipy)
        
        print("\nINTERPRETATION:")
        if median_speedup_cold > 1.2:
            print(f"  ✅ GNN warm-start provides {median_speedup_cold:.2f}x speedup vs cold LAP!")
            print(f"     The seeded solver converges faster with GNN-predicted duals.")
        elif median_speedup_cold > 0.8:
            print(f"  ⚠️  Speedup vs cold LAP is marginal ({median_speedup_cold:.2f}x)")
            print(f"     GNN overhead (~{np.median([r['gnn_time'] for r in results])*1000:.1f}ms) is comparable to savings.")
        else:
            print(f"  ❌ No speedup vs cold LAP ({median_speedup_cold:.2f}x)")
            print(f"     GNN overhead dominates on these small/easy problems.")
            print(f"     Try larger instances (2000+) or harder distributions.")
        
        if median_speedup_scipy > 1.5:
            print(f"  ✅ Good speedup vs SciPy ({median_speedup_scipy:.2f}x)")
        
        print(f"{'='*80}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gnn/checkpoints_clean/progressive_clean_best.pt')
    parser.add_argument('--data', type=str, default='data/string_dataset/processed/test.h5')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-instances', type=int, default=None)
    args = parser.parse_args()
    
    benchmark = STRINGTimingBenchmark(args.model, device=args.device)
    results = benchmark.run_benchmark(args.data, max_instances=args.max_instances)


if __name__ == "__main__":
    main()
