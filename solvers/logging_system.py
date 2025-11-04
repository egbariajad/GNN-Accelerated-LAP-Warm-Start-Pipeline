"""
Logging Module for LAP Benchmarking Suite

Provides structured logging capabilities for tracking experiments, results,
and performance metrics over time. Supports multiple output formats including
CSV, JSON, and human-readable logs.
"""

import os
import json
import csv
import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import platform
import numpy as np


class BenchmarkLogger:
    """
    Comprehensive logging system for LAP benchmarking experiments.
    
    Features:
    - Structured experiment logging
    - Multiple output formats (CSV, JSON, text)
    - Automatic timestamping and environment tracking
    - Performance metrics aggregation
    - Easy result retrieval and analysis
    """
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = None):
        """
        Initialize benchmark logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name for this experiment session
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.log_dir / "experiments").mkdir(exist_ok=True)
        (self.log_dir / "performance").mkdir(exist_ok=True)
        (self.log_dir / "detailed").mkdir(exist_ok=True)
        (self.log_dir / "summaries").mkdir(exist_ok=True)
        
        # Generate experiment ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            self.experiment_id = f"{experiment_name}_{timestamp}"
        else:
            self.experiment_id = f"exp_{timestamp}"
        
        # Log files
        self.csv_file = self.log_dir / "performance" / f"{self.experiment_id}.csv"
        self.json_file = self.log_dir / "experiments" / f"{self.experiment_id}.json"
        self.detail_file = self.log_dir / "detailed" / f"{self.experiment_id}.log"
        
        # Initialize experiment metadata
        self.metadata = {
            "experiment_id": self.experiment_id,
            "start_time": datetime.datetime.now().isoformat(),
            "environment": self._get_environment_info(),
            "results": []
        }
        
        # Initialize CSV file
        self._init_csv_file()
        
        # Log experiment start
        self._log_detail(f"Experiment {self.experiment_id} started")
        self._log_detail(f"Environment: {self.metadata['environment']}")
    
    def _get_environment_info(self) -> Dict[str, str]:
        """Collect environment information for reproducibility."""
        env_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "hostname": platform.node(),
        }
        
        # Thread settings
        thread_vars = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", 
                      "NUMEXPR_NUM_THREADS", "MKL_DYNAMIC", "PYTHONHASHSEED"]
        for var in thread_vars:
            env_info[var] = os.environ.get(var, "not_set")
        
        # Package versions
        try:
            import numpy
            env_info["numpy_version"] = numpy.__version__
        except ImportError:
            env_info["numpy_version"] = "not_available"
            
        try:
            import scipy
            env_info["scipy_version"] = scipy.__version__
        except ImportError:
            env_info["scipy_version"] = "not_available"
            
        try:
            import lap
            env_info["lap_version"] = "available"  # lap doesn't have __version__
        except ImportError:
            env_info["lap_version"] = "not_available"
        
        return env_info
    
    def _init_csv_file(self):
        """Initialize CSV file with headers."""
        headers = [
            "timestamp", "experiment_id", "dataset", "problem_size", 
            "problem_type", "noise_level", "solver_name", "time_ms", 
            "cost", "status", "notes"
        ]
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def _log_detail(self, message: str):
        """Write detailed log message."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.detail_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def log_result(self, 
                   dataset: str,
                   problem_size: int,
                   problem_type: str,
                   solver_name: str,
                   time_seconds: float,
                   cost: float,
                   noise_level: float = 0.0,
                   status: str = "success",
                   notes: str = "",
                   extra_data: Optional[Dict[str, Any]] = None):
        """
        Log a benchmark result.
        
        Args:
            dataset: Name of the dataset/problem
            problem_size: Size of the problem (n for nxn matrix)
            problem_type: Type of problem (uniform, sparse, etc.)
            solver_name: Name of the solver used
            time_seconds: Execution time in seconds
            cost: Optimal cost found
            noise_level: Noise level for seeded solvers
            status: success/failure/error
            notes: Additional notes
            extra_data: Additional structured data
        """
        timestamp = datetime.datetime.now().isoformat()
        
        # CSV logging
        csv_row = [
            timestamp, self.experiment_id, dataset, problem_size,
            problem_type, noise_level, solver_name, time_seconds * 1000,
            cost, status, notes
        ]
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_row)
        
        # Detailed logging
        self._log_detail(
            f"{solver_name} on {dataset} ({problem_size}x{problem_size}): "
            f"{time_seconds*1000:.2f}ms, cost={cost:.6f}, status={status}"
        )
        
        # JSON logging
        result_data = {
            "timestamp": timestamp,
            "dataset": dataset,
            "problem_size": problem_size,
            "problem_type": problem_type,
            "solver_name": solver_name,
            "time_seconds": time_seconds,
            "time_ms": time_seconds * 1000,
            "cost": cost,
            "noise_level": noise_level,
            "status": status,
            "notes": notes
        }
        
        if extra_data:
            result_data["extra_data"] = extra_data
        
        self.metadata["results"].append(result_data)
    
    def log_comparison(self,
                      dataset: str,
                      problem_size: int,
                      problem_type: str,
                      results: Dict[str, Dict[str, Any]],
                      noise_level: float = 0.0):
        """
        Log comparison results for multiple solvers.
        
        Args:
            dataset: Name of the dataset
            problem_size: Size of the problem
            problem_type: Type of problem
            results: Dict mapping solver_name -> {time, cost, status}
            noise_level: Noise level for seeded solvers
        """
        self._log_detail(f"\n=== Comparison: {dataset} ({problem_size}x{problem_size}) ===")
        
        # Log individual results
        for solver_name, result in results.items():
            self.log_result(
                dataset=dataset,
                problem_size=problem_size,
                problem_type=problem_type,
                solver_name=solver_name,
                time_seconds=result.get('time', 0.0),
                cost=result.get('cost', 0.0),
                noise_level=noise_level,
                status=result.get('status', 'unknown')
            )
        
        # Calculate and log speedups if applicable
        if 'scipy' in results and 'seeded' in results:
            if results['scipy']['status'] == 'success' and results['seeded']['status'] == 'success':
                speedup = results['scipy']['time'] / results['seeded']['time']
                self._log_detail(f"Speedup (Seeded vs SciPy): {speedup:.2f}x")
        
        if 'lap' in results and 'seeded' in results:
            if results['lap']['status'] == 'success' and results['seeded']['status'] == 'success':
                speedup = results['lap']['time'] / results['seeded']['time']
                self._log_detail(f"Speedup (Seeded vs LAP): {speedup:.2f}x")
    
    def save_experiment(self):
        """Save complete experiment data to JSON."""
        self.metadata["end_time"] = datetime.datetime.now().isoformat()
        
        with open(self.json_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        self._log_detail(f"Experiment {self.experiment_id} completed")
        self._log_detail(f"Results saved to {self.json_file}")
    
    def generate_summary(self, output_file: Optional[str] = None) -> str:
        """
        Generate a human-readable summary of the experiment.
        
        Args:
            output_file: Optional file to save summary to
            
        Returns:
            Summary text
        """
        if not self.metadata["results"]:
            return "No results to summarize."
        
        summary_lines = []
        summary_lines.append(f"Experiment Summary: {self.experiment_id}")
        summary_lines.append("=" * 60)
        summary_lines.append(f"Start Time: {self.metadata['start_time']}")
        summary_lines.append(f"Total Results: {len(self.metadata['results'])}")
        
        # Group by solver
        solver_stats = {}
        for result in self.metadata["results"]:
            solver = result["solver_name"]
            if solver not in solver_stats:
                solver_stats[solver] = {"times": [], "costs": [], "count": 0}
            
            if result["status"] == "success":
                solver_stats[solver]["times"].append(result["time_ms"])
                solver_stats[solver]["costs"].append(result["cost"])
            solver_stats[solver]["count"] += 1
        
        summary_lines.append("\nSolver Performance Summary:")
        summary_lines.append("-" * 40)
        
        for solver, stats in solver_stats.items():
            if stats["times"]:
                avg_time = np.mean(stats["times"])
                med_time = np.median(stats["times"])
                summary_lines.append(
                    f"{solver:12}: {stats['count']:3} runs, "
                    f"avg={avg_time:6.2f}ms, med={med_time:6.2f}ms"
                )
            else:
                summary_lines.append(f"{solver:12}: {stats['count']:3} runs, all failed")
        
        # Problem type analysis
        problem_types = {}
        for result in self.metadata["results"]:
            ptype = result["problem_type"]
            if ptype not in problem_types:
                problem_types[ptype] = 0
            problem_types[ptype] += 1
        
        summary_lines.append("\nProblem Types Tested:")
        summary_lines.append("-" * 25)
        for ptype, count in problem_types.items():
            summary_lines.append(f"{ptype:15}: {count:3} tests")
        
        summary_text = "\n".join(summary_lines)
        
        # Save to file if requested
        if output_file:
            summary_file = self.log_dir / "summaries" / output_file
        else:
            summary_file = self.log_dir / "summaries" / f"{self.experiment_id}_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write(summary_text)
        
        self._log_detail(f"Summary saved to {summary_file}")
        
        return summary_text


def get_latest_experiment(log_dir: str = "logs") -> Optional[str]:
    """Get the most recent experiment ID from logs."""
    log_path = Path(log_dir) / "experiments"
    if not log_path.exists():
        return None
    
    json_files = list(log_path.glob("*.json"))
    if not json_files:
        return None
    
    # Sort by modification time
    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    return latest_file.stem


def load_experiment(experiment_id: str, log_dir: str = "logs") -> Optional[Dict[str, Any]]:
    """Load experiment data from JSON file."""
    json_file = Path(log_dir) / "experiments" / f"{experiment_id}.json"
    
    if not json_file.exists():
        return None
    
    with open(json_file, 'r') as f:
        return json.load(f)


def list_experiments(log_dir: str = "logs") -> List[str]:
    """List all available experiment IDs."""
    log_path = Path(log_dir) / "experiments"
    if not log_path.exists():
        return []
    
    json_files = list(log_path.glob("*.json"))
    return [f.stem for f in sorted(json_files, key=lambda f: f.stat().st_mtime, reverse=True)]