#!/usr/bin/env python3
"""Progressive training orchestrator for bucket-aware training within 8h time limits."""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np


@dataclass
class TrainingStage:
    """Configuration for a single training stage."""
    name: str
    bucket_name: str
    filter_size: Optional[int]
    batch_size: int
    epochs: int
    train_path: Path
    val_path: Path
    checkpoint_init: Optional[Path] = None
    output_path: Optional[Path] = None


class ProgressiveTrainer:
    """Orchestrates progressive training across different matrix size buckets."""
    
    def __init__(self, base_dir: Path, checkpoint_dir: Path):
        self.base_dir = Path(base_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.model_config = {
            "hidden": 64,
            "layers": 3,
            "heads": 4,
            "dropout": 0.1,
            "lr": 5e-4,  # Reduced from 1e-3 to avoid NaN issues
        }
    
    def get_dataset_info(self, dataset_path: Path) -> Dict:
        """Get information about dataset size distribution."""
        if not dataset_path.exists():
            return {"total": 0, "size_distribution": {}}
            
        with h5py.File(dataset_path, "r") as f:
            sizes = f["n"][:]
            unique_sizes, counts = np.unique(sizes, return_counts=True)
            
        size_dist = {int(size): int(count) for size, count in zip(unique_sizes, counts)}
        return {
            "total": len(sizes),
            "size_distribution": size_dist
        }
    
    def create_training_plan(self) -> List[TrainingStage]:
        """Create the progressive training plan."""
        stages = []
        
        # Stage 1: Small bucket (512x512) - already done as debug
        small_train = self.base_dir / "data/generated/processed/small/full/train.h5"
        small_val = self.base_dir / "data/generated/processed/small/full/val.h5"
        
        if small_train.exists():
            info = self.get_dataset_info(small_train)
            print(f"Small dataset: {info}")
            
            # Train on 512-only subset
            stages.append(TrainingStage(
                name="small_512_only",
                bucket_name="small_512",
                filter_size=512,
                batch_size=8,
                epochs=30,
                train_path=small_train,
                val_path=small_val,
                output_path=self.checkpoint_dir / "small_512.pt"
            ))
            
            # Train on 1024 subset with smaller batch
            if 1024 in info.get("size_distribution", {}):
                stages.append(TrainingStage(
                    name="small_1024_only",
                    bucket_name="small_1024", 
                    filter_size=1024,
                    batch_size=2,
                    epochs=20,
                    train_path=small_train,
                    val_path=small_val,
                    checkpoint_init=self.checkpoint_dir / "small_512.pt",
                    output_path=self.checkpoint_dir / "small_1024.pt"
                ))
        
        # Stage 2: Mid buckets
        for bucket_size in [1536, 2048, 3072]:
            bucket_dir = self.base_dir / f"data/generated/processed/mid_{bucket_size}/full"
            train_path = bucket_dir / "train.h5"
            val_path = bucket_dir / "val.h5"
            
            if train_path.exists():
                info = self.get_dataset_info(train_path)
                print(f"Mid {bucket_size} dataset: {info}")
                
                # Determine appropriate batch size based on memory constraints
                if bucket_size <= 1536:
                    batch_size = 4
                elif bucket_size <= 2048:
                    batch_size = 2
                else:
                    batch_size = 1
                
                # Use previous checkpoint as initialization
                if stages:
                    init_checkpoint = stages[-1].output_path
                else:
                    init_checkpoint = None
                
                stages.append(TrainingStage(
                    name=f"mid_{bucket_size}",
                    bucket_name=f"mid_{bucket_size}",
                    filter_size=None,  # Use all sizes in this bucket
                    batch_size=batch_size,
                    epochs=25,
                    train_path=train_path,
                    val_path=val_path,
                    checkpoint_init=init_checkpoint,
                    output_path=self.checkpoint_dir / f"mid_{bucket_size}.pt"
                ))
        
        # Stage 3: Large bucket (if available)
        large_dir = self.base_dir / "data/generated/processed/large_4096/full"
        large_train = large_dir / "train.h5"
        large_val = large_dir / "val.h5"
        
        if large_train.exists():
            info = self.get_dataset_info(large_train)
            print(f"Large dataset: {info}")
            
            # Use previous checkpoint as initialization
            if stages:
                init_checkpoint = stages[-1].output_path
            else:
                init_checkpoint = None
                
            stages.append(TrainingStage(
                name="large_4096",
                bucket_name="large_4096",
                filter_size=None,
                batch_size=1,
                epochs=20,
                train_path=large_train,
                val_path=large_val,
                checkpoint_init=init_checkpoint,
                output_path=self.checkpoint_dir / "large_4096.pt"
            ))
        
        return stages
    
    def run_training_stage(self, stage: TrainingStage, dry_run: bool = False) -> bool:
        """Run a single training stage."""
        print(f"\\n{'='*60}")
        print(f"TRAINING STAGE: {stage.name}")
        print(f"{'='*60}")
        print(f"Dataset: {stage.train_path}")
        print(f"Filter size: {stage.filter_size}")
        print(f"Batch size: {stage.batch_size}")
        print(f"Epochs: {stage.epochs}")
        print(f"Init checkpoint: {stage.checkpoint_init}")
        print(f"Output: {stage.output_path}")
        
        # Build command
        cmd = [
            sys.executable, "gnn/train_progressive.py",
            "--train", str(stage.train_path),
            "--val", str(stage.val_path),
            "--epochs", str(stage.epochs),
            "--batch-size", str(stage.batch_size),
            "--hidden", str(self.model_config["hidden"]),
            "--layers", str(self.model_config["layers"]),
            "--heads", str(self.model_config["heads"]),
            "--dropout", str(self.model_config["dropout"]),
            "--lr", str(self.model_config["lr"]),
            "--device", "cuda",
            "--output", str(stage.output_path),
            "--checkpoint-save", str(self.checkpoint_dir / f"{stage.name}_checkpoint.pt"),
            "--resume",  # Enable resumption
        ]
        
        if stage.filter_size:
            cmd.extend(["--filter-size", str(stage.filter_size)])
            
        if stage.checkpoint_init and stage.checkpoint_init.exists():
            cmd.extend(["--checkpoint-init", str(stage.checkpoint_init)])
        
        print(f"Command: {' '.join(cmd)}")
        
        if dry_run:
            print("DRY RUN - Not executing")
            return True
        
        # Execute training
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                check=True,
                capture_output=True,
                text=True
            )
            
            duration = time.time() - start_time
            print(f"Stage completed successfully in {duration/3600:.2f} hours")
            print(f"stdout: {result.stdout[-1000:]}")  # Last 1000 chars
            
            return True
            
        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            print(f"Stage failed after {duration/3600:.2f} hours")
            print(f"Return code: {e.returncode}")
            print(f"stdout: {e.stdout[-1000:]}")
            print(f"stderr: {e.stderr[-1000:]}")
            
            return False
    
    def run_progressive_training(self, stages: Optional[List[str]] = None, dry_run: bool = False) -> None:
        """Run the full progressive training pipeline."""
        training_plan = self.create_training_plan()
        
        if stages:
            # Filter to specific stages
            training_plan = [stage for stage in training_plan if stage.name in stages]
        
        if not training_plan:
            print("No training stages found!")
            return
        
        print(f"\\nProgressive Training Plan:")
        print(f"{'='*60}")
        for i, stage in enumerate(training_plan, 1):
            print(f"{i}. {stage.name} - {stage.bucket_name} (batch={stage.batch_size})")
        print()
        
        total_start = time.time()
        successful_stages = 0
        
        for i, stage in enumerate(training_plan, 1):
            print(f"\\nStarting stage {i}/{len(training_plan)}: {stage.name}")
            
            success = self.run_training_stage(stage, dry_run=dry_run)
            
            if success:
                successful_stages += 1
                print(f"✓ Stage {stage.name} completed successfully")
            else:
                print(f"✗ Stage {stage.name} failed")
                if not dry_run:
                    print("Stopping progressive training due to failure")
                    break
        
        total_duration = time.time() - total_start
        print(f"\\n{'='*60}")
        print(f"PROGRESSIVE TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Completed {successful_stages}/{len(training_plan)} stages")
        print(f"Total time: {total_duration/3600:.2f} hours")
        
        if successful_stages == len(training_plan):
            final_model = training_plan[-1].output_path
            print(f"✓ All stages completed! Final model: {final_model}")
        else:
            print("⚠ Training incomplete - some stages failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."),
                       help="Base directory of the project")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("gnn/checkpoints"),
                       help="Directory to store checkpoints")
    parser.add_argument("--stages", nargs="+", 
                       help="Specific stages to run (e.g., small_512_only mid_1536)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without executing")
    parser.add_argument("--list", action="store_true",
                       help="List available training stages")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    trainer = ProgressiveTrainer(args.base_dir, args.checkpoint_dir)
    
    if args.list:
        stages = trainer.create_training_plan()
        print("Available training stages:")
        for i, stage in enumerate(stages, 1):
            print(f"{i}. {stage.name} - {stage.bucket_name}")
            print(f"   Dataset: {stage.train_path}")
            print(f"   Filter: {stage.filter_size}, Batch: {stage.batch_size}")
            print(f"   Epochs: {stage.epochs}")
            if stage.checkpoint_init:
                print(f"   Init: {stage.checkpoint_init}")
            print()
        return
    
    trainer.run_progressive_training(
        stages=args.stages,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()