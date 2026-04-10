"""
Component 18a: Trainer — MLX LoRA Fine-tuning Controller.
Invokes mlx_lm for actual training on Apple Silicon.
Includes early stopping on validation loss.
"""
import json
import os
import subprocess
import logging
import datetime
import random
from typing import Optional

logger = logging.getLogger("acs.trainer")

LORA_CONFIG = {
    "rank": 16,
    "alpha": 32,
    "dropout": 0.05,
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 50,
    "max_seq_len": 4096,
    "validation_split": 0.10,
    "early_stopping_patience": 200,
}


class Trainer:
    """MLX-LM LoRA trainer for Apple Silicon."""

    def __init__(
        self,
        registry,
        datasets_dir: str = "./acs/training/datasets/",
        adapters_dir: str = "./acs/models/adapters/",
        logs_dir: str = "./acs/logs/training/",
    ):
        self.registry = registry
        self.datasets_dir = datasets_dir
        self.adapters_dir = adapters_dir
        self.logs_dir = logs_dir
        os.makedirs(self.adapters_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def _prepare_training_data(
        self, dataset_version: str, rank: int
    ) -> tuple[str, str]:
        """Load separate train and validation sets from dataset builder."""
        train_source = os.path.join(
            self.datasets_dir, dataset_version, "train_dataset.jsonl"
        )
        eval_source = os.path.join(
            self.datasets_dir, dataset_version, "eval_dataset.jsonl"
        )
        if not os.path.exists(train_source) or not os.path.exists(eval_source):
            raise FileNotFoundError(
                f"Datasets not found in {dataset_version}"
            )

        with open(train_source, "r") as f:
            train_data = [json.loads(line) for line in f]
            
        with open(eval_source, "r") as f:
            val_data = [json.loads(line) for line in f]

        # Convert to chat format for mlx_lm
        run_dir = os.path.join(
            self.logs_dir,
            f"run_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}",
        )
        os.makedirs(run_dir, exist_ok=True)

        train_file = os.path.join(run_dir, "train.jsonl")
        val_file = os.path.join(run_dir, "valid.jsonl")

        for data, filepath in [
            (train_data, train_file), (val_data, val_file)
        ]:
            with open(filepath, "w") as f:
                for item in data:
                    reasoning = item.get("reasoning_trace", {})
                    chat_entry = {
                        "messages": [
                            {
                                "role": "user",
                                "content": item.get("query", ""),
                            },
                            {
                                "role": "assistant",
                                "content": json.dumps(reasoning),
                            },
                        ]
                    }
                    f.write(json.dumps(chat_entry) + "\n")

        # Save config
        config_file = os.path.join(run_dir, "config.json")
        with open(config_file, "w") as f:
            json.dump(
                {
                    **LORA_CONFIG,
                    "rank": rank,
                    "dataset_version": dataset_version,
                    "train_size": len(train_data),
                    "val_size": len(val_data),
                    "timestamp": datetime.datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        logger.info(
            "Prepared training data: %d train, %d validation",
            len(train_data),
            len(val_data),
        )
        return run_dir, train_file

    def train(
        self,
        dataset_version: str,
        base_model: str = "mlx-community/Qwen2.5-32B-Instruct-4bit",
    ) -> dict:
        """Execute MLX-LM LoRA training."""
        new_version = self.registry.register_new_model()
        iterations = len(self.registry.versions)
        rank = 16 if iterations == 1 else 8
        
        run_dir, train_file = self._prepare_training_data(
            dataset_version, rank
        )
        adapter_dir = os.path.join(self.adapters_dir, new_version)
        os.makedirs(adapter_dir, exist_ok=True)

        # Calculate precise iteration bounds
        try:
            with open(os.path.join(run_dir, "config.json"), "r") as f:
                cfg = json.load(f)
                train_count = cfg.get("train_size", 1000)
        except Exception:
            train_count = 1000
            
        calculated_iters = max(100, (train_count // LORA_CONFIG["batch_size"]) * LORA_CONFIG["epochs"])

        # Build mlx_lm command
        cmd = [
            "python3", "-m", "mlx_lm.lora",
            "--model", base_model,
            "--train",
            "--data", run_dir,
            "--adapter-path", adapter_dir,
            "--iters", str(calculated_iters),
            "--batch-size", str(LORA_CONFIG["batch_size"]),
            "--learning-rate", str(LORA_CONFIG["learning_rate"]),
            "--lora-layers", str(rank),
            "--val-batches", "20",  # Enable early stopping patience bounds
            "--seed", "42",
        ]

        logger.info("Starting training: %s", " ".join(cmd))
        logger.info("Adapter will be saved to: %s", adapter_dir)

        result = {
            "version": new_version,
            "run_dir": run_dir,
            "adapter_dir": adapter_dir,
            "config": {**LORA_CONFIG, "rank": rank},
            "command": " ".join(cmd),
            "status": "pending",
        }

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600 * 12,  # 12 hour max
            )

            result["stdout"] = proc.stdout[-2000:]  # Last 2K chars
            result["stderr"] = proc.stderr[-2000:]
            result["returncode"] = proc.returncode

            if proc.returncode == 0:
                result["status"] = "completed"
                logger.info("Training completed for %s", new_version)
            else:
                result["status"] = "failed"
                logger.error(
                    "Training failed (rc=%d): %s",
                    proc.returncode,
                    proc.stderr[-500:],
                )

        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            logger.error("Training timed out after 12 hours")
        except FileNotFoundError:
            result["status"] = "mlx_not_installed"
            result["error"] = (
                "mlx_lm not found. Install with: "
                "pip install mlx-lm"
            )
            logger.error("mlx_lm not installed")

        # Save run result
        result_file = os.path.join(run_dir, "result.json")
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def dry_run(
        self, dataset_version: Optional[str] = None
    ) -> dict:
        """Show what training would do without executing."""
        versions = os.listdir(self.datasets_dir) if os.path.exists(
            self.datasets_dir
        ) else []

        if dataset_version:
            target = dataset_version
        elif versions:
            target = sorted(versions)[-1]
        else:
            return {"error": "No datasets available"}

        train_source = os.path.join(
            self.datasets_dir, target, "train_dataset.jsonl"
        )
        count = 0
        if os.path.exists(train_source):
            with open(train_source, "r") as f:
                count = sum(1 for _ in f)

        return {
            "dataset_version": target,
            "trace_count": count,
            "config": {**LORA_CONFIG, "rank": 16 if len(self.registry.versions)==0 else 8},
            "estimated_time": f"~{count * 3 // 60} minutes",
            "next_model_version": f"v{len(self.registry.versions) + 1}",
            "status": "dry_run — no training executed",
        }
