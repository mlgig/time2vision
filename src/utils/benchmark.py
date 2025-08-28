"""
Benchmark utilities and runner
"""

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add current directory to path for relative imports
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))


from data.loader import MTSCDataLoader
from models.cnn import CNN1D
from models.lstm import BiLSTM
from models.transformer import TransformerClassifier
from models.vit import ViTAdapter
from training.trainer import Trainer


class BenchmarkRunner:
    """Main benchmark runner class"""

    def __init__(
        self, data_dir="./data/raw/Multivariate_ts", results_dir="./data/results"
    ):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)

        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_loader = MTSCDataLoader(data_dir)
        self.trainer = Trainer()

        print(
            f"Found {len(self.data_loader.datasets)} datasets: {self.data_loader.datasets}"
        )

    def get_models(self, n_channels, n_classes):
        """Get all models for benchmarking"""
        models = []

        # Baseline models
        models.extend(
            [
                ("CNN-1D", CNN1D(n_channels, n_classes), "all_channels"),
                ("BiLSTM", BiLSTM(n_channels, n_classes), "all_channels"),
                (
                    "Transformer",
                    TransformerClassifier(n_channels, n_classes),
                    "all_channels",
                ),
            ]
        )

        # ViT models with different channel strategies
        for strategy in ["first", "random", "pca"]:
            models.append(
                ("ViT", ViTAdapter(n_classes, channel_strategy=strategy), strategy)
            )

        return models

    def run_benchmark(
        self, datasets=None, epochs=15, save_file="benchmark_results.csv"
    ):
        """Run complete benchmark"""

        if datasets is None:
            datasets = self.data_loader.datasets

        if not datasets:
            print("No datasets found!")
            return None

        results_file = self.results_dir / save_file

        # Initialize results file
        with open(results_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "dataset",
                    "model",
                    "channel_strategy",
                    "accuracy",
                    "f1_score",
                    "training_time",
                    "n_channels",
                    "seq_length",
                    "n_classes",
                    "n_train",
                    "n_test",
                    "timestamp",
                ]
            )

        print(f"Starting benchmark on {len(datasets)} datasets...")

        total_results = []

        for dataset_name in datasets:
            print(f"Processing: {dataset_name}")

            try:
                # Load dataset
                X_train, y_train, X_test, y_test = self.data_loader.load_dataset(
                    dataset_name
                )
                info = self.data_loader.get_dataset_info(dataset_name)

                print(f"Shape: {X_train.shape}, Classes: {info['n_classes']}")

                # Skip if insufficient data
                if info["n_classes"] <= 1 or info["n_train"] < 10 or info["n_test"] < 5:
                    print("Skipping - insufficient data")
                    continue

                # Get models for this dataset
                models = self.get_models(info["n_channels"], info["n_classes"])

                # Train and evaluate each model
                for model_name, model, strategy in models:
                    print(f"Training {model_name} ({strategy})...", end=" ")

                    try:
                        start_time = time.time()
                        accuracy, f1 = self.trainer.train_and_evaluate(
                            model, X_train, y_train, X_test, y_test, epochs=epochs
                        )
                        training_time = time.time() - start_time

                        print(f"Acc: {accuracy:.4f}, F1: {f1:.4f}")

                        # Save result
                        result = [
                            dataset_name,
                            model_name,
                            strategy,
                            accuracy,
                            f1,
                            training_time,
                            info["n_channels"],
                            info["seq_length"],
                            info["n_classes"],
                            info["n_train"],
                            info["n_test"],
                            datetime.now().isoformat(),
                        ]

                        with open(results_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(result)

                        total_results.append(
                            {
                                "dataset": dataset_name,
                                "model": model_name,
                                "strategy": strategy,
                                "accuracy": accuracy,
                                "f1_score": f1,
                            }
                        )

                    except Exception as e:
                        print(f"Failed: {str(e)[:30]}")

                        # Save failed result
                        result = [
                            dataset_name,
                            model_name,
                            strategy,
                            0.0,
                            0.0,
                            0.0,
                            info["n_channels"],
                            info["seq_length"],
                            info["n_classes"],
                            info["n_train"],
                            info["n_test"],
                            datetime.now().isoformat(),
                        ]

                        with open(results_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(result)

            except Exception as e:
                print(f"Dataset error: {str(e)[:50]}")
                continue

        print(f"Benchmark completed! Results saved to: {results_file}")

        return str(results_file)
