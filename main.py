"""
vision2time: Cross-Modal Time Series Classification Benchmark
Main entry point for running benchmarks
"""

import argparse
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.utils.benchmark import BenchmarkRunner


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="vision2time: Cross-Modal Time Series Classification Benchmark"
    )

    parser.add_argument(
        "--mode",
        choices=["benchmark", "analyze"],
        default="benchmark",
        help="Operation mode",
    )

    parser.add_argument(
        "--data-dir", default="./data/raw/Multivariate_ts", help="Data directory path"
    )

    parser.add_argument(
        "--results-dir", default="./data/results", help="Results directory path"
    )

    parser.add_argument("--datasets", nargs="+", help="Specific datasets to benchmark")

    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs"
    )

    parser.add_argument(
        "--output", default="benchmark_results.csv", help="Output results filename"
    )

    args = parser.parse_args()

    if args.mode == "benchmark":
        print("vision2time Benchmark")
        print("=" * 50)

        # Check if data directory exists
        if not Path(args.data_dir).exists():
            print(f"Data directory not found: {args.data_dir}")
            return

        # Initialize benchmark runner
        runner = BenchmarkRunner(args.data_dir, args.results_dir)

        # Run benchmark
        results_file = runner.run_benchmark(
            datasets=args.datasets, epochs=args.epochs, save_file=args.output
        )

        if results_file:
            print(f"Benchmark completed! Results: {results_file}")


if __name__ == "__main__":
    main()
