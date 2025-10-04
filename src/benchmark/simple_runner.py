"""
Simple benchmark runner for search algorithms.

Usage examples:
    python -m src.benchmark.simple_runner
    python simple_runner.py --algorithms linear_search,binary_search --sizes 100,500,1000
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import after path setup
from data.reader import UsernameReader  # noqa: E402
from src.benchmark import BenchmarkConfig, create_search_benchmark  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Benchmark search algorithms")
    parser.add_argument(
        "--data-file", default="data/usernames.dat", help="Path to username data file"
    )
    parser.add_argument(
        "--algorithms",
        default="linear_search,bloom_filter",
        help="Comma-separated list of algorithms to test",
    )
    parser.add_argument(
        "--sizes",
        default="100,500,1000,5000,10000,50000,100000",
        help="Comma-separated list of dataset sizes",
    )
    parser.add_argument(
        "--test-existing",
        action="store_true",
        default=True,
        help="Test with existing usernames",
    )
    parser.add_argument(
        "--test-non-existing",
        action="store_true",
        help="Test with non-existing usernames",
    )
    parser.add_argument(
        "--output-prefix", default="benchmark", help="Prefix for output plot files"
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")

    args = parser.parse_args()

    # Parse algorithms and sizes
    algorithms = [alg.strip() for alg in args.algorithms.split(",")]
    sizes = [int(size.strip()) for size in args.sizes.split(",")]

    # Algorithm names mapping
    algorithm_names = {
        "linear_search": "Linear Search",
        "binary_search": "Binary Search",
        "bloom_filter": "Bloom Filter",
        "cuckoo_filter": "Cuckoo Filter",
    }

    # Colors for each algorithm
    colors = ["blue", "green", "red", "orange", "purple", "brown"]

    try:
        # Initialize reader
        reader = UsernameReader(args.data_file)
        print(f"Loaded {len(reader):,} usernames from {args.data_file}")

        # Run benchmark for existing usernames
        if args.test_existing:
            print("\nRunning benchmark for EXISTING usernames...")

            config = BenchmarkConfig(
                x_names=["N"],
                x_vals=sizes,
                line_arg="provider",
                line_vals=algorithms,
                line_names=[algorithm_names.get(alg, alg) for alg in algorithms],
                styles=[(colors[i % len(colors)], "-") for i in range(len(algorithms))],
                ylabel="Time (ms)",
                plot_name=f"{args.output_prefix}-existing",
                args={"false_positive_rate": 0.01},
                warmup_runs=5,
                measure_runs=10,
                min_runtime_ms=1000.0,
            )

            benchmark = create_search_benchmark(reader, config, test_existing=True)
            results = benchmark.run(show_plots=not args.no_plots, print_data=True)
            print("[OK] Existing username benchmark completed")

        # Run benchmark for non-existing usernames
        if args.test_non_existing:
            print("\nRunning benchmark for NON-EXISTING usernames...")

            config = BenchmarkConfig(
                x_names=["N"],
                x_vals=sizes,
                line_arg="provider",
                line_vals=algorithms,
                line_names=[algorithm_names.get(alg, alg) for alg in algorithms],
                styles=[
                    (colors[i % len(colors)], "--") for i in range(len(algorithms))
                ],
                ylabel="Time (ms)",
                plot_name=f"{args.output_prefix}-non-existing",
                args={"false_positive_rate": 0.01},
                warmup_runs=5,
                measure_runs=10,
                min_runtime_ms=5.0,
            )

            benchmark = create_search_benchmark(reader, config, test_existing=False)
            results = benchmark.run(show_plots=not args.no_plots, print_data=True)
            print("[OK] Non-existing username benchmark completed")

        print(f"\nBenchmark completed! Plots saved as {args.output_prefix}-*.png")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the data file exists. You may need to generate it first.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        if "reader" in locals():
            reader.close()


if __name__ == "__main__":
    main()
