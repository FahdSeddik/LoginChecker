#!/usr/bin/env python3

from data.reader import UsernameReader


def run_billion_benchmark():
    """Run benchmarks from 100 to 1 billion usernames with smart sizing and progress"""

    reader = UsernameReader("data/usernames.dat")
    print(f"Total usernames available: {reader.total_count:,}")

    # Smart exponential sizes that scale better
    all_sizes = [
        # Small scale (100 - 100K) - for all algorithms
        100,
        500,
        1000,
        5000,
        10000,
        50000,
        100000,
        # Medium scale (500K - 10M) - for filters + binary search
        500000,
        1000000,
        5000000,
        10000000,
        # Large scale (50M - 1B) - for filters only
        50000000,
        100000000,
        500000000,
        1000000000,
    ]

    # Filter to only available sizes
    available_sizes = [s for s in all_sizes if s <= reader.total_count]

    # Define size limits for each algorithm
    size_limits = {
        "linear_search": 100_000,  # Up to 100K only (too slow beyond)
        "binary_search": 100_000_000,  # Up to 100M (sorting becomes expensive)
        "bloom_filter": 1_000_000_000,  # Up to 1B (very scalable)
        "cuckoo_filter": 10_000_000,  # Up to 10M (scalable but needs more memory)
        "disk_hashset": 100_000_000,  # Up to 500M (very scalable with memory mapping)
    }

    print(f"\\nBenchmark sizes: {len(available_sizes)} points")
    print("Sizes:", ", ".join(f"{s:,}" for s in available_sizes))
    print("Size limits:", size_limits)

    # # 1. Bloom Filter - billion scale
    # bloom_sizes = [s for s in available_sizes if s <= size_limits["bloom_filter"]]
    # if bloom_sizes:
    #     print("\\n" + "=" * 60)
    #     print("BLOOM FILTER BENCHMARK (All Sizes)")
    #     print("=" * 60)

    #     bloom_config = BenchmarkConfig(
    #         x_names=["N"],
    #         x_vals=bloom_sizes,
    #         line_arg="provider",
    #         line_vals=["bloom_filter"],
    #         line_names=["Bloom Filter"],
    #         styles=[("green", "-")],
    #         ylabel="Search Time (ms)",
    #         plot_name="bloom-filter-billion-scale",
    #         args={"false_positive_rate": 0.01},
    #         measure_memory=True,
    #         measure_setup_time=True,
    #         warmup_runs=1,
    #         measure_runs=2,  # Faster for large scales
    #     )

    #     bloom_benchmark = create_search_benchmark(
    #         reader, bloom_config, test_existing=True
    #     )
    #     bloom_benchmark.run(show_plots=True, print_data=True)

    # # 2. Cuckoo Filter - billion scale
    # cuckoo_sizes = [s for s in available_sizes if s <= size_limits["cuckoo_filter"]]
    # if cuckoo_sizes:
    #     print("\\n" + "=" * 60)
    #     print("CUCKOO FILTER BENCHMARK (All Sizes)")
    #     print("=" * 60)

    #     cuckoo_config = BenchmarkConfig(
    #         x_names=["N"],
    #         x_vals=cuckoo_sizes,
    #         line_arg="provider",
    #         line_vals=["cuckoo_filter"],
    #         line_names=["Cuckoo Filter"],
    #         styles=[("red", "-")],
    #         ylabel="Search Time (ms)",
    #         plot_name="cuckoo-filter-billion-scale",
    #         args={"false_positive_rate": 0.01, "bucket_size": 4},
    #         measure_memory=True,
    #         measure_setup_time=True,
    #         warmup_runs=5,
    #         measure_runs=10,
    #     )

    #     cuckoo_benchmark = create_search_benchmark(
    #         reader, cuckoo_config, test_existing=True
    #     )
    #     cuckoo_benchmark.run(show_plots=True, print_data=True)

    # # 3. Binary Search - medium scale
    # binary_sizes = [s for s in available_sizes if s <= size_limits["binary_search"]]
    # if binary_sizes:
    #     print("\\n" + "=" * 60)
    #     print("BINARY SEARCH BENCHMARK (Medium Sizes)")
    #     print("=" * 60)
    #     print(f"Sizes: {', '.join(f'{s:,}' for s in binary_sizes)}")

    #     binary_config = BenchmarkConfig(
    #         x_names=["N"],
    #         x_vals=binary_sizes,
    #         line_arg="provider",
    #         line_vals=["binary_search"],
    #         line_names=["Binary Search"],
    #         styles=[("purple", "-")],
    #         ylabel="Search Time (ms)",
    #         plot_name="binary-search-medium-scale",
    #         args={},
    #         measure_memory=True,
    #         measure_setup_time=True,
    #         warmup_runs=5,
    #         measure_runs=10,
    #     )

    #     binary_benchmark = create_search_benchmark(
    #         reader, binary_config, test_existing=True
    #     )
    #     binary_benchmark.run(show_plots=True, print_data=True)

    # # 4. Linear Search - small scale only
    # linear_sizes = [s for s in available_sizes if s <= size_limits["linear_search"]]
    # if linear_sizes:
    #     print("\\n" + "=" * 60)
    #     print("LINEAR SEARCH BENCHMARK (Small Sizes Only)")
    #     print("=" * 60)
    #     print(f"Sizes: {', '.join(f'{s:,}' for s in linear_sizes)}")

    #     linear_config = BenchmarkConfig(
    #         x_names=["N"],
    #         x_vals=linear_sizes,
    #         line_arg="provider",
    #         line_vals=["linear_search"],
    #         line_names=["Linear Search"],
    #         styles=[("blue", "-")],
    #         ylabel="Search Time (ms)",
    #         plot_name="linear-search-small-scale",
    #         args={},
    #         measure_memory=True,
    #         measure_setup_time=True,
    #         warmup_runs=5,
    #         measure_runs=10,
    #     )

    #     linear_benchmark = create_search_benchmark(
    #         reader, linear_config, test_existing=True
    #     )
    #     linear_benchmark.run(show_plots=True, print_data=True)

    # # 5. Disk HashSet - billion scale
    # disk_hashset_sizes = [
    #     s for s in available_sizes if s <= size_limits["disk_hashset"]
    # ]
    # if disk_hashset_sizes:
    #     print("\n" + "=" * 60)
    #     print("DISK HASHSET BENCHMARK (All Sizes)")
    #     print("=" * 60)

    #     disk_hashset_config = BenchmarkConfig(
    #         x_names=["N"],
    #         x_vals=disk_hashset_sizes,
    #         line_arg="provider",
    #         line_vals=["disk_hashset"],
    #         line_names=["Disk HashSet"],
    #         styles=[("orange", "-")],
    #         ylabel="Search Time (ms)",
    #         plot_name="disk-hashset-billion-scale",
    #         args={},
    #         measure_memory=True,
    #         measure_setup_time=True,
    #         warmup_runs=5,
    #         measure_runs=10,
    #     )

    #     disk_hashset_benchmark = create_search_benchmark(
    #         reader, disk_hashset_config, test_existing=True
    #     )
    #     disk_hashset_benchmark.run(show_plots=True, print_data=True)

    # # 6. Multi-algorithm comparison on overlapping sizes
    # comparison_sizes = [
    #     s for s in available_sizes if s <= 100_000
    # ]  # Where all algorithms work well
    # if comparison_sizes:
    #     print("\\n" + "=" * 60)
    #     print("ALGORITHM COMPARISON (Small to Medium Sizes)")
    #     print("=" * 60)
    #     print(f"Sizes: {', '.join(f'{s:,}' for s in comparison_sizes)}")

    #     comparison_config = BenchmarkConfig(
    #         x_names=["N"],
    #         x_vals=comparison_sizes,
    #         line_arg="provider",
    #         line_vals=[
    #             "linear_search",
    #             "binary_search",
    #             "bloom_filter",
    #             "cuckoo_filter",
    #             "disk_hashset",
    #         ],
    #         line_names=[
    #             "Linear Search",
    #             "Binary Search",
    #             "Bloom Filter",
    #             "Cuckoo Filter",
    #             "Disk HashSet",
    #         ],
    #         styles=[
    #             ("blue", "-"),
    #             ("purple", "-"),
    #             ("green", "-"),
    #             ("red", "-"),
    #             ("orange", "-"),
    #         ],
    #         ylabel="Search Time (ms)",
    #         plot_name="algorithm-comparison",
    #         args={"false_positive_rate": 0.01, "bucket_size": 4},
    #         measure_memory=True,
    #         measure_setup_time=True,
    #         warmup_runs=5,
    #         measure_runs=10,
    #     )

    #     comparison_benchmark = create_search_benchmark(
    #         reader, comparison_config, test_existing=True
    #     )
    #     comparison_benchmark.run(show_plots=True, print_data=True)

    # print("\\n" + "=" * 80)
    # print(" BILLION-SCALE BENCHMARK COMPLETED!")
    # print("=" * 80)
    # print("Generated plots:")
    # # if bloom_sizes:
    # #     print("   bloom-filter-billion-scale.png")
    # if cuckoo_sizes:
    #     print("   cuckoo-filter-billion-scale.png")
    # if binary_sizes:
    #     print("   binary-search-medium-scale.png")
    # if linear_sizes:
    #     print("   linear-search-small-scale.png")
    # if "disk_hashset_sizes" in locals() and disk_hashset_sizes:
    #     print("   disk-hashset-billion-scale.png")
    # if comparison_sizes:
    #     print("   algorithm-comparison.png")


if __name__ == "__main__":
    try:
        run_billion_benchmark()
    except FileNotFoundError:
        print("ERROR: data/usernames.dat not found!")
        print("Generate it first: python data/generate.py")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
