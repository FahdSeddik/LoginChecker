import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run, similar to triton.testing.Benchmark"""

    x_names: List[str]
    x_vals: List[Union[int, float]]
    line_arg: str
    line_vals: List[str]
    line_names: List[str]
    styles: List[Tuple[str, str]]
    ylabel: str
    plot_name: str
    args: Dict[str, Any] = field(default_factory=dict)
    warmup_runs: int = 3
    measure_runs: int = 10
    min_runtime_ms: float = 10.0
    measure_memory: bool = True
    measure_setup_time: bool = True


@dataclass
class BenchmarkResult:
    """Result of a single benchmark measurement"""

    value: float
    std_dev: float
    measurements: List[float]
    config_name: str
    x_value: Union[int, float]
    setup_time: Optional[float] = None
    memory_usage: Optional[float] = None


class IncrementalDataStructureManager:
    """Manages data structures that need incremental building for large N values"""

    def __init__(self, reader):
        self.reader = reader
        self.data_structures: Dict[str, Optional[Dict[str, Any]]] = {}
        self.last_built_n: Dict[str, int] = {}

    def get_or_build_structure(
        self, structure_type: str, target_n: int, **kwargs
    ) -> Dict[str, Any]:
        """Get existing structure or build incrementally to target_n"""
        key = f"{structure_type}_{hash(str(sorted(kwargs.items())))}"

        # Initialize if first time
        if key not in self.data_structures:
            self.data_structures[key] = None
            self.last_built_n[key] = 0

        # Check if we need to build/extend
        last_n = self.last_built_n[key]
        if last_n < target_n:
            added_elements = target_n - last_n
            if added_elements > 1000:  # Only show progress for significant builds
                print(
                    f"Extending {structure_type} from N={last_n:,} to N={target_n:,} (+{added_elements:,})",
                    end="",
                    flush=True,
                )
            else:
                print(
                    f"Building {structure_type} for N={target_n:,}",
                    end="",
                    flush=True,
                )
            build_start = time.time()
            self.data_structures[key] = self._build_incremental(
                structure_type, self.data_structures[key], last_n, target_n, **kwargs
            )
            self.last_built_n[key] = target_n
            build_time = time.time() - build_start
            print(f" [{build_time:.1f}s]")

        # Set reader limit for current test
        self.reader.set_limit(target_n)

        result = self.data_structures[key] or {}
        return {**result, "reader": self.reader, "current_n": target_n}

    def _build_incremental(
        self,
        structure_type: str,
        existing_structure: Optional[Dict],
        from_n: int,
        to_n: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build data structure incrementally from from_n to to_n"""

        if structure_type == "bloom_filter":
            return self._build_bloom_filter_incremental(
                existing_structure, from_n, to_n, **kwargs
            )
        elif structure_type == "cuckoo_filter":
            return self._build_cuckoo_filter_incremental(
                existing_structure, from_n, to_n, **kwargs
            )
        elif structure_type == "sorted_list":
            return self._build_sorted_list_incremental(
                existing_structure, from_n, to_n, **kwargs
            )
        else:
            raise ValueError(f"Unknown structure type: {structure_type}")

    def _build_bloom_filter_incremental(
        self, existing: Optional[Dict], from_n: int, to_n: int, **kwargs
    ) -> Dict[str, Any]:
        """Build bloom filter incrementally with true incremental addition"""
        import os

        import psutil

        from ..data_structures.bloom_filter import BloomFilter

        false_positive_rate = kwargs.get("false_positive_rate", 0.01)

        # Measure setup time
        setup_start = time.perf_counter()

        if existing is None:
            # Create new bloom filter with large capacity to avoid rebuilds
            if to_n > 10_000_000:  # 10M+
                capacity = 2_000_000_000  # 2B capacity
            elif to_n > 1_000_000:  # 1M+
                capacity = 100_000_000  # 100M capacity
            else:
                capacity = max(to_n * 10, 1_000_000)  # At least 1M capacity

            bf = BloomFilter(capacity=capacity, false_positive_rate=false_positive_rate)
            actual_from_n = 0  # Start from beginning for new filter
        else:
            # Reuse existing filter
            bf = existing["bloom_filter"]
            actual_from_n = from_n  # Use the actual from_n for incremental

            # Check if existing filter has enough capacity
            if bf.capacity < to_n * 1.2:  # Need 20% headroom
                print("      Bloom filter approaching capacity limit")

        # Add ONLY the NEW elements from actual_from_n to to_n
        self.reader.set_limit(to_n)

        # Show progress for large increments
        total_new = to_n - actual_from_n
        if total_new > 100000:
            print(f"     Adding {total_new:,} new elements...", end="", flush=True)

        batch_size = min(50000, max(1000, total_new // 20))  # Adaptive batch size
        processed = 0

        for i in range(actual_from_n, to_n, batch_size):
            end_i = min(i + batch_size, to_n)
            for j in range(i, end_i):
                if j < len(self.reader):
                    username = self.reader[j]
                    bf.add(username)
                    processed += 1

            # Show progress for large batches
            if total_new > 100000 and processed % 500000 == 0:
                progress = (processed / total_new) * 100
                print(f" {progress:.0f}%", end="", flush=True)

        if total_new > 100000:
            print(" Done!")

        setup_end = time.perf_counter()
        setup_time = (setup_end - setup_start) * 1000  # Convert to ms

        # Measure memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB

        return {
            "bloom_filter": bf,
            "setup_time": setup_time,
            "memory_usage": memory_usage,
        }

    def _build_cuckoo_filter_incremental(
        self, existing: Optional[Dict], from_n: int, to_n: int, **kwargs
    ) -> Dict[str, Any]:
        """Build cuckoo filter incrementally with smart capacity management"""
        import os

        import psutil

        from ..data_structures.cuckoo_filter import CuckooFilter

        false_positive_rate = kwargs.get("false_positive_rate", 0.01)
        bucket_size = kwargs.get("bucket_size", 4)

        setup_start = time.perf_counter()

        if existing is None or from_n == 0:
            # For large capacities, use conservative approach
            if to_n > 50_000_000:  # 50M+
                capacity = min(
                    to_n * 3, 1_000_000_000
                )  # More space for cuckoo, cap at 1B
            else:
                capacity = to_n * 2

            cf = CuckooFilter(
                capacity=capacity,
                false_positive_rate=false_positive_rate,
                bucket_size=bucket_size,
            )
            from_n = 0
            insertion_failures = 0
        else:
            cf = existing["cuckoo_filter"]
            insertion_failures = existing.get("insertion_failures", 0)

            # Check if we need to rebuild due to capacity constraints
            if cf.capacity < to_n * 1.5:  # Cuckoo needs more headroom
                print(
                    f"      Rebuilding Cuckoo filter (capacity {cf.capacity:,} < needed {to_n * 1.5:,.0f})"
                )
                new_capacity = (
                    max(to_n * 3, 1_000_000_000) if to_n > 50_000_000 else to_n * 3
                )
                cf = CuckooFilter(
                    capacity=new_capacity,
                    false_positive_rate=false_positive_rate,
                    bucket_size=bucket_size,
                )
                from_n = 0
                insertion_failures = 0

        # Add elements from from_n to to_n
        self.reader.set_limit(to_n)
        batch_size = min(5000, max(500, to_n // 200))  # Smaller batches for cuckoo

        for i in range(from_n, to_n, batch_size):
            end_i = min(i + batch_size, to_n)
            for j in range(i, end_i):
                if j < len(self.reader):
                    username = self.reader[j]
                    if not cf.add(username):
                        insertion_failures += 1
                        # If too many failures, stop (filter is likely full)
                        if insertion_failures > to_n * 0.05:  # More than 5% failures
                            print(
                                f"      Cuckoo filter insertion failures: {insertion_failures:,}"
                            )
                            break

            # Check if we hit too many failures
            if insertion_failures > to_n * 0.05:
                break

        setup_end = time.perf_counter()
        setup_time = (setup_end - setup_start) * 1000

        # Measure memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024

        return {
            "cuckoo_filter": cf,
            "insertion_failures": insertion_failures,
            "setup_time": setup_time,
            "memory_usage": memory_usage,
        }

    def _build_sorted_list_incremental(
        self, existing: Optional[Dict], from_n: int, to_n: int, **kwargs
    ) -> Dict[str, Any]:
        """Build sorted list - note: this still requires full rebuild for proper sorting"""
        from ..algorithms.binary_search import BinarySearch
        from ..data_structures.sorted_list import SortedList

        # For sorted list, we need to rebuild completely as we can't incrementally sort
        # But we can reuse the file if it exists and hasn't changed

        self.reader.set_limit(to_n)
        sorted_list = SortedList(self.reader)
        sorted_list.sort()
        algorithm = BinarySearch(sorted_list)

        return {"sorted_list": sorted_list, "algorithm": algorithm}

    def cleanup(self):
        """Clean up all data structures"""
        for structures in self.data_structures.values():
            if structures:
                if "bloom_filter" in structures:
                    structures["bloom_filter"].close()
                if "cuckoo_filter" in structures:
                    structures["cuckoo_filter"].close()
                if "sorted_list" in structures:
                    structures["sorted_list"].close()
                if "disk_hashset" in structures:
                    structures["disk_hashset"].close()
                    # Also clean up temporary file if it exists
                    if "temp_file_path" in structures:
                        try:
                            import os

                            os.unlink(structures["temp_file_path"])
                        except (OSError, FileNotFoundError):
                            pass  # File might already be deleted


class BenchmarkRunner:
    """Core benchmarking runner that handles timing and statistics"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: Dict[str, List[BenchmarkResult]] = {}
        self.data_manager: Optional[IncrementalDataStructureManager] = None
        self.current_algorithm: Optional[str] = None
        self.current_size: Optional[Union[int, float]] = None
        self.total_steps: int = 0
        self.current_step: int = 0

    def do_bench(
        self, fn: Callable[[], Any], quantiles: Optional[List[float]] = None
    ) -> Tuple[float, float, List[float]]:
        """Time a function call multiple times and return statistics"""
        if quantiles is None:
            quantiles = [0.5, 0.2, 0.8]

        # Warmup runs
        for _ in range(self.config.warmup_runs):
            fn()

        # Measurement runs
        times: List[float] = []
        total_runtime = 0.0

        while (
            len(times) < self.config.measure_runs
            or total_runtime < self.config.min_runtime_ms
        ):
            start = time.perf_counter()
            fn()
            end = time.perf_counter()

            runtime_ms = (end - start) * 1000
            times.append(runtime_ms)
            total_runtime += runtime_ms

            if (
                len(times) >= self.config.measure_runs
                and total_runtime >= self.config.min_runtime_ms
            ):
                break

        # Calculate statistics
        mean_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0

        return mean_time, std_dev, times

    def run_benchmark(
        self,
        benchmark_fn: Callable[..., float],
        setup_fns: Dict[str, Callable[..., Any]],
        reader=None,
    ) -> None:
        """Run the complete benchmark suite with progress indication and incremental building"""

        # Initialize progress tracking
        self.total_steps = len(self.config.line_vals) * len(self.config.x_vals)
        self.current_step = 0

        # Initialize incremental data manager for scalable algorithms
        if reader is not None:
            self.data_manager = IncrementalDataStructureManager(reader)

        print(f"\nStarting benchmark: {self.config.plot_name}")
        print(
            f"Testing {len(self.config.line_vals)} algorithms on {len(self.config.x_vals)} sizes"
        )
        print(f"Started at {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)

        try:
            for i, line_val in enumerate(self.config.line_vals):
                self.current_algorithm = line_val
                line_results = []

                # Get friendly algorithm name
                algo_name = (
                    self.config.line_names[i]
                    if i < len(self.config.line_names)
                    else line_val.replace("_", " ").title()
                )

                print(
                    f"\n🔍 [{i + 1}/{len(self.config.line_vals)}] Testing {algo_name}"
                )
                print("-" * 60)

                for j, x_val in enumerate(self.config.x_vals):
                    self.current_step += 1
                    self.current_size = x_val

                    # Progress indicator
                    progress = (self.current_step / self.total_steps) * 100
                    print(
                        f"[{self.current_step:2d}/{self.total_steps}] "
                        f"N={x_val:>12,} ({progress:5.1f}%) ",
                        end="",
                        flush=True,
                    )

                    start_time = time.time()

                    try:
                        # Setup arguments
                        args = self.config.args.copy()
                        args[self.config.x_names[0]] = x_val
                        args[self.config.line_arg] = line_val

                        # Use incremental building for scalable algorithms
                        setup_time = None
                        memory_usage = None

                        if self._is_scalable_algorithm(line_val) and self.data_manager:
                            # Use incremental data structure building
                            setup_data = self._setup_incremental(
                                line_val, x_val, **args
                            )
                            args.update(setup_data)
                            if (
                                self.config.measure_setup_time
                                and "setup_time" in setup_data
                            ):
                                setup_time = setup_data["setup_time"]
                            if (
                                self.config.measure_memory
                                and "memory_usage" in setup_data
                            ):
                                memory_usage = setup_data["memory_usage"]
                        elif line_val in setup_fns:
                            # Use traditional setup for small algorithms
                            setup_data = setup_fns[line_val](**args)
                            args.update(setup_data)
                            if (
                                self.config.measure_setup_time
                                and "setup_time" in setup_data
                            ):
                                setup_time = setup_data["setup_time"]
                            if (
                                self.config.measure_memory
                                and "memory_usage" in setup_data
                            ):
                                memory_usage = setup_data["memory_usage"]

                        # Benchmark the search function
                        mean_time, std_dev, measurements = self.do_bench(
                            lambda: benchmark_fn(**args)
                        )

                        result = BenchmarkResult(
                            value=mean_time,
                            std_dev=std_dev,
                            measurements=measurements,
                            config_name=line_val,
                            x_value=x_val,
                            setup_time=setup_time,
                            memory_usage=memory_usage,
                        )
                        line_results.append(result)

                        # Clean up temporary resources for current run
                        self._cleanup_current_run(args)

                        elapsed = time.time() - start_time
                        print(
                            f"→ {mean_time:8.3f}ms (±{std_dev:6.3f}) [{elapsed:4.1f}s]"
                        )

                    except Exception as e:
                        elapsed = time.time() - start_time
                        print(f"→ FAILED: {str(e)[:50]}... [{elapsed:4.1f}s]")
                        # Create a dummy result to maintain structure
                        result = BenchmarkResult(
                            value=float("inf"),
                            std_dev=0.0,
                            measurements=[],
                            config_name=line_val,
                            x_value=x_val,
                            setup_time=None,
                            memory_usage=None,
                        )
                        line_results.append(result)

                self.results[line_val] = line_results

        finally:
            # Cleanup incremental data structures
            if self.data_manager:
                print("\nCleaning up data structures...")
                self.data_manager.cleanup()

        print("\n" + "=" * 80)
        print(f"Benchmark completed at {datetime.now().strftime('%H:%M:%S')}")

    def _is_scalable_algorithm(self, algorithm: str) -> bool:
        """Check if algorithm can benefit from incremental building"""
        scalable_algorithms = {"bloom_filter", "cuckoo_filter"}
        return algorithm in scalable_algorithms

    def _setup_incremental(
        self, algorithm: str, target_n: Union[int, float], **kwargs
    ) -> Dict[str, Any]:
        """Setup data structure using incremental building"""
        if self.data_manager is None:
            return {}

        target_n_int = int(target_n)  # Convert to int for data structure methods

        if algorithm == "bloom_filter":
            return self.data_manager.get_or_build_structure(
                "bloom_filter", target_n_int, **kwargs
            )
        elif algorithm == "cuckoo_filter":
            return self.data_manager.get_or_build_structure(
                "cuckoo_filter", target_n_int, **kwargs
            )
        else:
            return {}

    def _cleanup_current_run(self, args: Dict[str, Any]) -> None:
        """Clean up resources from the current benchmark run"""
        # Clean up disk hashset temporary files
        if "disk_hashset" in args:
            disk_hashset = args["disk_hashset"]
            disk_hashset.close()

        if "temp_file_path" in args:
            try:
                import os

                os.unlink(args["temp_file_path"])
            except (OSError, FileNotFoundError):
                pass  # File might already be deleted

    def generate_plot(self, show_plots: bool = True, save_plot: bool = True) -> None:
        """Generate performance plot with error bars"""
        # Generate search time plot
        self._generate_single_plot(
            "Search Time",
            self.config.ylabel,
            lambda r: r.value,
            lambda r: r.std_dev,
            show_plots,
            save_plot,
        )

        # Generate setup time plot if measured
        if self.config.measure_setup_time:
            self._generate_single_plot(
                "Setup Time",
                "Setup Time (ms)",
                lambda r: r.setup_time,
                lambda r: 0,
                show_plots,
                save_plot,
                suffix="-setup",
            )

        # Generate memory usage plot if measured
        if self.config.measure_memory:
            self._generate_single_plot(
                "Memory Usage",
                "Memory (MB)",
                lambda r: r.memory_usage,
                lambda r: 0,
                show_plots,
                save_plot,
                suffix="-memory",
            )

    def _generate_single_plot(
        self,
        title_suffix: str,
        ylabel: str,
        value_fn: Callable,
        error_fn: Callable,
        show_plots: bool,
        save_plot: bool,
        suffix: str = "",
    ) -> None:
        """Generate a single plot"""
        plt.figure(figsize=(12, 8))

        for i, line_val in enumerate(self.config.line_vals):
            if line_val not in self.results:
                continue

            results = self.results[line_val]
            x_values = [r.x_value for r in results]
            y_values = [value_fn(r) for r in results if value_fn(r) is not None]
            y_errors = [error_fn(r) for r in results if value_fn(r) is not None]

            # Skip if no valid values
            if not y_values:
                continue

            color, style = (
                self.config.styles[i] if i < len(self.config.styles) else ("blue", "-")
            )
            label = (
                self.config.line_names[i]
                if i < len(self.config.line_names)
                else line_val
            )

            plt.errorbar(
                x_values[: len(y_values)],
                y_values,
                yerr=y_errors,
                color=color,
                linestyle=style,
                marker="o",
                label=label,
                capsize=5,
                capthick=2,
            )

        plt.xlabel(self.config.x_names[0])
        plt.ylabel(ylabel)
        plt.title(f"{self.config.plot_name} - {title_suffix}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_plot:
            filename = f"{self.config.plot_name}{suffix}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")

        if show_plots:
            plt.show()
        else:
            plt.close()

    def print_data(self) -> None:
        """Print detailed benchmark results"""
        print(f"\n{self.config.plot_name} Benchmark Results")
        print("=" * 80)

        for line_val in self.config.line_vals:
            if line_val not in self.results:
                continue

            results = self.results[line_val]
            line_name = self.config.line_names[self.config.line_vals.index(line_val)]

            print(f"\n{line_name} ({line_val}):")

            # Header with conditional columns
            header = f"{'N':<10} {'Search (ms)':<12} {'Std Dev':<10}"
            if self.config.measure_setup_time:
                header += f" {'Setup (ms)':<12}"
            if self.config.measure_memory:
                header += f" {'Memory (MB)':<12}"
            print(header)
            print("-" * len(header))

            for result in results:
                row = f"{result.x_value:<10} {result.value:<12.4f} {result.std_dev:<10.4f}"
                if self.config.measure_setup_time and result.setup_time is not None:
                    row += f" {result.setup_time:<12.4f}"
                elif self.config.measure_setup_time:
                    row += f" {'N/A':<12}"
                if self.config.measure_memory and result.memory_usage is not None:
                    row += f" {result.memory_usage:<12.2f}"
                elif self.config.measure_memory:
                    row += f" {'N/A':<12}"

                print(row)


def perf_report(config: BenchmarkConfig):
    """Decorator for performance reporting, similar to triton.testing.perf_report"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # If called with setup functions, run the benchmark
            if "setup_fns" in kwargs:
                setup_fns = kwargs.pop("setup_fns")
                reader = kwargs.pop("reader", None)
                runner = BenchmarkRunner(config)
                runner.run_benchmark(func, setup_fns, reader=reader)
                return runner
            else:
                # Regular function call
                return func(*args, **kwargs)

        # Add run method to the wrapper
        def run(
            show_plots: bool = True,
            print_data: bool = True,
            setup_fns: Optional[Dict[str, Callable]] = None,
            reader=None,
        ):
            if setup_fns is None:
                setup_fns = {}
            runner = BenchmarkRunner(config)
            runner.run_benchmark(func, setup_fns, reader=reader)

            if print_data:
                runner.print_data()
            if show_plots or config.plot_name:
                runner.generate_plot(show_plots=show_plots)

            return runner

        wrapper.run = run
        wrapper.config = config
        return wrapper

    return decorator


class ComplexityBenchmarkHelper:
    """Helper class for measuring time and space complexity of data structures"""

    @staticmethod
    def setup_linear_search(reader, N: int, **kwargs) -> Dict[str, Any]:
        """Setup for linear search - just needs a reader with limit"""
        import os

        import psutil

        setup_start = time.perf_counter()
        reader.set_limit(N)

        from ..algorithms.linear_search import LinearSearch

        algorithm = LinearSearch(reader)

        setup_end = time.perf_counter()
        setup_time = (setup_end - setup_start) * 1000

        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024

        return {
            "algorithm": algorithm,
            "reader": reader,
            "setup_time": setup_time,
            "memory_usage": memory_usage,
        }

    @staticmethod
    def setup_binary_search(reader, N: int, **kwargs) -> Dict[str, Any]:
        """Setup for binary search - needs sorted list creation"""
        import os

        import psutil

        setup_start = time.perf_counter()
        reader.set_limit(N)

        from ..algorithms.binary_search import BinarySearch
        from ..data_structures.sorted_list import SortedList

        # Create sorted list from reader data
        sorted_list = SortedList(reader)
        sorted_list.sort()
        algorithm = BinarySearch(sorted_list)

        setup_end = time.perf_counter()
        setup_time = (setup_end - setup_start) * 1000

        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024

        return {
            "algorithm": algorithm,
            "reader": reader,
            "sorted_list": sorted_list,
            "setup_time": setup_time,
            "memory_usage": memory_usage,
        }

    @staticmethod
    def setup_bloom_filter(reader, N: int, **kwargs) -> Dict[str, Any]:
        """Setup for bloom filter - create fresh for each N to measure complexity"""
        import os

        import psutil

        setup_start = time.perf_counter()
        reader.set_limit(N)

        from ..data_structures.bloom_filter import BloomFilter

        # Create bloom filter with capacity for CURRENT N only
        false_positive_rate = kwargs.get("false_positive_rate", 0.01)
        bf = BloomFilter(capacity=N, false_positive_rate=false_positive_rate)

        # Populate with all usernames in range
        for username in reader:
            bf.add(username)

        setup_end = time.perf_counter()
        setup_time = (setup_end - setup_start) * 1000

        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024

        return {
            "bloom_filter": bf,
            "reader": reader,
            "setup_time": setup_time,
            "memory_usage": memory_usage,
        }

    @staticmethod
    def setup_cuckoo_filter(reader, N: int, **kwargs) -> Dict[str, Any]:
        """Setup for cuckoo filter - create fresh for each N to measure complexity"""
        import os

        import psutil

        setup_start = time.perf_counter()
        reader.set_limit(N)

        from ..data_structures.cuckoo_filter import CuckooFilter

        # Create cuckoo filter with capacity for CURRENT N only
        false_positive_rate = kwargs.get("false_positive_rate", 0.01)
        bucket_size = kwargs.get("bucket_size", 4)
        cf = CuckooFilter(
            capacity=N, false_positive_rate=false_positive_rate, bucket_size=bucket_size
        )

        # Populate with all usernames in range
        insertion_failures = 0
        for username in reader:
            if not cf.add(username):
                insertion_failures += 1

        setup_end = time.perf_counter()
        setup_time = (setup_end - setup_start) * 1000

        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024

        return {
            "cuckoo_filter": cf,
            "reader": reader,
            "setup_time": setup_time,
            "memory_usage": memory_usage,
            "insertion_failures": insertion_failures,
        }

    @staticmethod
    def setup_disk_hashset(reader, N: int, **kwargs) -> Dict[str, Any]:
        """Setup for disk hashset - create fresh for each N to measure complexity"""
        import os
        import tempfile

        import psutil

        setup_start = time.perf_counter()
        reader.set_limit(N)

        from ..data_structures.disk_hashset import DiskHashSet

        # Create temporary file for the disk hashset
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".dat")
        temp_file.close()
        path = temp_file.name

        # Calculate appropriate number of slots based on N
        # Use load factor of ~0.75 for good performance
        target_slots = int(N / 0.75)
        # Find next power of 2
        num_slots_power = max(3, (target_slots - 1).bit_length())

        # Create disk hashset with appropriate capacity
        dhs = DiskHashSet(path, num_slots_power=num_slots_power, create=True)

        # Populate with all usernames in range
        insertion_failures = 0
        for username in reader:
            try:
                if not dhs.add(username):
                    # Already exists (shouldn't happen with unique usernames)
                    pass
            except RuntimeError:
                # Hash table is full
                insertion_failures += 1

        setup_end = time.perf_counter()
        setup_time = (setup_end - setup_start) * 1000

        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024

        return {
            "disk_hashset": dhs,
            "reader": reader,
            "setup_time": setup_time,
            "memory_usage": memory_usage,
            "insertion_failures": insertion_failures,
            "temp_file_path": path,
        }

    @staticmethod
    def get_existing_username(reader, **kwargs) -> str:
        """Get a random existing username from the reader"""
        n = len(reader)
        if n == 0:
            return ""
        idx = random.randint(0, n - 1)
        return reader[idx]

    @staticmethod
    def get_non_existing_username(**kwargs) -> str:
        """Get a username that definitely doesn't exist"""
        return "@not@exists@"


# Keep the old helper for backward compatibility
SearchBenchmarkHelper = ComplexityBenchmarkHelper


def create_search_benchmark(
    reader, config: BenchmarkConfig, test_existing: bool = True
) -> Callable:
    """Create a search benchmark function for different algorithms"""

    setup_functions = {
        "linear_search": ComplexityBenchmarkHelper.setup_linear_search,
        "binary_search": ComplexityBenchmarkHelper.setup_binary_search,
        "bloom_filter": ComplexityBenchmarkHelper.setup_bloom_filter,
        "cuckoo_filter": ComplexityBenchmarkHelper.setup_cuckoo_filter,
        "disk_hashset": ComplexityBenchmarkHelper.setup_disk_hashset,
    }

    @perf_report(config)
    def benchmark(N: int, provider: str, **kwargs):
        """Benchmark function that measures search performance"""

        # Get the search target
        if test_existing:
            target = ComplexityBenchmarkHelper.get_existing_username(reader)
        else:
            target = ComplexityBenchmarkHelper.get_non_existing_username()

        # Perform the search based on provider type
        if provider == "linear_search":
            algorithm = kwargs["algorithm"]
            result = algorithm.search(target)
            return result.time_taken * 1000  # Convert to milliseconds

        elif provider == "binary_search":
            algorithm = kwargs["algorithm"]
            result = algorithm.search(target)
            return result.time_taken * 1000  # Convert to milliseconds

        elif provider == "bloom_filter":
            bloom_filter = kwargs["bloom_filter"]
            start_time = time.perf_counter()
            bloom_filter.contains(target)
            end_time = time.perf_counter()
            return (end_time - start_time) * 1000  # Convert to milliseconds

        elif provider == "cuckoo_filter":
            cuckoo_filter = kwargs["cuckoo_filter"]
            start_time = time.perf_counter()
            cuckoo_filter.contains(target)
            end_time = time.perf_counter()
            return (end_time - start_time) * 1000  # Convert to milliseconds

        elif provider == "disk_hashset":
            disk_hashset = kwargs["disk_hashset"]
            start_time = time.perf_counter()
            disk_hashset.contains(target)
            end_time = time.perf_counter()
            return (end_time - start_time) * 1000  # Convert to milliseconds

        else:
            raise ValueError(f"Unknown provider: {provider}")

    # Set up the setup functions for the benchmark
    benchmark_setup_fns = {}
    for provider in config.line_vals:
        if provider in setup_functions:
            benchmark_setup_fns[provider] = lambda reader=reader, **kw: setup_functions[
                provider
            ](reader, **kw)

    # Override the run method to include setup functions and reader
    original_run = benchmark.run

    def enhanced_run(show_plots: bool = True, print_data: bool = True):
        return original_run(
            show_plots=show_plots,
            print_data=print_data,
            setup_fns=benchmark_setup_fns,
            reader=reader,
        )

    benchmark.run = enhanced_run
    return benchmark
