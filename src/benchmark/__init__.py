"""
Benchmarking module for search algorithms.

This module provides a Triton-inspired benchmarking framework for comparing
different search algorithms and data structures with complexity analysis.
"""

from .benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    ComplexityBenchmarkHelper,
    SearchBenchmarkHelper,  # Backward compatibility alias
    create_search_benchmark,
    perf_report,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRunner",
    "ComplexityBenchmarkHelper",
    "SearchBenchmarkHelper",
    "create_search_benchmark",
    "perf_report",
]
