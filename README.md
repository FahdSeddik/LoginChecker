# LoginChecker: Billion-Scale Membership Testing

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)]()

A comprehensive implementation and comparison of membership testing algorithms for billion-scale datasets. This project implements and benchmarks five different approaches: Linear Search, Binary Search, Disk-based Hash Table, Bloom Filter, and Cuckoo Filter.

## Features

- Billion-scale capability with memory-mapped storage
- Five membership testing algorithms: Linear Search, Binary Search, Hash Table, Bloom Filter, Cuckoo Filter
- Performance benchmarking with detailed metrics
- Memory-efficient persistent storage
- Comprehensive test suite

## Algorithm Complexity Summary

| Algorithm | Time Complexity | Space Complexity | Scale Limit | Key Features |
|-----------|----------------|------------------|-------------|--------------|
| Linear Search | O(n) | O(1) | 100K | Simple, no preprocessing |
| Binary Search | O(log n) | O(n) | 100M | Requires sorted data |
| Hash Table | O(1) avg, O(n) worst | O(n) | 1B+ | Memory-mapped, persistent |
| Bloom Filter | O(k) | O(m) | 1B+ | Probabilistic, no false negatives |
| Cuckoo Filter | O(1) expected | O(n) | 1B+ | Supports deletions, exact FPR |

## Installation

### Prerequisites
- Python 3.12+
- Git

### Setup
```bash
git clone https://github.com/FahdSeddik/LoginChecker.git
cd LoginChecker
uv sync
python data/generate.py
```

## Usage

### Quick Start
```python
from data.reader import UsernameReader
from src.algorithms.linear_search import LinearSearch
from src.data_structures.bloom_filter import BloomFilter

# Load username data
reader = UsernameReader("data/usernames.dat")

# Linear search example
linear_search = LinearSearch(reader)
result = linear_search.search("john_doe")
print(f"Found: {result.found}, Comparisons: {result.comparisons}")

# Bloom filter example
bloom = BloomFilter(capacity=1_000_000, false_positive_rate=0.01)
bloom.add("john_doe")
print(f"Contains 'john_doe': {bloom.contains('john_doe')}")
```

### Run Benchmarks
```bash
# Quick algorithm comparison
python src/benchmark/simple_runner.py --algorithms linear_search,bloom_filter --sizes 1000,10000,100000

# Full benchmark suite (not recommended)
python src/benchmark/run_billion_benchmark.py
```

### Run Tests
```bash
# All tests
pytest --run-slow
```

## Project Structure

```
├── src/
│   ├── algorithms/          # Search algorithms
│   │   ├── algorithm.py
│   │   ├── binary_search.py
│   │   └── linear_search.py
│   ├── benchmark/           # Benchmarking tools
│   │   ├── benchmark.py
│   │   ├── run_billion_benchmark.py
│   │   └── simple_runner.py
│   └── data_structures/     # Data structures
│       ├── bloom_filter.py
│       ├── cuckoo_filter.py
│       ├── disk_hashset.py
│       └── sorted_list.py
├── data/                    # Data generation and access
│   ├── generate.py
│   ├── reader.py
│   └── writer.py
├── tests/                   # Test suite
├── report/                  # Academic report
└── pyproject.toml
```
