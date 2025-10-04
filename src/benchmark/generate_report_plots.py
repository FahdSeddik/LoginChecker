#!/usr/bin/env python3
"""
Generate plots for the LoginChecker report based on experimental results.
This script processes the benchmark data and creates publication-quality plots.
"""

import math

import matplotlib.pyplot as plt

# Set up matplotlib for publication quality plots
plt.style.use("seaborn-v0_8")
plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Experimental results data
bloom_filter_data = {
    "N": [
        100,
        500,
        1000,
        5000,
        10000,
        50000,
        100000,
        500000,
        1000000,
        5000000,
        10000000,
        50000000,
        100000000,
        500000000,
        1000000000,
    ],
    "Search (ms)": [
        0.0032,
        0.0033,
        0.0034,
        0.0035,
        0.0035,
        0.0036,
        0.0037,
        0.0040,
        0.0041,
        0.0044,
        0.0043,
        0.0083,
        0.0050,
        0.0052,
        0.0528,
    ],
    "Std Dev": [
        0.0005,
        0.0005,
        0.0010,
        0.0004,
        0.0004,
        0.0010,
        0.0012,
        0.0018,
        0.0012,
        0.0010,
        0.0003,
        0.0024,
        0.0016,
        0.0015,
        0.1026,
    ],
    "Setup (ms)": [
        1.3305,
        2.0770,
        3.5136,
        13.2839,
        25.3581,
        120.0453,
        235.2801,
        1375.2768,
        2746.6367,
        15192.4652,
        30039.1935,
        307629.3582,
        623796.2715,
        3323986.3993,
        5637885.9782,
    ],
    "Memory (MB)": [
        62.79,
        64.04,
        65.29,
        66.66,
        68.04,
        70.16,
        72.41,
        85.79,
        106.66,
        296.91,
        507.54,
        3559.29,
        6802.54,
        16753.16,
        20666.54,
    ],
}

cuckoo_filter_data = {
    "N": [
        100,
        500,
        1000,
        5000,
        10000,
        50000,
        100000,
        500000,
        1000000,
        5000000,
        10000000,
    ],
    "Search (ms)": [
        0.0028,
        0.0027,
        0.0028,
        0.0029,
        0.0029,
        0.0028,
        0.0028,
        0.0029,
        0.0031,
        0.0032,
        0.0033,
    ],
    "Std Dev": [
        0.0015,
        0.0010,
        0.0003,
        0.0003,
        0.0003,
        0.0012,
        0.0008,
        0.0011,
        0.0012,
        0.0014,
        0.0008,
    ],
    "Setup (ms)": [
        0.7158,
        1.3239,
        2.1859,
        10.4952,
        20.3683,
        99.5435,
        202.2363,
        953.7534,
        2080.3311,
        18948.4189,
        240735.7746,
    ],
    "Memory (MB)": [
        61.25,
        61.38,
        61.62,
        61.75,
        62.12,
        63.50,
        65.08,
        77.02,
        94.63,
        202.99,
        363.10,
    ],
}

binary_search_data = {
    "N": [
        100,
        500,
        1000,
        5000,
        10000,
        50000,
        100000,
        500000,
        1000000,
        5000000,
        10000000,
        50000000,
        100000000,
    ],
    "Search (ms)": [
        0.0051,
        0.0067,
        0.0072,
        0.0085,
        0.0092,
        0.0106,
        0.0114,
        0.0133,
        0.0143,
        0.0207,
        0.0280,
        0.0385,
        0.0429,
    ],
    "Std Dev": [
        0.0009,
        0.0010,
        0.0009,
        0.0011,
        0.0012,
        0.0011,
        0.0012,
        0.0027,
        0.0033,
        0.0071,
        0.0122,
        0.0124,
        0.0120,
    ],
    "Setup (ms)": [
        11.5492,
        1.4319,
        2.2796,
        9.1561,
        17.9948,
        90.5822,
        195.8448,
        1086.2816,
        2197.8684,
        11890.4127,
        24270.5936,
        121197.3644,
        241709.7607,
    ],
    "Memory (MB)": [
        273.63,
        273.63,
        273.75,
        273.75,
        273.73,
        275.60,
        276.68,
        283.44,
        290.85,
        280.37,
        283.38,
        1066.70,
        2047.30,
    ],
}

linear_search_data = {
    "N": [100, 500, 1000, 5000, 10000, 50000, 100000],
    "Search (ms)": [0.0246, 0.1186, 0.2406, 1.2008, 2.4590, 11.9414, 18.4995],
    "Std Dev": [0.0131, 0.0684, 0.1387, 0.7038, 1.3926, 7.0158, 12.4949],
    "Setup (ms)": [1.1293, 0.0067, 0.0074, 0.0058, 0.0071, 0.0070, 0.0064],
    "Memory (MB)": [61.64, 63.27, 63.64, 63.77, 63.77, 63.89, 64.64],
}

disk_hashset_data = {
    "N": [
        100,
        500,
        1000,
        5000,
        10000,
        50000,
        100000,
        500000,
        1000000,
        5000000,
        10000000,
        50000000,
        100000000,
    ],
    "Search (ms)": [
        0.0027,
        0.0023,
        0.0022,
        0.0024,
        0.0024,
        0.0023,
        0.0025,
        0.0023,
        0.0026,
        0.0028,
        0.0026,
        0.0027,
        0.0028,
    ],
    "Std Dev": [
        0.0013,
        0.0009,
        0.0004,
        0.0006,
        0.0008,
        0.0007,
        0.0003,
        0.0005,
        0.0017,
        0.0005,
        0.0011,
        0.0004,
        0.0008,
    ],
    "Setup (ms)": [
        0.9550,
        0.8818,
        1.5668,
        7.4231,
        14.9100,
        71.0121,
        138.8424,
        694.3424,
        1377.4587,
        7097.6995,
        13669.9660,
        74786.4100,
        200865.6347,
    ],
    "Memory (MB)": [
        61.70,
        61.70,
        62.07,
        62.32,
        62.94,
        67.44,
        72.50,
        104.32,
        146.00,
        414.63,
        766.00,
        3068.69,
        6075.17,
    ],
}


def create_search_time_scaling_plot():
    """Create search time scaling plot showing all algorithms to their limits."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colors and markers for consistency
    colors = {
        "linear": "blue",
        "binary": "purple",
        "bloom": "green",
        "cuckoo": "red",
        "disk": "orange",
    }
    markers = {"linear": "o", "binary": "s", "bloom": "^", "cuckoo": "d", "disk": "h"}

    # Plot all algorithms to their maximum tested scales
    ax.loglog(
        linear_search_data["N"],
        linear_search_data["Search (ms)"],
        f"{markers['linear']}-",
        label="Linear Search",
        linewidth=2.5,
        markersize=6,
        color=colors["linear"],
    )
    ax.loglog(
        binary_search_data["N"],
        binary_search_data["Search (ms)"],
        f"{markers['binary']}-",
        label="Binary Search",
        linewidth=2.5,
        markersize=6,
        color=colors["binary"],
    )
    ax.loglog(
        bloom_filter_data["N"],
        bloom_filter_data["Search (ms)"],
        f"{markers['bloom']}-",
        label="Bloom Filter",
        linewidth=2.5,
        markersize=6,
        color=colors["bloom"],
    )
    ax.loglog(
        cuckoo_filter_data["N"],
        cuckoo_filter_data["Search (ms)"],
        f"{markers['cuckoo']}-",
        label="Cuckoo Filter",
        linewidth=2.5,
        markersize=6,
        color=colors["cuckoo"],
    )
    ax.loglog(
        disk_hashset_data["N"],
        disk_hashset_data["Search (ms)"],
        f"{markers['disk']}-",
        label="Disk HashSet",
        linewidth=2.5,
        markersize=6,
        color=colors["disk"],
    )

    # Add termination markers
    ax.scatter(
        [linear_search_data["N"][-1]],
        [linear_search_data["Search (ms)"][-1]],
        marker="X",
        s=150,
        color=colors["linear"],
        zorder=10,
    )
    ax.scatter(
        [cuckoo_filter_data["N"][-1]],
        [cuckoo_filter_data["Search (ms)"][-1]],
        marker="X",
        s=150,
        color=colors["cuckoo"],
        zorder=10,
    )
    ax.scatter(
        [binary_search_data["N"][-1]],
        [binary_search_data["Search (ms)"][-1]],
        marker="X",
        s=150,
        color=colors["binary"],
        zorder=10,
    )
    ax.scatter(
        [disk_hashset_data["N"][-1]],
        [disk_hashset_data["Search (ms)"][-1]],
        marker="X",
        s=150,
        color=colors["disk"],
        zorder=10,
    )

    ax.set_xlabel("Dataset Size (N)", fontsize=14)
    ax.set_ylabel("Search Time (ms)", fontsize=14)
    ax.set_title("Search Time Scaling Comparison", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "/home/fseddik/MSc/Fall2025/Algo/A1/report/search-time-scaling.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_setup_time_scaling_plot():
    """Create setup time scaling plot showing all algorithms to their limits."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colors and markers for consistency
    colors = {
        "linear": "blue",
        "binary": "purple",
        "bloom": "green",
        "cuckoo": "red",
        "disk": "orange",
    }
    markers = {"linear": "o", "binary": "s", "bloom": "^", "cuckoo": "d", "disk": "h"}

    ax.loglog(
        linear_search_data["N"],
        linear_search_data["Setup (ms)"],
        f"{markers['linear']}-",
        label="Linear Search",
        linewidth=2.5,
        markersize=6,
        color=colors["linear"],
    )
    ax.loglog(
        binary_search_data["N"],
        binary_search_data["Setup (ms)"],
        f"{markers['binary']}-",
        label="Binary Search",
        linewidth=2.5,
        markersize=6,
        color=colors["binary"],
    )
    ax.loglog(
        bloom_filter_data["N"],
        bloom_filter_data["Setup (ms)"],
        f"{markers['bloom']}-",
        label="Bloom Filter",
        linewidth=2.5,
        markersize=6,
        color=colors["bloom"],
    )
    ax.loglog(
        cuckoo_filter_data["N"],
        cuckoo_filter_data["Setup (ms)"],
        f"{markers['cuckoo']}-",
        label="Cuckoo Filter",
        linewidth=2.5,
        markersize=6,
        color=colors["cuckoo"],
    )
    ax.loglog(
        disk_hashset_data["N"],
        disk_hashset_data["Setup (ms)"],
        f"{markers['disk']}-",
        label="Disk HashSet",
        linewidth=2.5,
        markersize=6,
        color=colors["disk"],
    )

    # Add termination markers
    ax.scatter(
        [linear_search_data["N"][-1]],
        [linear_search_data["Setup (ms)"][-1]],
        marker="X",
        s=150,
        color=colors["linear"],
        zorder=10,
    )
    ax.scatter(
        [cuckoo_filter_data["N"][-1]],
        [cuckoo_filter_data["Setup (ms)"][-1]],
        marker="X",
        s=150,
        color=colors["cuckoo"],
        zorder=10,
    )
    ax.scatter(
        [binary_search_data["N"][-1]],
        [binary_search_data["Setup (ms)"][-1]],
        marker="X",
        s=150,
        color=colors["binary"],
        zorder=10,
    )
    ax.scatter(
        [disk_hashset_data["N"][-1]],
        [disk_hashset_data["Setup (ms)"][-1]],
        marker="X",
        s=150,
        color=colors["disk"],
        zorder=10,
    )

    ax.set_xlabel("Dataset Size (N)", fontsize=14)
    ax.set_ylabel("Setup Time (ms)", fontsize=14)
    ax.set_title("Setup Time Scaling Comparison", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "/home/fseddik/MSc/Fall2025/Algo/A1/report/setup-time-scaling.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_memory_usage_scaling_plot():
    """Create memory usage scaling plot showing all algorithms to their limits."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colors and markers for consistency
    colors = {
        "linear": "blue",
        "binary": "purple",
        "bloom": "green",
        "cuckoo": "red",
        "disk": "orange",
    }
    markers = {"linear": "o", "binary": "s", "bloom": "^", "cuckoo": "d", "disk": "h"}

    ax.loglog(
        linear_search_data["N"],
        linear_search_data["Memory (MB)"],
        f"{markers['linear']}-",
        label="Linear Search",
        linewidth=2.5,
        markersize=6,
        color=colors["linear"],
    )
    ax.loglog(
        binary_search_data["N"],
        binary_search_data["Memory (MB)"],
        f"{markers['binary']}-",
        label="Binary Search",
        linewidth=2.5,
        markersize=6,
        color=colors["binary"],
    )
    ax.loglog(
        bloom_filter_data["N"],
        bloom_filter_data["Memory (MB)"],
        f"{markers['bloom']}-",
        label="Bloom Filter",
        linewidth=2.5,
        markersize=6,
        color=colors["bloom"],
    )
    ax.loglog(
        cuckoo_filter_data["N"],
        cuckoo_filter_data["Memory (MB)"],
        f"{markers['cuckoo']}-",
        label="Cuckoo Filter",
        linewidth=2.5,
        markersize=6,
        color=colors["cuckoo"],
    )
    ax.loglog(
        disk_hashset_data["N"],
        disk_hashset_data["Memory (MB)"],
        f"{markers['disk']}-",
        label="Disk HashSet",
        linewidth=2.5,
        markersize=6,
        color=colors["disk"],
    )

    # Add termination markers
    ax.scatter(
        [linear_search_data["N"][-1]],
        [linear_search_data["Memory (MB)"][-1]],
        marker="X",
        s=150,
        color=colors["linear"],
        zorder=10,
    )
    ax.scatter(
        [cuckoo_filter_data["N"][-1]],
        [cuckoo_filter_data["Memory (MB)"][-1]],
        marker="X",
        s=150,
        color=colors["cuckoo"],
        zorder=10,
    )
    ax.scatter(
        [binary_search_data["N"][-1]],
        [binary_search_data["Memory (MB)"][-1]],
        marker="X",
        s=150,
        color=colors["binary"],
        zorder=10,
    )
    ax.scatter(
        [disk_hashset_data["N"][-1]],
        [disk_hashset_data["Memory (MB)"][-1]],
        marker="X",
        s=150,
        color=colors["disk"],
        zorder=10,
    )

    ax.set_xlabel("Dataset Size (N)", fontsize=14)
    ax.set_ylabel("Memory Usage (MB)", fontsize=14)
    ax.set_title("Memory Usage Scaling Comparison", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "/home/fseddik/MSc/Fall2025/Algo/A1/report/memory-usage-scaling.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_efficiency_comparison_plot():
    """Create efficiency comparison plot showing normalized performance."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colors and markers for consistency
    colors = {
        "linear": "blue",
        "binary": "purple",
        "bloom": "green",
        "cuckoo": "red",
        "disk": "orange",
    }
    markers = {"linear": "o", "binary": "s", "bloom": "^", "cuckoo": "d", "disk": "h"}

    # Calculate log(N) for normalization where applicable
    linear_log_n = [math.log(n) for n in linear_search_data["N"]]
    binary_log_n = [math.log(n) for n in binary_search_data["N"]]

    ax.semilogx(
        linear_search_data["N"],
        [
            t / log_n
            for t, log_n in zip(linear_search_data["Search (ms)"], linear_log_n)
        ],
        f"{markers['linear']}-",
        label="Linear Search (÷log n)",
        linewidth=2.5,
        markersize=6,
        color=colors["linear"],
    )
    ax.semilogx(
        binary_search_data["N"],
        [
            t / log_n
            for t, log_n in zip(binary_search_data["Search (ms)"], binary_log_n)
        ],
        f"{markers['binary']}-",
        label="Binary Search (÷log n)",
        linewidth=2.5,
        markersize=6,
        color=colors["binary"],
    )
    ax.semilogx(
        bloom_filter_data["N"],
        [t for t in bloom_filter_data["Search (ms)"]],
        f"{markers['bloom']}-",
        label="Bloom Filter (O(k))",
        linewidth=2.5,
        markersize=6,
        color=colors["bloom"],
    )
    ax.semilogx(
        cuckoo_filter_data["N"],
        [t for t in cuckoo_filter_data["Search (ms)"]],
        f"{markers['cuckoo']}-",
        label="Cuckoo Filter (O(1))",
        linewidth=2.5,
        markersize=6,
        color=colors["cuckoo"],
    )
    ax.semilogx(
        disk_hashset_data["N"],
        [t for t in disk_hashset_data["Search (ms)"]],
        f"{markers['disk']}-",
        label="Disk HashSet (O(1))",
        linewidth=2.5,
        markersize=6,
        color=colors["disk"],
    )

    ax.set_xlabel("Dataset Size (N)", fontsize=14)
    ax.set_ylabel("Normalized Search Time", fontsize=14)
    ax.set_title("Algorithm Efficiency Comparison", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "/home/fseddik/MSc/Fall2025/Algo/A1/report/efficiency-comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_search_time_comparison():
    """Create search time comparison plot for smaller datasets where all algorithms work."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get data for sizes <= 100K where all algorithms work
    max_n = 100000

    # Filter data
    linear_n = [n for n in linear_search_data["N"] if n <= max_n]
    linear_search = [
        linear_search_data["Search (ms)"][i]
        for i, n in enumerate(linear_search_data["N"])
        if n <= max_n
    ]

    binary_n = [n for n in binary_search_data["N"] if n <= max_n]
    binary_search = [
        binary_search_data["Search (ms)"][i]
        for i, n in enumerate(binary_search_data["N"])
        if n <= max_n
    ]

    bloom_n = [n for n in bloom_filter_data["N"] if n <= max_n]
    bloom_search = [
        bloom_filter_data["Search (ms)"][i]
        for i, n in enumerate(bloom_filter_data["N"])
        if n <= max_n
    ]

    cuckoo_n = [n for n in cuckoo_filter_data["N"] if n <= max_n]
    cuckoo_search = [
        cuckoo_filter_data["Search (ms)"][i]
        for i, n in enumerate(cuckoo_filter_data["N"])
        if n <= max_n
    ]

    disk_n = [n for n in disk_hashset_data["N"] if n <= max_n]
    disk_search = [
        disk_hashset_data["Search (ms)"][i]
        for i, n in enumerate(disk_hashset_data["N"])
        if n <= max_n
    ]

    # Plot with log scale for better visualization
    ax.loglog(
        linear_n,
        linear_search,
        "o-",
        label="Linear Search",
        linewidth=2,
        markersize=5,
        color="blue",
    )
    ax.loglog(
        binary_n,
        binary_search,
        "s-",
        label="Binary Search",
        linewidth=2,
        markersize=5,
        color="purple",
    )
    ax.loglog(
        bloom_n,
        bloom_search,
        "^-",
        label="Bloom Filter",
        linewidth=2,
        markersize=5,
        color="green",
    )
    ax.loglog(
        cuckoo_n,
        cuckoo_search,
        "d-",
        label="Cuckoo Filter",
        linewidth=2,
        markersize=5,
        color="red",
    )
    ax.loglog(
        disk_n,
        disk_search,
        "h-",
        label="Disk HashSet",
        linewidth=2,
        markersize=5,
        color="orange",
    )

    ax.set_xlabel("Dataset Size (N)")
    ax.set_ylabel("Search Time (ms)")
    ax.set_title("Search Time Comparison (≤100K elements)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "/home/fseddik/MSc/Fall2025/Algo/A1/report/algorithm-comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_bloom_filter_scaling():
    """Create Bloom filter scaling plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Search time scaling
    ax1.semilogx(
        bloom_filter_data["N"],
        bloom_filter_data["Search (ms)"],
        "o-",
        linewidth=2,
        markersize=5,
        color="green",
    )
    ax1.set_xlabel("Dataset Size (N)")
    ax1.set_ylabel("Search Time (ms)")
    ax1.set_title("Bloom Filter: Search Time")
    ax1.grid(True, alpha=0.3)

    # Memory usage scaling
    ax2.loglog(
        bloom_filter_data["N"],
        bloom_filter_data["Memory (MB)"],
        "s-",
        linewidth=2,
        markersize=5,
        color="darkgreen",
    )
    ax2.set_xlabel("Dataset Size (N)")
    ax2.set_ylabel("Memory Usage (MB)")
    ax2.set_title("Bloom Filter: Memory Usage")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "/home/fseddik/MSc/Fall2025/Algo/A1/report/bloom-filter-scaling.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def print_extended_comparison_table():
    """Print extended comparison table showing algorithms to their limits."""
    print("EXTENDED ALGORITHM COMPARISON TABLE:")
    print("Algorithm\t\tN=100\tN=1K\tN=10K\tN=100K\tN=1M\tN=10M\tN=100M\tN=1B")
    print("-" * 95)

    algorithms = {
        "Linear Search": linear_search_data,
        "Binary Search": binary_search_data,
        "Bloom Filter": bloom_filter_data,
        "Cuckoo Filter": cuckoo_filter_data,
        "Disk HashSet": disk_hashset_data,
    }

    target_sizes = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]

    for alg_name, data in algorithms.items():
        row = [alg_name]

        for target in target_sizes:
            if target in data["N"]:
                idx = data["N"].index(target)
                row.append(f"{data['Search (ms)'][idx]:.4f}")
            else:
                row.append("—")  # Em dash for unavailable

        print(
            f"{row[0]:<16}\t{row[1]:<6}\t{row[2]:<6}\t{row[3]:<6}\t{row[4]:<6}\t{row[5]:<6}\t{row[6]:<6}\t{row[7]:<6}\t{row[8] if len(row) > 8 else '—':<6}"
        )


if __name__ == "__main__":
    print("Generating plots for LoginChecker report...")

    # Create separate scaling plots
    create_search_time_scaling_plot()
    print("✓ Generated search-time-scaling.png")

    create_setup_time_scaling_plot()
    print("✓ Generated setup-time-scaling.png")

    create_memory_usage_scaling_plot()
    print("✓ Generated memory-usage-scaling.png")

    create_efficiency_comparison_plot()
    print("✓ Generated efficiency-comparison.png")

    # Create other plots
    create_search_time_comparison()
    print("✓ Generated algorithm-comparison.png")

    create_bloom_filter_scaling()
    print("✓ Generated bloom-filter-scaling.png")

    print("\nAll plots saved to report/ directory")

    print("\n" + "=" * 80)
    print_extended_comparison_table()
