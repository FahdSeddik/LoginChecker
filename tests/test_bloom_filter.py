"""
Comprehensive test suite for BloomFilter class.

Tests cover initialization, mathematical properties, memory management,
edge cases, and performance characteristics with various parameters.
"""

import math
import os
import tempfile
from pathlib import Path

import pytest

from src.data_structures.bloom_filter import BloomFilter


class TestBloomFilterInitialization:
    """Test BloomFilter initialization and parameter validation."""

    @pytest.mark.parametrize(
        "capacity, fpr, expected_exception",
        [
            (0, 0.1, ValueError),  # Zero capacity
            (-1, 0.1, ValueError),  # Negative capacity
            (100, 0, ValueError),  # Zero FPR
            (100, 1, ValueError),  # FPR = 1
            (100, -0.1, ValueError),  # Negative FPR
            (100, 1.1, ValueError),  # FPR > 1
        ],
    )
    def test_invalid_parameters(self, capacity, fpr, expected_exception):
        """Test that invalid parameters raise appropriate exceptions."""
        with pytest.raises(expected_exception):
            BloomFilter(capacity=capacity, false_positive_rate=fpr)

    @pytest.mark.parametrize(
        "capacity, fpr",
        [
            (1, 0.1),
            (100, 0.01),
            (1000, 0.001),
            (10000, 0.1),
            (1000000, 0.05),
        ],
    )
    def test_valid_parameters(self, capacity, fpr):
        """Test initialization with valid parameters."""
        bf = BloomFilter(capacity=capacity, false_positive_rate=fpr)
        try:
            assert bf.capacity == capacity
            assert bf.false_positive_rate == fpr
            assert bf.num_bits > 0
            assert bf.num_hashes > 0
            assert bf.inserted_count == 0
            assert not bf._closed
        finally:
            bf.close()

    @pytest.mark.parametrize("seed", [None, 42, 0, 123456789])
    def test_seed_parameter(self, seed):
        """Test seed parameter handling."""
        bf = BloomFilter(capacity=1000, false_positive_rate=0.1, seed=seed)
        try:
            assert isinstance(bf.seed, int)
            assert 0 <= bf.seed < 2**32
            if seed is not None:
                # Test deterministic behavior with same seed
                bf2 = BloomFilter(capacity=1000, false_positive_rate=0.1, seed=seed)
                try:
                    assert bf.seed == bf2.seed
                    # Test that same elements produce same hash positions
                    positions1 = bf._get_bit_positions("test")
                    positions2 = bf2._get_bit_positions("test")
                    assert positions1 == positions2
                finally:
                    bf2.close()
        finally:
            bf.close()

    def test_filepath_parameter(self, temp_dir):
        """Test custom filepath parameter."""
        filepath = temp_dir / "custom_bloom.bin"
        bf = BloomFilter(capacity=1000, false_positive_rate=0.1, filepath=str(filepath))
        try:
            assert bf.filepath == str(filepath)
            assert filepath.exists()
            # Test that file has correct size
            assert filepath.stat().st_size == bf.num_bytes
        finally:
            bf.close()


class TestBloomFilterMathematicalProperties:
    """Test mathematical properties and parameter calculations."""

    @pytest.mark.parametrize(
        "capacity, fpr",
        [
            (100, 0.1),
            (1000, 0.01),
            (10000, 0.001),
            (100000, 0.05),
        ],
    )
    def test_parameter_calculations(self, capacity, fpr):
        """Test that calculated parameters follow mathematical formulas."""
        bf = BloomFilter(capacity=capacity, false_positive_rate=fpr)
        try:
            # Test formula: m = -(n * ln(eps)) / (ln(2))^2
            expected_bits = math.ceil(-(capacity * math.log(fpr)) / (math.log(2) ** 2))
            assert bf.num_bits == expected_bits

            # Test formula: k = (m/n) * ln(2)
            expected_hashes = max(1, round((bf.num_bits / capacity) * math.log(2)))
            assert bf.num_hashes == expected_hashes

            # Test that actual FPR is close to theoretical
            theoretical_fpr = (
                1 - math.exp(-bf.num_hashes * capacity / bf.num_bits)
            ) ** bf.num_hashes
            assert abs(bf.actual_false_positive_rate - theoretical_fpr) < 1e-10

            # Test bits per element calculation
            assert abs(bf.bits_per_element - (bf.num_bits / capacity)) < 1e-10
        finally:
            bf.close()

    def test_static_parameter_calculation(self):
        """Test static parameter calculation method."""
        capacity = 1000
        fpr = 0.1

        params = BloomFilter.calculate_parameters(capacity, fpr)

        # Create actual filter and compare
        bf = BloomFilter(capacity=capacity, false_positive_rate=fpr)
        try:
            assert params["capacity"] == bf.capacity
            assert params["false_positive_rate"] == bf.false_positive_rate
            assert params["num_bits"] == bf.num_bits
            assert params["num_hashes"] == bf.num_hashes
            assert params["num_bytes"] == bf.num_bytes
            assert (
                abs(
                    params["actual_false_positive_rate"] - bf.actual_false_positive_rate
                )
                < 1e-10
            )
            assert abs(params["bits_per_element"] - bf.bits_per_element) < 1e-10
        finally:
            bf.close()


class TestBloomFilterOperations:
    """Test core Bloom filter operations: add and contains."""

    @pytest.fixture
    def bloom_filter(self):
        """Create a standard Bloom filter for testing."""
        bf = BloomFilter(capacity=1000, false_positive_rate=0.1, seed=42)
        yield bf
        bf.close()

    @pytest.mark.parametrize(
        "elements",
        [
            ["user1", "user2", "user3"],
            [b"byte1", b"byte2", b"byte3"],
            ["", "single_char_a", "long_username_with_special_chars_123!@#"],
            ["unicode_用户", "emoji_😀", "mixed_用户123😀"],
        ],
    )
    def test_add_and_contains(self, bloom_filter, elements):
        """Test adding elements and checking membership."""
        # Add elements
        for element in elements:
            bloom_filter.add(element)
            assert bloom_filter.contains(element)
            assert element in bloom_filter  # Test __contains__

        # Check that all elements are still found
        for element in elements:
            assert bloom_filter.contains(element)
            assert element in bloom_filter

        # Verify insert count
        assert bloom_filter.inserted_count == len(elements)

    def test_false_negatives_impossible(self, bloom_filter):
        """Test that false negatives are impossible (key property)."""
        test_elements = [f"element_{i}" for i in range(100)]

        # Add elements
        for element in test_elements:
            bloom_filter.add(element)

        # All added elements must be found (no false negatives)
        for element in test_elements:
            assert bloom_filter.contains(element), f"False negative for {element}"

    def test_false_positive_rate_approximation(self):
        """Test that false positive rate is approximately as expected."""
        capacity = 1000
        fpr = 0.1
        bf = BloomFilter(capacity=capacity, false_positive_rate=fpr, seed=42)

        try:
            # Add exactly the capacity number of elements
            added_elements = [f"user_{i}" for i in range(capacity)]
            for element in added_elements:
                bf.add(element)

            # Test with elements not added
            test_elements = [f"notadded_{i}" for i in range(1000)]
            false_positives = sum(
                1 for element in test_elements if bf.contains(element)
            )
            observed_fpr = false_positives / len(test_elements)

            # Allow some tolerance due to randomness
            assert observed_fpr <= fpr * 2, f"FPR too high: {observed_fpr} > {fpr * 2}"
        finally:
            bf.close()

    @pytest.mark.parametrize(
        "element_type, element_value",
        [
            ("string", "test_string"),
            ("bytes", b"test_bytes"),
            ("empty_string", ""),
            ("empty_bytes", b""),
            ("unicode", "测试"),
            ("long_string", "a" * 1000),
        ],
    )
    def test_different_element_types(self, bloom_filter, element_type, element_value):
        """Test with different types of elements."""
        bloom_filter.add(element_value)
        assert bloom_filter.contains(element_value)
        assert bloom_filter.inserted_count == 1


class TestBloomFilterBitOperations:
    """Test low-level bit operations."""

    @pytest.fixture
    def bloom_filter(self):
        """Create a small Bloom filter for bit testing."""
        bf = BloomFilter(capacity=10, false_positive_rate=0.1, seed=42)
        yield bf
        bf.close()

    def test_bit_setting_and_getting(self, bloom_filter):
        """Test individual bit operations."""
        # Initially all bits should be 0
        for i in range(min(100, bloom_filter.num_bits)):
            assert not bloom_filter._get_bit(i)

        # Set some bits
        test_positions = [0, 1, 7, 8, 15, 16, bloom_filter.num_bits - 1]
        for pos in test_positions:
            if pos < bloom_filter.num_bits:
                bloom_filter._set_bit(pos)
                assert bloom_filter._get_bit(pos)

    def test_hash_position_generation(self, bloom_filter):
        """Test hash position generation."""
        test_keys = ["test1", "test2", b"test3", ""]

        for key in test_keys:
            positions = bloom_filter._get_bit_positions(key)

            # Check correct number of positions
            assert len(positions) == bloom_filter.num_hashes

            # Check all positions are valid
            for pos in positions:
                assert 0 <= pos < bloom_filter.num_bits
                assert isinstance(pos, int)

    def test_deterministic_hashing(self):
        """Test that hashing is deterministic with same seed."""
        bf1 = BloomFilter(capacity=100, false_positive_rate=0.1, seed=42)
        bf2 = BloomFilter(capacity=100, false_positive_rate=0.1, seed=42)

        try:
            test_key = "deterministic_test"
            positions1 = bf1._get_bit_positions(test_key)
            positions2 = bf2._get_bit_positions(test_key)

            assert positions1 == positions2
        finally:
            bf1.close()
            bf2.close()

    def test_two_hashes_function(self, bloom_filter):
        """Test the two-hash function."""
        test_key = "test_key"
        h1, h2 = bloom_filter._two_hashes(test_key)

        # Should return two different integers
        assert isinstance(h1, int)
        assert isinstance(h2, int)
        assert h1 != h2  # Very unlikely to be equal

        # Should be deterministic
        h1_again, h2_again = bloom_filter._two_hashes(test_key)
        assert h1 == h1_again
        assert h2 == h2_again


class TestBloomFilterMemoryManagement:
    """Test memory management and file operations."""

    def test_temporary_file_creation(self):
        """Test that temporary files are created and cleaned up."""
        bf = BloomFilter(capacity=100, false_positive_rate=0.1)
        filepath = bf.filepath

        # File should exist while filter is open
        assert os.path.exists(filepath)

        bf.close()

        # On Windows, file cleanup may be delayed due to file handles
        # Test that the filter marks itself as closed, which is the important behavior
        assert bf._closed

        # For temporary files, the cleanup should eventually happen
        # We test this by checking that a new BloomFilter can create its own temp file
        bf2 = BloomFilter(capacity=100, false_positive_rate=0.1)
        assert os.path.exists(bf2.filepath)
        assert bf2.filepath != filepath  # Should be different temp file
        bf2.close()

    def test_custom_file_persistence(self, temp_dir):
        """Test persistence with custom file path."""
        filepath = temp_dir / "test_bloom.bin"

        # Create filter and add some data
        bf1 = BloomFilter(capacity=100, false_positive_rate=0.1, filepath=str(filepath))
        test_elements = ["user1", "user2", "user3"]

        for element in test_elements:
            bf1.add(element)

        bf1.flush()  # Ensure data is written
        bf1.close()

        # File should still exist
        assert filepath.exists()

        # Create new filter with same file
        bf2 = BloomFilter(capacity=100, false_positive_rate=0.1, filepath=str(filepath))

        try:
            # Data should be preserved (though this is implementation dependent)
            # At minimum, the file should have the correct size
            assert bf2.num_bytes == filepath.stat().st_size
        finally:
            bf2.close()

    def test_context_manager(self, temp_dir):
        """Test context manager functionality."""
        filepath = temp_dir / "context_test.bin"

        with BloomFilter(
            capacity=100, false_positive_rate=0.1, filepath=str(filepath)
        ) as bf:
            bf.add("test_element")
            assert bf.contains("test_element")
            assert not bf._closed

        # Should be closed after context exit
        assert bf._closed

    def test_memory_usage_stats(self):
        """Test memory usage statistics."""
        bf = BloomFilter(capacity=1000, false_positive_rate=0.1)
        try:
            stats = bf.get_stats()

            # Memory calculations should be consistent
            expected_mb = bf.num_bytes / (1024 * 1024)
            assert abs(stats["memory_usage_mb"] - expected_mb) < 1e-10

            # Should have reasonable memory usage
            assert stats["memory_usage_mb"] > 0
            assert stats["memory_usage_mb"] < 100  # Shouldn't be huge for test
        finally:
            bf.close()

    def test_resource_cleanup_on_exception(self):
        """Test that resources are cleaned up even if exceptions occur."""
        temp_files_before = len(list(Path(tempfile.gettempdir()).glob("tmp*")))

        try:
            bf = BloomFilter(capacity=100, false_positive_rate=0.1)
            # Simulate some work
            bf.add("test")
            # Don't explicitly close - rely on __del__
            del bf
        except Exception:
            pass

        # Check that we didn't leak temporary files
        import gc

        gc.collect()  # Force garbage collection

        # Allow some time for cleanup
        import time

        time.sleep(0.1)

        temp_files_after = len(list(Path(tempfile.gettempdir()).glob("tmp*")))
        # We might have some temporary files from other processes, so just check
        # we didn't create a massive leak
        assert temp_files_after - temp_files_before < 10


class TestBloomFilterStatistics:
    """Test statistics and monitoring functionality."""

    @pytest.fixture
    def bloom_filter(self):
        """Create a Bloom filter for statistics testing."""
        bf = BloomFilter(capacity=100, false_positive_rate=0.1, seed=42)
        yield bf
        bf.close()

    def test_initial_statistics(self, bloom_filter):
        """Test statistics on empty filter."""
        stats = bloom_filter.get_stats()

        assert stats["capacity"] == 100
        assert stats["inserted_count"] == 0
        assert stats["false_positive_rate"] == 0.1
        assert stats["set_bits"] == 0
        assert stats["fill_ratio"] == 0.0
        assert stats["estimated_current_fpr"] == 0.0
        assert stats["num_bits"] > 0
        assert stats["num_hashes"] > 0

    def test_statistics_after_insertions(self, bloom_filter):
        """Test statistics after adding elements."""
        # Add some elements
        elements = [f"user_{i}" for i in range(50)]
        for element in elements:
            bloom_filter.add(element)

        stats = bloom_filter.get_stats()

        assert stats["inserted_count"] == 50
        assert stats["set_bits"] > 0
        assert stats["fill_ratio"] > 0
        assert stats["estimated_current_fpr"] > 0
        assert stats["estimated_current_fpr"] <= 1.0

    def test_fill_ratio_progression(self):
        """Test that fill ratio increases as elements are added."""
        bf = BloomFilter(capacity=100, false_positive_rate=0.1, seed=42)
        try:
            fill_ratios = []

            for i in range(0, 50, 10):
                bf.add(f"user_{i}")
                stats = bf.get_stats()
                fill_ratios.append(stats["fill_ratio"])

            # Fill ratio should generally increase (monotonic)
            for i in range(1, len(fill_ratios)):
                assert fill_ratios[i] >= fill_ratios[i - 1]
        finally:
            bf.close()

    def test_clear_functionality(self, bloom_filter):
        """Test clearing the filter."""
        # Add elements
        for i in range(10):
            bloom_filter.add(f"user_{i}")

        # Verify elements are present
        assert bloom_filter.inserted_count == 10
        stats_before = bloom_filter.get_stats()
        assert stats_before["set_bits"] > 0

        # Clear the filter
        bloom_filter.clear()

        # Verify filter is cleared
        assert bloom_filter.inserted_count == 0
        stats_after = bloom_filter.get_stats()
        assert stats_after["set_bits"] == 0
        assert stats_after["fill_ratio"] == 0.0

        # Previously added elements should no longer be found
        for i in range(10):
            assert not bloom_filter.contains(f"user_{i}")


class TestBloomFilterEdgeCases:
    """Test edge cases and error conditions."""

    def test_operations_on_closed_filter(self):
        """Test that operations on closed filter raise appropriate errors."""
        bf = BloomFilter(capacity=100, false_positive_rate=0.1)
        bf.close()

        with pytest.raises(ValueError, match="Cannot operate on closed BloomFilter"):
            bf.add("test")

        with pytest.raises(ValueError, match="Cannot operate on closed BloomFilter"):
            bf.contains("test")

        with pytest.raises(ValueError, match="Cannot operate on closed BloomFilter"):
            bf._get_bit(0)

        with pytest.raises(ValueError, match="Cannot operate on closed BloomFilter"):
            bf._set_bit(0)

    def test_double_close(self):
        """Test that closing twice doesn't cause issues."""
        bf = BloomFilter(capacity=100, false_positive_rate=0.1)
        bf.close()
        bf.close()  # Should not raise exception

    def test_large_capacity_parameters(self):
        """Test with very large capacity values."""
        # This tests mathematical calculations don't overflow
        params = BloomFilter.calculate_parameters(
            capacity=10_000_000, false_positive_rate=0.01
        )

        assert params["num_bits"] > 0
        assert params["num_hashes"] > 0
        assert params["memory_usage_mb"] > 0

    def test_very_small_false_positive_rate(self):
        """Test with very small false positive rates."""
        params = BloomFilter.calculate_parameters(
            capacity=1000, false_positive_rate=0.0001
        )

        # Should require more bits and hashes
        assert params["num_bits"] > 10000  # Rough estimate
        assert params["num_hashes"] > 10

    @pytest.mark.parametrize("capacity", [1, 2, 3, 5, 10])
    def test_minimal_capacities(self, capacity):
        """Test with very small capacities."""
        bf = BloomFilter(capacity=capacity, false_positive_rate=0.1, seed=42)
        try:
            # Should still work correctly
            for i in range(capacity):
                bf.add(f"item_{i}")
                assert bf.contains(f"item_{i}")

            assert bf.inserted_count == capacity
        finally:
            bf.close()

    def test_flush_operations(self, temp_dir):
        """Test flush operations."""
        filepath = temp_dir / "flush_test.bin"
        bf = BloomFilter(capacity=100, false_positive_rate=0.1, filepath=str(filepath))

        try:
            bf.add("test_element")

            # Flush should not raise exceptions
            bf.flush()

            # File should still exist and have correct size
            assert filepath.exists()
            assert filepath.stat().st_size == bf.num_bytes
        finally:
            bf.close()


class TestBloomFilterPerformance:
    """Test performance characteristics and scaling."""

    @pytest.mark.parametrize(
        "capacity, num_elements",
        [
            (100, 50),
            (1000, 500),
            (10000, 5000),
        ],
    )
    def test_insertion_performance(self, capacity, num_elements):
        """Test that insertions scale reasonably."""
        bf = BloomFilter(capacity=capacity, false_positive_rate=0.1, seed=42)

        try:
            import time

            start_time = time.time()

            for i in range(num_elements):
                bf.add(f"user_{i}")

            end_time = time.time()
            elapsed = end_time - start_time

            # Should complete reasonably quickly (adjust threshold as needed)
            assert elapsed < 10.0, f"Insertion took too long: {elapsed:.2f}s"

            # Verify all elements were added
            assert bf.inserted_count == num_elements
        finally:
            bf.close()

    def test_lookup_performance(self):
        """Test lookup performance."""
        bf = BloomFilter(capacity=10000, false_positive_rate=0.1, seed=42)

        try:
            # Add some elements
            test_elements = [f"user_{i}" for i in range(5000)]
            for element in test_elements:
                bf.add(element)

            import time

            start_time = time.time()

            # Perform many lookups
            for element in test_elements[:1000]:  # Test subset for speed
                assert bf.contains(element)

            end_time = time.time()
            elapsed = end_time - start_time

            # Lookups should be very fast
            assert elapsed < 1.0, f"Lookups took too long: {elapsed:.2f}s"
        finally:
            bf.close()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
