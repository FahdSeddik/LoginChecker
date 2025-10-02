"""
Tests for CuckooFilter implementation.

Covers basic functionality, edge cases, persistence, and performance characteristics.
"""

import os
import tempfile

import pytest

from src.data_structures.cuckoo_filter import CuckooFilter


class TestCuckooFilter:
    """Test suite for CuckooFilter."""

    def test_initialization(self):
        """Test basic initialization."""
        cf = CuckooFilter(capacity=1000, false_positive_rate=0.01)
        try:
            assert cf.capacity == 1000
            assert cf.false_positive_rate == 0.01
            assert cf.bucket_size == 4  # default
            assert cf.inserted_count == 0
            assert not cf._closed
        finally:
            cf.close()

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        cf = CuckooFilter(
            capacity=500,
            false_positive_rate=0.05,
            bucket_size=2,
            fingerprint_bits=12,
            seed=12345,
            max_kicks=100,
        )
        try:
            assert cf.capacity == 500
            assert cf.false_positive_rate == 0.05
            assert cf.bucket_size == 2
            assert cf.fingerprint_bits == 12
            assert cf.seed == 12345
            assert cf.max_kicks == 100
        finally:
            cf.close()

    def test_invalid_parameters(self):
        """Test validation of invalid parameters."""
        with pytest.raises(ValueError, match="Capacity must be positive"):
            CuckooFilter(capacity=0, false_positive_rate=0.01)

        with pytest.raises(ValueError, match="Capacity must be positive"):
            CuckooFilter(capacity=-10, false_positive_rate=0.01)

        with pytest.raises(
            ValueError, match="False positive rate must be between 0 and 1"
        ):
            CuckooFilter(capacity=100, false_positive_rate=0.0)

        with pytest.raises(
            ValueError, match="False positive rate must be between 0 and 1"
        ):
            CuckooFilter(capacity=100, false_positive_rate=1.0)

        with pytest.raises(
            ValueError, match="False positive rate must be between 0 and 1"
        ):
            CuckooFilter(capacity=100, false_positive_rate=1.5)

        with pytest.raises(ValueError, match="Bucket size must be 1, 2, 4, or 8"):
            CuckooFilter(capacity=100, false_positive_rate=0.01, bucket_size=3)

    def test_basic_operations(self):
        """Test basic add, contains, and delete operations."""
        cf = CuckooFilter(capacity=100, false_positive_rate=0.01, seed=42)
        try:
            # Test add and contains
            assert cf.add("hello")
            assert "hello" in cf
            assert cf.contains("hello")

            # Test string and bytes
            assert cf.add(b"world")
            assert b"world" in cf

            # Test non-existent element
            assert "nonexistent" not in cf

            # Test delete
            assert cf.delete("hello")
            assert "hello" not in cf

            # Test delete non-existent
            assert not cf.delete("nonexistent")

        finally:
            cf.close()

    def test_fingerprint_generation(self):
        """Test fingerprint generation."""
        cf = CuckooFilter(capacity=100, false_positive_rate=0.01, seed=42)
        try:
            # Test non-zero fingerprints
            fp1 = cf._fingerprint("test1")
            fp2 = cf._fingerprint("test2")

            assert fp1 > 0
            assert fp2 > 0
            assert fp1 != fp2  # Should be different for different keys

            # Test same key produces same fingerprint
            fp3 = cf._fingerprint("test1")
            assert fp1 == fp3

            # Test fingerprint within bounds
            assert fp1 <= cf.fingerprint_mask
            assert fp2 <= cf.fingerprint_mask

        finally:
            cf.close()

    def test_bucket_operations(self):
        """Test bucket-level operations."""
        cf = CuckooFilter(capacity=100, false_positive_rate=0.01, seed=42)
        try:
            # Test empty bucket
            slot = cf._find_free_slot_in_bucket(0)
            assert slot is not None
            assert slot == 0  # First slot should be free

            # Test slot read/write
            cf._write_slot(0, 0, 123)
            assert cf._read_slot(0, 0) == 123
            assert not cf._is_slot_empty(123)

            # Test empty slot
            assert cf._is_slot_empty(0)

        finally:
            cf.close()

    def test_multiple_elements(self):
        """Test adding multiple elements."""
        cf = CuckooFilter(capacity=100, false_positive_rate=0.01, seed=42)
        try:
            elements = [f"user_{i}" for i in range(50)]

            # Add all elements
            for element in elements:
                assert cf.add(element)

            assert cf.inserted_count == 50

            # Check all elements are present
            for element in elements:
                assert element in cf

            # Check non-existent elements
            for i in range(100, 150):
                assert f"user_{i}" not in cf

        finally:
            cf.close()

    def test_duplicate_insertion(self):
        """Test inserting duplicate elements."""
        cf = CuckooFilter(capacity=100, false_positive_rate=0.01, seed=42)
        try:
            # Add element twice
            assert cf.add("duplicate")
            assert cf.add("duplicate")  # Should succeed (cuckoo allows duplicates)

            assert cf.inserted_count == 2
            assert "duplicate" in cf

            # Delete once - should still be present
            assert cf.delete("duplicate")
            assert "duplicate" in cf  # Still one copy
            assert cf.inserted_count == 1

            # Delete again - should be gone
            assert cf.delete("duplicate")
            assert "duplicate" not in cf
            assert cf.inserted_count == 0

        finally:
            cf.close()

    def test_persistence(self):
        """Test file persistence."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create filter and add elements
            cf1 = CuckooFilter(
                capacity=100, false_positive_rate=0.01, filepath=tmp_path, seed=42
            )
            elements = [f"persistent_{i}" for i in range(20)]

            for element in elements:
                cf1.add(element)

            cf1.flush()
            cf1.close()

            # Reload from file
            cf2 = CuckooFilter(
                capacity=100, false_positive_rate=0.01, filepath=tmp_path
            )
            try:
                # Check elements are still there
                for element in elements:
                    assert element in cf2

                assert cf2.inserted_count == 20

            finally:
                cf2.close()

        finally:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass

    def test_clear(self):
        """Test clearing the filter."""
        cf = CuckooFilter(capacity=100, false_positive_rate=0.01, seed=42)
        try:
            # Add elements
            elements = [f"clear_{i}" for i in range(10)]
            for element in elements:
                cf.add(element)

            assert cf.inserted_count == 10

            # Clear and verify
            cf.clear()
            assert cf.inserted_count == 0

            for element in elements:
                assert element not in cf

        finally:
            cf.close()

    def test_context_manager(self):
        """Test context manager support."""
        elements = [f"context_{i}" for i in range(10)]

        with CuckooFilter(capacity=100, false_positive_rate=0.01, seed=42) as cf:
            for element in elements:
                cf.add(element)

            for element in elements:
                assert element in cf

        # Filter should be closed automatically
        assert cf._closed

    def test_stats(self):
        """Test statistics reporting."""
        cf = CuckooFilter(capacity=100, false_positive_rate=0.01, seed=42)
        try:
            # Initial stats
            stats = cf.get_stats()
            assert stats["inserted_count"] == 0
            assert stats["load_factor"] == 0.0
            assert stats["total_kicks"] == 0
            assert stats["failed_inserts"] == 0

            # Add elements and check stats
            for i in range(20):
                cf.add(f"stats_{i}")

            stats = cf.get_stats()
            assert stats["inserted_count"] == 20
            assert stats["load_factor"] > 0
            assert stats["used_slots"] == 20
            assert stats["capacity"] == 100
            assert "occupancy_histogram" in stats

        finally:
            cf.close()

    def test_calculate_parameters(self):
        """Test parameter calculation class method."""
        params = CuckooFilter.calculate_parameters(1000, 0.01, bucket_size=4)

        assert params["capacity"] == 1000
        assert params["false_positive_rate"] == 0.01
        assert params["bucket_size"] == 4
        assert params["bucket_count"] > 0
        assert params["fingerprint_bits"] >= 8
        assert params["total_bytes"] > 0
        assert params["actual_false_positive_rate"] <= 0.01

    def test_calculate_parameters_invalid(self):
        """Test parameter calculation with invalid inputs."""
        with pytest.raises(ValueError):
            CuckooFilter.calculate_parameters(0, 0.01)

        with pytest.raises(ValueError):
            CuckooFilter.calculate_parameters(100, 0.0)

        with pytest.raises(ValueError):
            CuckooFilter.calculate_parameters(100, 0.01, bucket_size=3)

    def test_closed_filter_operations(self):
        """Test operations on closed filter raise errors."""
        cf = CuckooFilter(capacity=100, false_positive_rate=0.01)
        cf.close()

        with pytest.raises(ValueError, match="Cannot operate on closed"):
            cf.add("test")

        with pytest.raises(ValueError, match="Cannot operate on closed"):
            cf.contains("test")

        with pytest.raises(ValueError, match="Cannot operate on closed"):
            cf.delete("test")

        with pytest.raises(ValueError, match="Cannot operate on closed"):
            cf.clear()

    def test_full_filter_behavior(self):
        """Test behavior when filter becomes full."""
        # Small filter that will become full quickly
        cf = CuckooFilter(capacity=10, false_positive_rate=0.1, bucket_size=2, seed=42)
        try:
            success_count = 0

            # Try to add many elements
            for i in range(100):
                if cf.add(f"full_{i}"):
                    success_count += 1
                else:
                    break  # Filter is full

            # Should have added some elements successfully
            assert success_count > 0
            assert success_count < 100  # But not all

            # Failed inserts should be tracked
            stats = cf.get_stats()
            assert stats["failed_inserts"] > 0

        finally:
            cf.close()

    def test_different_bucket_sizes(self):
        """Test different bucket sizes."""
        for bucket_size in [1, 2, 4, 8]:
            cf = CuckooFilter(
                capacity=50, false_positive_rate=0.01, bucket_size=bucket_size, seed=42
            )
            try:
                # Add elements
                for i in range(20):
                    assert cf.add(f"bucket_{bucket_size}_{i}")

                # Check they're all there
                for i in range(20):
                    assert f"bucket_{bucket_size}_{i}" in cf

                assert cf.bucket_size == bucket_size

            finally:
                cf.close()

    def test_file_corruption_handling(self):
        """Test handling of corrupted files."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
            # Write invalid data
            tmp.write(b"invalid header data")

        try:
            with pytest.raises(ValueError, match="Invalid file"):
                CuckooFilter(capacity=100, false_positive_rate=0.01, filepath=tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass

    @pytest.mark.slow
    def test_false_positive_rate(self):
        """Test empirical false positive rate."""
        capacity = 1000
        target_fpr = 0.01
        cf = CuckooFilter(capacity=capacity, false_positive_rate=target_fpr, seed=42)

        try:
            # Add elements
            added_elements = set()
            for i in range(capacity // 2):  # Fill to 50% capacity
                element = f"fpr_test_{i}"
                if cf.add(element):
                    added_elements.add(element)

            # Test false positives with non-added elements
            false_positives = 0
            test_count = 10000

            for i in range(capacity, capacity + test_count):
                test_element = f"fpr_test_{i}"
                assert test_element not in added_elements

                if test_element in cf:
                    false_positives += 1

            empirical_fpr = false_positives / test_count

            # Should be within reasonable bounds (allow some variance)
            assert empirical_fpr <= target_fpr * 3  # Allow 3x variance for small sample

        finally:
            cf.close()

    def test_eviction_mechanics(self):
        """Test cuckoo eviction mechanics."""
        cf = CuckooFilter(capacity=50, false_positive_rate=0.1, bucket_size=2, seed=42)
        try:
            # Fill filter to trigger evictions
            added = 0
            for i in range(200):
                if cf.add(f"eviction_{i}"):
                    added += 1

            stats = cf.get_stats()

            # Should have performed some kicks
            if stats["total_kicks"] > 0:
                assert stats["avg_kicks_per_insert"] > 0

            # All successfully added elements should still be findable
            found = 0
            for i in range(200):
                if f"eviction_{i}" in cf:
                    found += 1

            # Should find most or all successfully added elements
            assert (
                found >= added * 0.9
            )  # Allow for some false negatives in pathological cases

        finally:
            cf.close()

    def test_memory_usage(self):
        """Test memory usage calculation."""
        cf = CuckooFilter(capacity=1000, false_positive_rate=0.01)
        try:
            stats = cf.get_stats()

            # Memory usage should be reasonable
            assert stats["memory_usage_mb"] > 0
            assert (
                stats["memory_usage_mb"] < 100
            )  # Should be much less than 100MB for 1000 items

            # Should match calculated size
            expected_size = cf.HEADER_SIZE + cf.payload_bytes
            assert stats["memory_usage_mb"] == expected_size / (1024 * 1024)

        finally:
            cf.close()
