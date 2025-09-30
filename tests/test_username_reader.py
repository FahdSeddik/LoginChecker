"""
Comprehensive tests for UsernameReader class.

Tests cover random access, iteration, limits, error handling, and memory efficiency.
"""

import os
import struct
import tempfile
from pathlib import Path

import pytest

from data.generate import EfficientUsernameStorage, UsernameGenerator
from data.reader import UsernameReader


class TestUsernameReader:
    """Test suite for UsernameReader class."""

    def setup_method(self):
        """Set up test fixtures with sample data."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_filepath = Path(self.temp_dir) / "test_usernames.dat"

        # Create test data
        self.test_usernames = [
            "alice",
            "bob",
            "charlie",
            "diana",
            "eve",
            "frank",
            "grace",
            "henry",
            "iris",
            "jack",
        ]
        self._create_test_files()

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up files
        for file in Path(self.temp_dir).glob("*"):
            try:
                file.unlink()
            except Exception:
                pass
        try:
            os.rmdir(self.temp_dir)
        except Exception:
            pass

    def _create_test_files(self):
        """Create test data and index files manually."""
        # Create data file
        with open(self.test_filepath, "wb") as data_file:
            positions = []
            current_pos = 0

            for username in self.test_usernames:
                username_bytes = username.encode("utf-8")
                positions.append(current_pos)
                data_file.write(username_bytes)
                current_pos += len(username_bytes)

        # Create index file
        with open(self.test_filepath.with_suffix(".idx"), "wb") as index_file:
            for pos in positions:
                index_file.write(struct.pack("<Q", pos))

    def test_init_basic(self):
        """Test basic initialization."""
        reader = UsernameReader(self.test_filepath)

        assert reader.filepath == self.test_filepath
        assert reader.index_filepath == self.test_filepath.with_suffix(".idx")
        assert len(reader) == len(self.test_usernames)
        assert reader.total_count == len(self.test_usernames)

        reader.close()

    def test_init_with_limit(self):
        """Test initialization with limit."""
        limit = 5
        reader = UsernameReader(self.test_filepath, limit=limit)

        assert len(reader) == limit
        assert reader.effective_count == limit
        assert reader.get_limit() == limit

        reader.close()

    def test_init_file_not_found(self):
        """Test initialization with non-existent files."""
        non_existent = Path(self.temp_dir) / "nonexistent.dat"

        with pytest.raises(FileNotFoundError):
            UsernameReader(non_existent)

    def test_init_index_file_not_found(self):
        """Test initialization when index file is missing."""
        # Remove index file
        self.test_filepath.with_suffix(".idx").unlink()

        with pytest.raises(FileNotFoundError):
            UsernameReader(self.test_filepath)

    def test_random_access_basic(self):
        """Test basic random access functionality."""
        with UsernameReader(self.test_filepath) as reader:
            # Test forward indexing
            assert reader[0] == "alice"
            assert reader[4] == "eve"
            assert reader[9] == "jack"

            # Test negative indexing
            assert reader[-1] == "jack"
            assert reader[-2] == "iris"

    def test_random_access_out_of_bounds(self):
        """Test random access with invalid indices."""
        with UsernameReader(self.test_filepath) as reader:
            with pytest.raises(IndexError):
                _ = reader[100]

            with pytest.raises(IndexError):
                _ = reader[-100]

    def test_random_access_with_limit(self):
        """Test random access with limit applied."""
        limit = 5
        with UsernameReader(self.test_filepath, limit=limit) as reader:
            assert reader[0] == "alice"
            assert reader[4] == "eve"

            # Should not be able to access beyond limit
            with pytest.raises(IndexError):
                _ = reader[5]

    def test_iteration_basic(self):
        """Test basic iteration functionality."""
        with UsernameReader(self.test_filepath) as reader:
            usernames = list(reader)
            assert usernames == self.test_usernames

    def test_iteration_with_limit(self):
        """Test iteration with limit applied."""
        limit = 7
        with UsernameReader(self.test_filepath, limit=limit) as reader:
            usernames = list(reader)
            assert usernames == self.test_usernames[:limit]

    def test_iter_usernames_range(self):
        """Test iteration with specific range."""
        with UsernameReader(self.test_filepath) as reader:
            usernames = list(reader.iter_usernames(2, 6))
            assert usernames == self.test_usernames[2:6]

    def test_iter_usernames_invalid_range(self):
        """Test iteration with invalid ranges."""
        with UsernameReader(self.test_filepath) as reader:
            with pytest.raises(ValueError):
                list(reader.iter_usernames(-1, 5))

            with pytest.raises(ValueError):
                list(reader.iter_usernames(5, 3))  # start > end

            with pytest.raises(ValueError):
                list(reader.iter_usernames(0, 100))  # end > count

    def test_iter_batch_basic(self):
        """Test batch iteration functionality."""
        with UsernameReader(self.test_filepath) as reader:
            batches = list(reader.iter_batch(3, 0, 7))

            assert len(batches) == 3  # 3, 3, 1
            assert batches[0] == self.test_usernames[0:3]
            assert batches[1] == self.test_usernames[3:6]
            assert batches[2] == self.test_usernames[6:7]

    def test_iter_batch_larger_than_data(self):
        """Test batch iteration with batch size larger than data."""
        with UsernameReader(self.test_filepath) as reader:
            batches = list(reader.iter_batch(100))

            assert len(batches) == 1
            assert batches[0] == self.test_usernames

    def test_search_functionality(self):
        """Test search functionality."""
        with UsernameReader(self.test_filepath) as reader:
            # Case sensitive search
            results = list(reader.search("a", case_sensitive=True))
            expected = [
                (i, name) for i, name in enumerate(self.test_usernames) if "a" in name
            ]
            assert results == expected

            # Case insensitive search
            results = list(reader.search("A", case_sensitive=False))
            expected = [
                (i, name)
                for i, name in enumerate(self.test_usernames)
                if "a" in name.lower()
            ]
            assert results == expected

    def test_search_no_matches(self):
        """Test search with no matches."""
        with UsernameReader(self.test_filepath) as reader:
            results = list(reader.search("xyz"))
            assert results == []

    def test_search_with_limit(self):
        """Test search functionality with limit."""
        limit = 5
        with UsernameReader(self.test_filepath, limit=limit) as reader:
            results = list(reader.search("a"))
            # Should only search within limited range
            assert all(idx < limit for idx, _ in results)

    def test_limit_operations(self):
        """Test limit setting and clearing operations."""
        with UsernameReader(self.test_filepath) as reader:
            # Initial state
            assert reader.get_limit() is None
            assert len(reader) == len(self.test_usernames)

            # Set limit
            reader.set_limit(5)
            assert reader.get_limit() == 5
            assert len(reader) == 5

            # Clear limit
            reader.clear_limit()
            assert reader.get_limit() is None
            assert len(reader) == len(self.test_usernames)

            # Set limit to None explicitly
            reader.set_limit(None)
            assert reader.get_limit() is None

    def test_limit_validation(self):
        """Test limit validation."""
        with UsernameReader(self.test_filepath) as reader:
            with pytest.raises(ValueError):
                reader.set_limit(-1)

    def test_get_stats(self):
        """Test statistics functionality."""
        with UsernameReader(self.test_filepath) as reader:
            stats = reader.get_stats()

            # Verify expected fields
            required_fields = {
                "total_usernames",
                "effective_usernames",
                "limit",
                "data_file_size",
                "index_file_size",
                "total_size",
                "avg_username_length",
                "data_file_size_gb",
                "index_file_size_mb",
                "total_size_gb",
                "index_entry_size",
                "index_entries",
            }
            assert set(stats.keys()) == required_fields

            # Verify values
            assert stats["total_usernames"] == len(self.test_usernames)
            assert stats["effective_usernames"] == len(self.test_usernames)
            assert stats["limit"] is None
            assert stats["index_entry_size"] == 8
            assert stats["index_entries"] == len(self.test_usernames)

    def test_get_stats_with_limit(self):
        """Test statistics with limit applied."""
        limit = 5
        with UsernameReader(self.test_filepath, limit=limit) as reader:
            stats = reader.get_stats()

            assert stats["total_usernames"] == len(self.test_usernames)
            assert stats["effective_usernames"] == limit
            assert stats["limit"] == limit

    def test_sample_basic(self):
        """Test sampling functionality."""
        with UsernameReader(self.test_filepath) as reader:
            sample = reader.sample(5)

            assert len(sample) == 5
            assert all(username in self.test_usernames for username in sample)

    def test_sample_larger_than_data(self):
        """Test sampling when count exceeds data size."""
        with UsernameReader(self.test_filepath) as reader:
            sample = reader.sample(100)

            assert len(sample) == len(self.test_usernames)

    def test_sample_with_limit(self):
        """Test sampling with limit applied."""
        limit = 5
        with UsernameReader(self.test_filepath, limit=limit) as reader:
            sample = reader.sample(3)

            assert len(sample) == 3
            # Should only sample from limited range
            assert all(username in self.test_usernames[:limit] for username in sample)

    def test_context_manager(self):
        """Test context manager functionality."""
        with UsernameReader(self.test_filepath) as reader:
            assert reader[0] == "alice"

        # Reader should be closed after context exit
        # Note: We can't easily test this without accessing private attributes

    def test_manual_close(self):
        """Test manual close functionality."""
        reader = UsernameReader(self.test_filepath)
        assert reader[0] == "alice"

        reader.close()
        # After closing, operations should still work due to mmap
        # but resources should be cleaned up

    def test_unicode_handling(self):
        """Test handling of Unicode usernames."""
        # Create test data with Unicode
        unicode_usernames = ["alice", "böb", "charlie_😀", "диана", "eve"]
        unicode_filepath = Path(self.temp_dir) / "unicode_test.dat"

        # Create files with Unicode content
        with open(unicode_filepath, "wb") as data_file:
            positions = []
            current_pos = 0

            for username in unicode_usernames:
                username_bytes = username.encode("utf-8")
                positions.append(current_pos)
                data_file.write(username_bytes)
                current_pos += len(username_bytes)

        with open(unicode_filepath.with_suffix(".idx"), "wb") as index_file:
            for pos in positions:
                index_file.write(struct.pack("<Q", pos))

        # Test reading Unicode data
        with UsernameReader(unicode_filepath) as reader:
            assert len(reader) == len(unicode_usernames)
            for i, expected in enumerate(unicode_usernames):
                assert reader[i] == expected

    def test_empty_file(self):
        """Test handling of empty files."""
        empty_filepath = Path(self.temp_dir) / "empty.dat"

        # Create empty files
        with open(empty_filepath, "wb"):
            pass
        with open(empty_filepath.with_suffix(".idx"), "wb"):
            pass

        with UsernameReader(empty_filepath) as reader:
            assert len(reader) == 0
            assert reader.total_count == 0

            with pytest.raises(IndexError):
                _ = reader[0]

    def test_large_indices(self):
        """Test handling of large file positions."""
        # Create a file with artificially large positions to test 64-bit handling
        large_filepath = Path(self.temp_dir) / "large.dat"

        # Create minimal data file
        with open(large_filepath, "wb") as f:
            f.write(b"test")

        # Create index with large position
        large_position = 2**32 + 1000  # Larger than 32-bit
        with open(large_filepath.with_suffix(".idx"), "wb") as f:
            f.write(struct.pack("<Q", 0))  # First at position 0
            f.write(struct.pack("<Q", large_position))  # Second at large position

        # This should handle large positions without overflow
        reader = UsernameReader(large_filepath)
        assert reader.total_count == 2
        # Note: This test mainly verifies no integer overflow in position handling
        reader.close()

    @pytest.mark.parametrize("limit", [None, 3, 7, 15])
    def test_various_limits(self, limit):
        """Test various limit values."""
        with UsernameReader(self.test_filepath, limit=limit) as reader:
            expected_len = min(
                limit or len(self.test_usernames), len(self.test_usernames)
            )
            assert len(reader) == expected_len
            assert reader.effective_count == expected_len

            # Test that iteration respects limit
            usernames = list(reader)
            assert len(usernames) == expected_len

    def test_memory_mapping_efficiency(self):
        """Test that memory mapping is used efficiently."""
        # This test verifies that the reader uses memory mapping
        # We can't easily test memory usage directly, but we can verify
        # that large files can be opened without loading everything into memory

        with UsernameReader(self.test_filepath) as reader:
            # Multiple accesses should be efficient
            for i in range(min(100, len(reader))):
                _ = reader[i % len(reader)]

            # Should complete without excessive memory usage

    def test_concurrent_access(self):
        """Test concurrent access to the same file."""
        import threading

        results = []
        errors = []

        def read_usernames():
            try:
                with UsernameReader(self.test_filepath) as reader:
                    for i in range(len(reader)):
                        results.append(reader[i])
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=read_usernames)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should have no errors
        assert len(errors) == 0
        # Should have read data multiple times
        assert len(results) == 3 * len(self.test_usernames)

    def test_positions_proxy_behavior(self):
        """Test the internal PositionsProxy behavior."""
        with UsernameReader(self.test_filepath) as reader:
            positions = reader.positions

            # Test length
            assert len(positions) == len(self.test_usernames)

            # Test indexing
            assert positions[0] == 0  # First position should be 0

            # Test out of bounds
            with pytest.raises(IndexError):
                _ = positions[100]

            with pytest.raises(IndexError):
                _ = positions[-1]  # Negative indexing not supported

    def test_error_handling_corrupted_index(self):
        """Test error handling with corrupted index file."""
        # Create corrupted index file (wrong size)
        corrupted_filepath = Path(self.temp_dir) / "corrupted.dat"

        # Create valid data file
        with open(corrupted_filepath, "wb") as f:
            f.write(b"test")

        # Create index file with incorrect size (not multiple of 8)
        with open(corrupted_filepath.with_suffix(".idx"), "wb") as f:
            f.write(b"corrupted")  # 9 bytes, not multiple of 8

        # Should handle gracefully
        with UsernameReader(corrupted_filepath) as reader:
            # Should calculate count based on available complete entries
            assert reader.total_count == 1  # 9 // 8 = 1 complete entry


class TestUsernameReaderIntegration:
    """Integration tests with actual generated data."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_filepath = Path(self.temp_dir) / "integration_test.dat"

    def teardown_method(self):
        """Clean up integration test fixtures."""
        for file in Path(self.temp_dir).glob("*"):
            try:
                file.unlink()
            except Exception:
                pass
        try:
            os.rmdir(self.temp_dir)
        except Exception:
            pass

    def test_integration_with_generator(self):
        """Test integration with actual generator output."""
        # Generate test data using the actual generator
        generator = UsernameGenerator(seed=42)
        storage = EfficientUsernameStorage(self.test_filepath)

        test_count = 1000
        storage.store_usernames(generator, test_count, 100)

        # Test reading back the data
        with UsernameReader(self.test_filepath) as reader:
            assert len(reader) == test_count

            # Verify random access works
            first = reader[0]
            middle = reader[test_count // 2]
            last = reader[-1]

            assert isinstance(first, str)
            assert isinstance(middle, str)
            assert isinstance(last, str)

            # Verify iteration works
            count = 0
            for username in reader:
                assert isinstance(username, str)
                assert len(username) <= 20
                count += 1

            assert count == test_count

    def test_integration_reproducibility(self):
        """Test that reading is reproducible across multiple readers."""
        # Generate test data
        generator = UsernameGenerator(seed=42)
        storage = EfficientUsernameStorage(self.test_filepath)
        storage.store_usernames(generator, 100, 50)

        # Read with multiple readers
        usernames1 = []
        with UsernameReader(self.test_filepath) as reader1:
            usernames1 = list(reader1)

        usernames2 = []
        with UsernameReader(self.test_filepath) as reader2:
            usernames2 = list(reader2)

        # Should be identical
        assert usernames1 == usernames2
