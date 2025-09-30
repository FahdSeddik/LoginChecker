"""
Comprehensive tests for SortedList class.

Tests cover external merge sort, chunk handling, sorted access, and memory efficiency.
"""

import os
import tempfile
import unittest.mock
from pathlib import Path
from unittest.mock import patch

import pytest

from data.generate import EfficientUsernameStorage, UsernameGenerator
from data.reader import UsernameReader
from src.data_structures.sorted_list import SortedList


class TestSortedList:
    """Test suite for SortedList class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_filepath = Path(self.temp_dir) / "test_data.dat"

        # Create test data
        self.test_usernames = ["zebra", "alice", "charlie", "bob", "diana"]
        self._create_test_data()

        self.reader = UsernameReader(self.test_filepath)
        self.sorted_filepath = Path(self.temp_dir) / "sorted_data.dat"

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "reader"):
            self.reader.close()

        for file in Path(self.temp_dir).glob("*"):
            try:
                file.unlink()
            except Exception:
                pass
        try:
            os.rmdir(self.temp_dir)
        except Exception:
            pass

    def _create_test_data(self):
        """Create test data files."""
        storage = EfficientUsernameStorage(self.test_filepath)

        # Create custom generator for our test data
        class TestGenerator:
            def __init__(self, usernames):
                self.usernames = usernames
                self.index = 0

            def generate_batch(self, count):
                batch = []
                for _ in range(count):
                    if self.index < len(self.usernames):
                        batch.append(self.usernames[self.index])
                        self.index += 1
                    else:
                        break
                return batch

        test_gen = TestGenerator(self.test_usernames)
        storage.store_usernames(test_gen, len(self.test_usernames), 3)

    def test_init_basic(self):
        """Test basic initialization."""
        sorted_list = SortedList(self.reader)

        assert sorted_list.reader == self.reader
        assert sorted_list.sorted_filepath is None
        assert not sorted_list._is_sorted
        assert sorted_list._sorted_reader is None

    def test_init_with_filepath(self):
        """Test initialization with specified sorted filepath."""
        sorted_list = SortedList(self.reader, self.sorted_filepath)

        assert sorted_list.sorted_filepath == self.sorted_filepath

    def test_sort_basic(self):
        """Test basic sorting functionality."""
        sorted_list = SortedList(self.reader, self.sorted_filepath)

        sorted_list.sort(chunk_size=3)

        # Verify sort completed
        assert sorted_list._is_sorted
        assert sorted_list._sorted_reader is not None
        assert sorted_list.sorted_filepath.exists()
        assert sorted_list.sorted_filepath.with_suffix(".idx").exists()

        # Verify data is sorted
        expected_sorted = sorted(self.test_usernames)
        actual_sorted = [sorted_list[i] for i in range(len(sorted_list))]
        assert actual_sorted == expected_sorted

    def test_sort_without_filepath(self):
        """Test sorting without pre-specified filepath."""
        sorted_list = SortedList(self.reader)

        sorted_list.sort(chunk_size=3)

        # Should create filepath automatically
        assert sorted_list.sorted_filepath is not None
        assert sorted_list.sorted_filepath.exists()
        assert str(sorted_list.sorted_filepath).endswith("_sorted.dat")

    def test_sort_single_chunk(self):
        """Test sorting when all data fits in single chunk."""
        sorted_list = SortedList(self.reader, self.sorted_filepath)

        # Use chunk size larger than data
        sorted_list.sort(chunk_size=100)

        assert sorted_list._is_sorted
        expected_sorted = sorted(self.test_usernames)
        actual_sorted = [sorted_list[i] for i in range(len(sorted_list))]
        assert actual_sorted == expected_sorted

    def test_sort_multiple_chunks(self):
        """Test sorting with multiple chunks."""
        sorted_list = SortedList(self.reader, self.sorted_filepath)

        # Use small chunk size to force multiple chunks
        sorted_list.sort(chunk_size=2)

        assert sorted_list._is_sorted
        expected_sorted = sorted(self.test_usernames)
        actual_sorted = [sorted_list[i] for i in range(len(sorted_list))]
        assert actual_sorted == expected_sorted

    def test_sort_progress_reporting(self, capsys):
        """Test progress reporting during sort."""
        sorted_list = SortedList(self.reader, self.sorted_filepath)

        sorted_list.sort(chunk_size=2)

        captured = capsys.readouterr()
        assert "Starting sort" in captured.out
        assert "Sort complete" in captured.out

    def test_getitem_before_sort(self):
        """Test accessing items before sorting raises error."""
        sorted_list = SortedList(self.reader, self.sorted_filepath)

        with pytest.raises(RuntimeError, match="Must call sort"):
            _ = sorted_list[0]

    def test_len_before_sort(self):
        """Test getting length before sorting raises error."""
        sorted_list = SortedList(self.reader, self.sorted_filepath)

        with pytest.raises(RuntimeError, match="Must call sort"):
            _ = len(sorted_list)

    def test_getitem_after_sort(self):
        """Test item access after sorting."""
        sorted_list = SortedList(self.reader, self.sorted_filepath)
        sorted_list.sort(chunk_size=3)

        # Test various access patterns
        assert sorted_list[0] == "alice"  # First in sorted order
        assert sorted_list[-1] == "zebra"  # Last in sorted order

        # Test all positions
        expected_sorted = sorted(self.test_usernames)
        for i in range(len(sorted_list)):
            assert sorted_list[i] == expected_sorted[i]

    def test_len_after_sort(self):
        """Test length after sorting."""
        sorted_list = SortedList(self.reader, self.sorted_filepath)
        sorted_list.sort(chunk_size=3)

        assert len(sorted_list) == len(self.test_usernames)

    def test_context_manager(self):
        """Test context manager functionality."""
        with SortedList(self.reader, self.sorted_filepath) as sorted_list:
            sorted_list.sort(chunk_size=3)
            assert sorted_list[0] == "alice"

        # Should be properly closed after context exit

    def test_manual_close(self):
        """Test manual close functionality."""
        sorted_list = SortedList(self.reader, self.sorted_filepath)
        sorted_list.sort(chunk_size=3)

        sorted_list.close()
        # After closing, the sorted reader should be closed

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 10])
    def test_sort_various_chunk_sizes(self, chunk_size):
        """Test sorting with various chunk sizes."""
        sorted_list = SortedList(self.reader, self.sorted_filepath)

        sorted_list.sort(chunk_size=chunk_size)

        # Should produce same sorted result regardless of chunk size
        expected_sorted = sorted(self.test_usernames)
        actual_sorted = [sorted_list[i] for i in range(len(sorted_list))]
        assert actual_sorted == expected_sorted

    def test_sort_with_temp_dir(self):
        """Test sorting with specified temporary directory."""
        temp_sort_dir = Path(self.temp_dir) / "custom_temp"
        temp_sort_dir.mkdir()

        sorted_list = SortedList(self.reader, self.sorted_filepath)
        sorted_list.sort(chunk_size=2, temp_dir=str(temp_sort_dir))

        # Should complete successfully
        assert sorted_list._is_sorted
        expected_sorted = sorted(self.test_usernames)
        actual_sorted = [sorted_list[i] for i in range(len(sorted_list))]
        assert actual_sorted == expected_sorted

    def test_create_sorted_chunks(self):
        """Test internal chunk creation functionality."""
        sorted_list = SortedList(self.reader, self.sorted_filepath)

        # Test chunk creation with mocked temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            chunk_files = sorted_list._create_sorted_chunks(2, temp_path)

            # Should create multiple chunk files
            assert len(chunk_files) > 1  # With chunk_size=2 and 5 items

            # Each chunk file should exist
            for chunk_file in chunk_files:
                assert chunk_file.exists()
                assert chunk_file.with_suffix(".idx").exists()

    def test_merge_chunks_functionality(self):
        """Test internal chunk merging functionality."""
        sorted_list = SortedList(self.reader, self.sorted_filepath)

        # Create some test chunks manually
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test chunks
            chunk_files = sorted_list._create_sorted_chunks(2, temp_path)

            # Merge chunks
            sorted_list._merge_chunks(chunk_files)

            # Verify merge result
            assert sorted_list.sorted_filepath.exists()
            assert sorted_list.sorted_filepath.with_suffix(".idx").exists()

    def test_sort_empty_data(self):
        """Test sorting with empty data."""
        # Create empty data file
        empty_filepath = Path(self.temp_dir) / "empty.dat"
        with open(empty_filepath, "wb"):
            pass
        with open(empty_filepath.with_suffix(".idx"), "wb"):
            pass

        empty_reader = UsernameReader(empty_filepath)
        sorted_list = SortedList(empty_reader, self.sorted_filepath)

        sorted_list.sort(chunk_size=3)

        assert sorted_list._is_sorted
        assert len(sorted_list) == 0

        empty_reader.close()

    def test_sort_single_item(self):
        """Test sorting with single item."""
        # Create single-item data
        single_filepath = Path(self.temp_dir) / "single.dat"
        single_usernames = ["single"]

        storage = EfficientUsernameStorage(single_filepath)

        class SingleGenerator:
            def generate_batch(self, count):
                return single_usernames[:count]

        storage.store_usernames(SingleGenerator(), 1, 1)

        single_reader = UsernameReader(single_filepath)
        sorted_list = SortedList(single_reader, self.sorted_filepath)

        sorted_list.sort(chunk_size=3)

        assert sorted_list._is_sorted
        assert len(sorted_list) == 1
        assert sorted_list[0] == "single"

        single_reader.close()

    def test_sort_duplicate_items(self):
        """Test sorting with duplicate usernames."""
        # Create data with duplicates
        dup_filepath = Path(self.temp_dir) / "duplicates.dat"
        dup_usernames = ["alice", "bob", "alice", "charlie", "bob"]

        class DuplicateGenerator:
            def __init__(self, usernames):
                self.usernames = usernames
                self.index = 0

            def generate_batch(self, count):
                batch = []
                for _ in range(count):
                    if self.index < len(self.usernames):
                        batch.append(self.usernames[self.index])
                        self.index += 1
                return batch

        storage = EfficientUsernameStorage(dup_filepath)
        storage.store_usernames(
            DuplicateGenerator(dup_usernames), len(dup_usernames), 3
        )

        dup_reader = UsernameReader(dup_filepath)
        sorted_list = SortedList(dup_reader, self.sorted_filepath)

        sorted_list.sort(chunk_size=2)

        expected_sorted = sorted(dup_usernames)
        actual_sorted = [sorted_list[i] for i in range(len(sorted_list))]
        assert actual_sorted == expected_sorted

        dup_reader.close()

    def test_sort_unicode_usernames(self):
        """Test sorting with Unicode usernames."""
        unicode_filepath = Path(self.temp_dir) / "unicode.dat"
        unicode_usernames = ["zebra", "älice", "bob", "ñoël", "charlie"]

        class UnicodeGenerator:
            def __init__(self, usernames):
                self.usernames = usernames
                self.index = 0

            def generate_batch(self, count):
                batch = []
                for _ in range(count):
                    if self.index < len(self.usernames):
                        batch.append(self.usernames[self.index])
                        self.index += 1
                return batch

        storage = EfficientUsernameStorage(unicode_filepath)
        storage.store_usernames(
            UnicodeGenerator(unicode_usernames), len(unicode_usernames), 3
        )

        unicode_reader = UsernameReader(unicode_filepath)
        sorted_list = SortedList(unicode_reader, self.sorted_filepath)

        sorted_list.sort(chunk_size=2)

        # Unicode sorting should work correctly
        expected_sorted = sorted(unicode_usernames)
        actual_sorted = [sorted_list[i] for i in range(len(sorted_list))]
        assert actual_sorted == expected_sorted

        unicode_reader.close()

    def test_sort_memory_efficiency(self):
        """Test that sorting doesn't load all data into memory at once."""
        # This is more of a design verification test
        sorted_list = SortedList(self.reader, self.sorted_filepath)

        # Mock the chunk creation to verify it's called with reasonable chunk sizes
        with patch.object(sorted_list, "_create_sorted_chunks") as mock_create:
            mock_create.return_value = []
            with patch.object(sorted_list, "_merge_chunks"):
                sorted_list.sort(chunk_size=2)

            # Should have been called with the specified chunk size
            mock_create.assert_called_once_with(2, unittest.mock.ANY)

    def test_error_handling_during_sort(self):
        """Test error handling during sort operations."""
        sorted_list = SortedList(self.reader, self.sorted_filepath)

        # Test error during chunk creation
        with patch.object(sorted_list, "_create_sorted_chunks") as mock_create:
            mock_create.side_effect = Exception("Chunk creation error")

            with pytest.raises(Exception, match="Chunk creation error"):
                sorted_list.sort(chunk_size=2)

    def test_sort_file_permissions(self):
        """Test that sorted files are created with proper permissions."""
        sorted_list = SortedList(self.reader, self.sorted_filepath)

        sorted_list.sort(chunk_size=3)

        # Verify files are readable
        assert os.access(sorted_list.sorted_filepath, os.R_OK)
        assert os.access(sorted_list.sorted_filepath.with_suffix(".idx"), os.R_OK)

    def test_sort_cleanup_temp_files(self):
        """Test that temporary files are cleaned up after sorting."""
        sorted_list = SortedList(self.reader, self.sorted_filepath)

        # Track temporary directory
        temp_dirs_created = []
        original_temp_dir = tempfile.TemporaryDirectory

        class TrackedTempDir(original_temp_dir):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                temp_dirs_created.append(self.name)

        with patch("tempfile.TemporaryDirectory", TrackedTempDir):
            sorted_list.sort(chunk_size=2)

        # Temporary directories should be cleaned up automatically
        for temp_dir in temp_dirs_created:
            assert not Path(temp_dir).exists()

    def test_concurrent_sort_safety(self):
        """Test concurrent sorting safety considerations."""
        # Create multiple readers and sorted lists
        reader1 = UsernameReader(self.test_filepath)
        reader2 = UsernameReader(self.test_filepath)

        sorted_list1 = SortedList(reader1, Path(self.temp_dir) / "sorted1.dat")
        sorted_list2 = SortedList(reader2, Path(self.temp_dir) / "sorted2.dat")

        import threading

        errors = []

        def sort_data(sorted_list):
            try:
                sorted_list.sort(chunk_size=2)
            except Exception as e:
                errors.append(e)

        # Run concurrent sorts
        thread1 = threading.Thread(target=sort_data, args=(sorted_list1,))
        thread2 = threading.Thread(target=sort_data, args=(sorted_list2,))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # Should complete without errors
        assert len(errors) == 0

        # Both should be sorted correctly
        expected_sorted = sorted(self.test_usernames)
        actual1 = [sorted_list1[i] for i in range(len(sorted_list1))]
        actual2 = [sorted_list2[i] for i in range(len(sorted_list2))]

        assert actual1 == expected_sorted
        assert actual2 == expected_sorted

        # Cleanup
        reader1.close()
        reader2.close()
        sorted_list1.close()
        sorted_list2.close()


class TestSortedListIntegration:
    """Integration tests for SortedList with larger datasets."""

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

    def test_integration_with_generated_data(self):
        """Test sorting with actual generated username data."""
        # Generate larger dataset for more realistic testing
        generator = UsernameGenerator(seed=42)
        storage = EfficientUsernameStorage(self.test_filepath)

        test_count = 1000
        storage.store_usernames(generator, test_count, 100)

        # Test sorting
        with UsernameReader(self.test_filepath) as reader:
            sorted_filepath = Path(self.temp_dir) / "sorted_integration.dat"
            with SortedList(reader, sorted_filepath) as sorted_list:
                sorted_list.sort(chunk_size=100)

                # Verify sorting worked
                assert len(sorted_list) == test_count

                # Check that data is actually sorted
                prev_username = sorted_list[0]
                for i in range(1, min(100, len(sorted_list))):  # Check first 100
                    current_username = sorted_list[i]
                    assert prev_username <= current_username
                    prev_username = current_username

    def test_performance_characteristics(self):
        """Test performance characteristics of sorting."""
        import time

        # Generate test data
        generator = UsernameGenerator(seed=42)
        storage = EfficientUsernameStorage(self.test_filepath)
        storage.store_usernames(generator, 500, 100)

        with UsernameReader(self.test_filepath) as reader:
            sorted_filepath = Path(self.temp_dir) / "perf_test.dat"
            sorted_list = SortedList(reader, sorted_filepath)

            # Time the sorting operation
            start_time = time.time()
            sorted_list.sort(chunk_size=50)
            sort_time = time.time() - start_time

            # Should complete in reasonable time (< 10 seconds for 500 items)
            assert sort_time < 10.0

            # Verify correctness
            assert len(sorted_list) == 500

            sorted_list.close()
