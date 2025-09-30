"""
Comprehensive tests for LinearSearch algorithm.

Tests cover search functionality, performance tracking, statistics, and edge cases.
"""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data.generate import EfficientUsernameStorage
from data.reader import UsernameReader
from src.algorithms.algorithm import SearchResult
from src.algorithms.linear_search import LinearSearch


class TestLinearSearch:
    """Test suite for LinearSearch algorithm."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_filepath = Path(self.temp_dir) / "test_data.dat"

        # Create test data
        self.test_usernames = ["alice", "bob", "charlie", "diana", "eve"]
        self._create_test_data()

        self.reader = UsernameReader(self.test_filepath)

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

        storage = EfficientUsernameStorage(self.test_filepath)
        test_gen = TestGenerator(self.test_usernames)
        storage.store_usernames(test_gen, len(self.test_usernames), 3)

    def test_init_basic(self):
        """Test basic initialization."""
        linear_search = LinearSearch(self.reader)

        assert linear_search.reader == self.reader
        assert linear_search.track_performance is True  # default
        assert linear_search.total_comparisons == 0
        assert linear_search.total_searches == 0
        assert linear_search.total_time == 0.0

    def test_init_without_performance_tracking(self):
        """Test initialization without performance tracking."""
        linear_search = LinearSearch(self.reader, track_performance=False)

        assert linear_search.track_performance is False

    def test_search_found_first_element(self):
        """Test searching for first element."""
        linear_search = LinearSearch(self.reader)

        result = linear_search.search("alice")

        assert isinstance(result, SearchResult)
        assert result.found is True
        assert result.index == 0
        assert result.comparisons == 1
        assert result.time_taken > 0

    def test_search_found_middle_element(self):
        """Test searching for middle element."""
        linear_search = LinearSearch(self.reader)

        result = linear_search.search("charlie")

        assert result.found is True
        assert result.index == 2
        assert result.comparisons == 3  # Checked alice, bob, charlie
        assert result.time_taken > 0

    def test_search_found_last_element(self):
        """Test searching for last element."""
        linear_search = LinearSearch(self.reader)

        result = linear_search.search("eve")

        assert result.found is True
        assert result.index == 4
        assert result.comparisons == 5  # Checked all elements
        assert result.time_taken > 0

    def test_search_not_found(self):
        """Test searching for non-existent element."""
        linear_search = LinearSearch(self.reader)

        result = linear_search.search("frank")

        assert result.found is False
        assert result.index == -1
        assert result.comparisons == 5  # Checked all elements
        assert result.time_taken > 0

    def test_search_invalid_target(self):
        """Test searching with invalid target."""
        linear_search = LinearSearch(self.reader)

        result = linear_search.search("")

        assert result.found is False
        assert result.index == -1
        assert result.comparisons == 0
        assert result.time_taken == 0.0

    @pytest.mark.parametrize(
        "target,expected_found,expected_index,expected_comparisons",
        [
            ("alice", True, 0, 1),
            ("bob", True, 1, 2),
            ("charlie", True, 2, 3),
            ("diana", True, 3, 4),
            ("eve", True, 4, 5),
            ("zulu", False, -1, 5),
        ],
    )
    def test_search_various_targets(
        self, target, expected_found, expected_index, expected_comparisons
    ):
        """Test search with various targets."""
        linear_search = LinearSearch(self.reader)

        result = linear_search.search(target)

        assert result.found == expected_found
        assert result.index == expected_index
        assert result.comparisons == expected_comparisons

    def test_search_performance_tracking_enabled(self):
        """Test performance tracking when enabled."""
        linear_search = LinearSearch(self.reader, track_performance=True)

        # Perform multiple searches
        linear_search.search("alice")
        linear_search.search("charlie")
        linear_search.search("frank")

        # Check statistics were updated
        assert linear_search.total_searches == 3
        assert linear_search.total_comparisons == 1 + 3 + 5  # 9 total
        assert linear_search.total_time > 0

    def test_search_performance_tracking_disabled(self):
        """Test performance tracking when disabled."""
        linear_search = LinearSearch(self.reader, track_performance=False)

        # Perform searches
        linear_search.search("alice")
        linear_search.search("charlie")

        # Statistics should not be updated
        assert linear_search.total_searches == 0
        assert linear_search.total_comparisons == 0
        assert linear_search.total_time == 0.0

    def test_get_algorithm_name(self):
        """Test get_algorithm_name method."""
        linear_search = LinearSearch(self.reader)

        name = linear_search.get_algorithm_name()

        assert name == "LinearSearch"

    def test_get_performance_stats_no_searches(self):
        """Test performance statistics with no searches performed."""
        linear_search = LinearSearch(self.reader)

        stats = linear_search.get_performance_stats()

        expected = {
            "total_searches": 0,
            "total_comparisons": 0,
            "total_time": 0.0,
            "avg_comparisons": 0.0,
            "avg_time": 0.0,
            "comparisons_per_second": 0.0,
        }
        assert stats == expected

    def test_get_performance_stats_with_searches(self):
        """Test performance statistics after searches."""
        linear_search = LinearSearch(self.reader)

        # Perform some searches
        linear_search.search("alice")  # 1 comparison
        linear_search.search("charlie")  # 3 comparisons
        linear_search.search("frank")  # 5 comparisons (not found)

        stats = linear_search.get_performance_stats()

        assert stats["total_searches"] == 3
        assert stats["total_comparisons"] == 9
        assert stats["total_time"] > 0
        assert stats["avg_comparisons"] == 3.0  # 9/3
        assert stats["avg_time"] > 0
        assert stats["comparisons_per_second"] > 0

    def test_reset_statistics(self):
        """Test resetting performance statistics."""
        linear_search = LinearSearch(self.reader)

        # Perform searches to build up statistics
        linear_search.search("alice")
        linear_search.search("bob")

        # Verify statistics exist
        assert linear_search.total_searches > 0
        assert linear_search.total_comparisons > 0
        assert linear_search.total_time > 0

        # Reset and verify
        linear_search.reset_statistics()

        assert linear_search.total_searches == 0
        assert linear_search.total_comparisons == 0
        assert linear_search.total_time == 0.0

    def test_search_with_none_reader(self):
        """Test error handling with None reader."""
        linear_search = LinearSearch(None)

        with pytest.raises(ValueError, match="Reader cannot be None or empty"):
            linear_search.search("alice")

    def test_search_with_empty_reader(self):
        """Test searching with empty reader."""
        # Create empty data file
        empty_filepath = Path(self.temp_dir) / "empty.dat"
        with open(empty_filepath, "wb"):
            pass
        with open(empty_filepath.with_suffix(".idx"), "wb"):
            pass

        empty_reader = UsernameReader(empty_filepath)
        linear_search = LinearSearch(empty_reader)

        with pytest.raises(ValueError, match="Reader cannot be None or empty"):
            linear_search.search("alice")

        empty_reader.close()

    def test_search_timing_accuracy(self):
        """Test that timing measurements are reasonable."""
        linear_search = LinearSearch(self.reader)

        start_time = time.perf_counter()
        result = linear_search.search("charlie")
        end_time = time.perf_counter()
        actual_time = end_time - start_time

        # Measured time should be in the same order of magnitude
        assert result.time_taken <= actual_time * 1.1  # Allow 10% overhead
        assert result.time_taken > 0

    def test_search_unicode_targets(self):
        """Test searching with Unicode targets."""
        # Create data with Unicode usernames
        unicode_filepath = Path(self.temp_dir) / "unicode.dat"
        unicode_usernames = ["alice", "böb", "测试", "🙂user"]

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
            UnicodeGenerator(unicode_usernames), len(unicode_usernames), 2
        )

        unicode_reader = UsernameReader(unicode_filepath)
        linear_search = LinearSearch(unicode_reader)

        # Test Unicode search
        result = linear_search.search("测试")
        assert result.found is True
        assert result.index == 2

        unicode_reader.close()

    def test_search_case_sensitivity(self):
        """Test that search is case sensitive."""
        linear_search = LinearSearch(self.reader)

        # Exact match
        result1 = linear_search.search("alice")
        assert result1.found is True

        # Different case
        result2 = linear_search.search("Alice")
        assert result2.found is False

        result3 = linear_search.search("ALICE")
        assert result3.found is False

    def test_search_whitespace_handling(self):
        """Test search with whitespace in targets."""
        linear_search = LinearSearch(self.reader)

        # Target with leading/trailing spaces should not match
        result = linear_search.search(" alice ")
        assert result.found is False

    def test_multiple_search_instances(self):
        """Test multiple LinearSearch instances with same reader."""
        linear_search1 = LinearSearch(self.reader, track_performance=True)
        linear_search2 = LinearSearch(self.reader, track_performance=True)

        # Perform searches on different instances
        result1 = linear_search1.search("alice")
        result2 = linear_search2.search("bob")

        # Both should work independently
        assert result1.found is True
        assert result1.index == 0
        assert result2.found is True
        assert result2.index == 1

        # Statistics should be independent
        assert linear_search1.total_searches == 1
        assert linear_search2.total_searches == 1

    def test_search_with_duplicates(self):
        """Test searching in data with duplicate entries."""
        # Create data with duplicates
        dup_filepath = Path(self.temp_dir) / "duplicates.dat"
        dup_usernames = ["alice", "bob", "alice", "charlie", "alice"]

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
        linear_search = LinearSearch(dup_reader)

        # Should find first occurrence
        result = linear_search.search("alice")
        assert result.found is True
        assert result.index == 0  # First occurrence
        assert result.comparisons == 1

        dup_reader.close()

    def test_search_performance_with_large_dataset(self):
        """Test performance characteristics with larger dataset."""
        # Create larger test dataset
        large_filepath = Path(self.temp_dir) / "large.dat"
        large_usernames = [f"user_{i:04d}" for i in range(1000)]

        class LargeGenerator:
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

        storage = EfficientUsernameStorage(large_filepath)
        storage.store_usernames(
            LargeGenerator(large_usernames), len(large_usernames), 100
        )

        large_reader = UsernameReader(large_filepath)
        linear_search = LinearSearch(large_reader)

        # Test search at beginning, middle, end
        result_first = linear_search.search("user_0000")
        result_middle = linear_search.search("user_0500")
        result_last = linear_search.search("user_0999")

        assert result_first.found is True
        assert result_first.comparisons == 1

        assert result_middle.found is True
        assert result_middle.comparisons == 501

        assert result_last.found is True
        assert result_last.comparisons == 1000

        large_reader.close()

    def test_error_handling_reader_exceptions(self):
        """Test error handling when reader raises exceptions."""
        # Create mock reader that raises exceptions
        mock_reader = MagicMock()
        mock_reader.__len__ = MagicMock(return_value=5)
        mock_reader.__getitem__ = MagicMock(side_effect=IndexError("Mock error"))

        linear_search = LinearSearch(mock_reader)

        with pytest.raises(IndexError):
            linear_search.search("alice")

    def test_update_statistics_internal_method(self):
        """Test internal _update_statistics method."""
        linear_search = LinearSearch(self.reader, track_performance=True)

        # Call internal method directly
        linear_search._update_statistics(10, 0.005)

        assert linear_search.total_comparisons == 10
        assert linear_search.total_searches == 1
        assert linear_search.total_time == 0.005

    def test_inheritance_from_algorithm(self):
        """Test that LinearSearch properly inherits from Algorithm."""
        from src.algorithms.algorithm import Algorithm

        linear_search = LinearSearch(self.reader)

        assert isinstance(linear_search, Algorithm)
        assert hasattr(linear_search, "validate_target")
        assert hasattr(linear_search, "search")
        assert hasattr(linear_search, "get_algorithm_name")

    def test_search_result_structure_compliance(self):
        """Test that search results comply with SearchResult structure."""
        linear_search = LinearSearch(self.reader)

        result = linear_search.search("alice")

        # Check all required fields are present
        assert hasattr(result, "found")
        assert hasattr(result, "index")
        assert hasattr(result, "comparisons")
        assert hasattr(result, "time_taken")
        assert hasattr(result, "hash_operations")
        assert hasattr(result, "false_positive")
        assert hasattr(result, "additional_info")

        # Check types
        assert isinstance(result.found, bool)
        assert isinstance(result.index, int)
        assert isinstance(result.comparisons, int)
        assert isinstance(result.time_taken, float)

    def test_concurrent_searches(self):
        """Test concurrent search operations."""
        import threading

        linear_search = LinearSearch(self.reader, track_performance=True)
        results = []
        errors = []

        def perform_search(target):
            try:
                result = linear_search.search(target)
                results.append((target, result))
            except Exception as e:
                errors.append(e)

        # Create threads for concurrent searches
        targets = ["alice", "bob", "charlie", "diana", "eve"]
        threads = []

        for target in targets:
            thread = threading.Thread(target=perform_search, args=(target,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should complete without errors
        assert len(errors) == 0
        assert len(results) == 5

        # All searches should have found their targets
        for target, result in results:
            assert result.found is True

    @patch("time.perf_counter")
    def test_timing_measurement_mocking(self, mock_time):
        """Test timing measurement with mocked time."""
        # Mock time progression
        mock_time.side_effect = [0.0, 0.001]  # start, end

        linear_search = LinearSearch(self.reader)
        result = linear_search.search("alice")

        assert result.time_taken == 0.001

    def test_string_representation(self):
        """Test string representation of LinearSearch."""
        linear_search = LinearSearch(self.reader)

        str_repr = str(linear_search)
        assert str_repr == "LinearSearch"

    def test_performance_stats_division_by_zero_protection(self):
        """Test that performance stats handle division by zero."""
        linear_search = LinearSearch(self.reader)

        # Get stats with no searches (should not cause division by zero)
        stats = linear_search.get_performance_stats()

        assert stats["avg_comparisons"] == 0.0
        assert stats["avg_time"] == 0.0
        assert stats["comparisons_per_second"] == 0.0
