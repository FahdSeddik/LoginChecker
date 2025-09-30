"""
Comprehensive tests for BinarySearch algorithm.

Tests cover search functionality on sorted data, edge cases, and performance characteristics.
"""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from data.generate import EfficientUsernameStorage
from data.reader import UsernameReader
from src.algorithms.algorithm import SearchResult
from src.algorithms.binary_search import BinarySearch
from src.data_structures.sorted_list import SortedList


class TestBinarySearch:
    """Test suite for BinarySearch algorithm."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_filepath = Path(self.temp_dir) / "test_data.dat"
        self.sorted_filepath = Path(self.temp_dir) / "sorted_data.dat"

        # Create test data (unsorted)
        self.test_usernames = ["zebra", "alice", "charlie", "bob", "diana"]
        self.sorted_usernames = sorted(
            self.test_usernames
        )  # ["alice", "bob", "charlie", "diana", "zebra"]
        self._create_test_data()

        # Create sorted list
        self.reader = UsernameReader(self.test_filepath)
        self.sorted_list = SortedList(self.reader, self.sorted_filepath)
        self.sorted_list.sort(chunk_size=3)

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "reader"):
            self.reader.close()
        if hasattr(self, "sorted_list"):
            self.sorted_list.close()

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
        binary_search = BinarySearch(self.sorted_list)

        assert binary_search.sorted_list == self.sorted_list
        assert binary_search.reader == self.sorted_list  # Inherited from Algorithm

    def test_search_found_first_element(self):
        """Test searching for first element in sorted data."""
        binary_search = BinarySearch(self.sorted_list)

        result = binary_search.search("alice")

        assert isinstance(result, SearchResult)
        assert result.found is True
        assert result.index == 0
        assert result.comparisons >= 1
        assert result.time_taken > 0

    def test_search_found_last_element(self):
        """Test searching for last element in sorted data."""
        binary_search = BinarySearch(self.sorted_list)

        result = binary_search.search("zebra")

        assert result.found is True
        assert result.index == 4
        assert result.comparisons >= 1
        assert result.time_taken > 0

    def test_search_found_middle_element(self):
        """Test searching for middle element."""
        binary_search = BinarySearch(self.sorted_list)

        result = binary_search.search("charlie")

        assert result.found is True
        assert result.index == 2
        assert result.comparisons >= 1
        assert result.time_taken > 0

    def test_search_not_found(self):
        """Test searching for non-existent element."""
        binary_search = BinarySearch(self.sorted_list)

        result = binary_search.search("frank")

        assert result.found is False
        assert result.index == -1
        assert result.comparisons >= 1
        assert result.time_taken > 0

    def test_search_invalid_target(self):
        """Test searching with invalid target."""
        binary_search = BinarySearch(self.sorted_list)

        result = binary_search.search("")

        assert result.found is False
        assert result.index == -1
        assert result.comparisons == 0
        assert result.time_taken == 0.0

    @pytest.mark.parametrize(
        "target,expected_found,expected_index",
        [
            ("alice", True, 0),
            ("bob", True, 1),
            ("charlie", True, 2),
            ("diana", True, 3),
            ("zebra", True, 4),
            ("aaron", False, -1),  # Before first
            ("zulu", False, -1),  # After last
            ("caroline", False, -1),  # Between existing
        ],
    )
    def test_search_various_targets(self, target, expected_found, expected_index):
        """Test search with various targets."""
        binary_search = BinarySearch(self.sorted_list)

        result = binary_search.search(target)

        assert result.found == expected_found
        assert result.index == expected_index

    def test_search_efficiency_comparison_count(self):
        """Test that binary search uses logarithmic comparisons."""
        binary_search = BinarySearch(self.sorted_list)

        # For 5 elements, binary search should use at most ceil(log2(5)) = 3 comparisons
        result = binary_search.search("zebra")  # Worst case

        assert result.comparisons <= 3
        assert result.comparisons >= 1

    def test_get_algorithm_name(self):
        """Test get_algorithm_name method."""
        binary_search = BinarySearch(self.sorted_list)

        name = binary_search.get_algorithm_name()

        assert name == "Binary Search"

    def test_search_timing_accuracy(self):
        """Test that timing measurements are reasonable."""
        binary_search = BinarySearch(self.sorted_list)

        start_time = time.perf_counter()
        result = binary_search.search("charlie")
        end_time = time.perf_counter()
        actual_time = end_time - start_time

        # Measured time should be in the same order of magnitude
        assert result.time_taken <= actual_time * 1.1  # Allow 10% overhead
        assert result.time_taken > 0

    def test_search_case_sensitivity(self):
        """Test that search is case sensitive."""
        binary_search = BinarySearch(self.sorted_list)

        # Exact match
        result1 = binary_search.search("alice")
        assert result1.found is True

        # Different case
        result2 = binary_search.search("Alice")
        assert result2.found is False

        result3 = binary_search.search("ALICE")
        assert result3.found is False

    def test_search_whitespace_handling(self):
        """Test search with whitespace in targets."""
        binary_search = BinarySearch(self.sorted_list)

        # Target with leading/trailing spaces should not match
        result = binary_search.search(" alice ")
        assert result.found is False

    def test_binary_search_algorithm_correctness(self):
        """Test the correctness of binary search algorithm implementation."""
        binary_search = BinarySearch(self.sorted_list)

        # Test all elements to ensure algorithm works correctly
        for i, username in enumerate(self.sorted_usernames):
            result = binary_search.search(username)
            assert result.found is True
            assert result.index == i

    def test_search_with_single_element(self):
        """Test binary search with single element."""
        # Create single-element sorted list
        single_filepath = Path(self.temp_dir) / "single.dat"
        single_usernames = ["single"]

        class SingleGenerator:
            def generate_batch(self, count):
                return single_usernames[:count]

        storage = EfficientUsernameStorage(single_filepath)
        storage.store_usernames(SingleGenerator(), 1, 1)

        single_reader = UsernameReader(single_filepath)
        single_sorted_filepath = Path(self.temp_dir) / "single_sorted.dat"
        single_sorted_list = SortedList(single_reader, single_sorted_filepath)
        single_sorted_list.sort(chunk_size=2)

        binary_search = BinarySearch(single_sorted_list)

        # Search for the element
        result1 = binary_search.search("single")
        assert result1.found is True
        assert result1.index == 0
        assert result1.comparisons == 1

        # Search for non-existent element
        result2 = binary_search.search("other")
        assert result2.found is False
        assert result2.index == -1

        single_reader.close()
        single_sorted_list.close()

    def test_search_with_empty_list(self):
        """Test binary search with empty list."""
        # Create empty sorted list
        empty_filepath = Path(self.temp_dir) / "empty.dat"
        with open(empty_filepath, "wb"):
            pass
        with open(empty_filepath.with_suffix(".idx"), "wb"):
            pass

        empty_reader = UsernameReader(empty_filepath)
        empty_sorted_filepath = Path(self.temp_dir) / "empty_sorted.dat"
        empty_sorted_list = SortedList(empty_reader, empty_sorted_filepath)
        empty_sorted_list.sort(chunk_size=2)

        binary_search = BinarySearch(empty_sorted_list)

        result = binary_search.search("anything")
        assert result.found is False
        assert result.index == -1
        assert result.comparisons == 0

        empty_reader.close()
        empty_sorted_list.close()

    def test_search_with_duplicates(self):
        """Test binary search with duplicate elements."""
        # Create sorted list with duplicates
        dup_filepath = Path(self.temp_dir) / "duplicates.dat"
        dup_usernames = ["alice", "alice", "bob", "bob", "charlie"]

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
        dup_sorted_filepath = Path(self.temp_dir) / "dup_sorted.dat"
        dup_sorted_list = SortedList(dup_reader, dup_sorted_filepath)
        dup_sorted_list.sort(chunk_size=2)

        binary_search = BinarySearch(dup_sorted_list)

        # Should find one of the duplicate instances
        result = binary_search.search("alice")
        assert result.found is True
        assert result.index in [0, 1]  # Could be either duplicate

        dup_reader.close()
        dup_sorted_list.close()

    def test_search_unicode_targets(self):
        """Test searching with Unicode targets."""
        # Create sorted list with Unicode usernames
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
        unicode_sorted_filepath = Path(self.temp_dir) / "unicode_sorted.dat"
        unicode_sorted_list = SortedList(unicode_reader, unicode_sorted_filepath)
        unicode_sorted_list.sort(chunk_size=2)

        binary_search = BinarySearch(unicode_sorted_list)

        # Test Unicode search
        result = binary_search.search("测试")
        assert result.found is True

        unicode_reader.close()
        unicode_sorted_list.close()

    def test_large_dataset_performance(self):
        """Test binary search performance with larger dataset."""
        # Create larger sorted dataset
        large_filepath = Path(self.temp_dir) / "large.dat"
        large_usernames = [f"user_{i:04d}" for i in range(100)]  # 100 sorted usernames

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
            LargeGenerator(large_usernames), len(large_usernames), 50
        )

        large_reader = UsernameReader(large_filepath)
        large_sorted_filepath = Path(self.temp_dir) / "large_sorted.dat"
        large_sorted_list = SortedList(large_reader, large_sorted_filepath)
        large_sorted_list.sort(chunk_size=25)

        binary_search = BinarySearch(large_sorted_list)

        # Test searches at different positions
        result_first = binary_search.search("user_0000")
        result_middle = binary_search.search("user_0050")
        result_last = binary_search.search("user_0099")

        # All should be found
        assert result_first.found is True
        assert result_middle.found is True
        assert result_last.found is True

        # Comparisons should be logarithmic (at most ceil(log2(100)) = 7)
        assert result_first.comparisons <= 7
        assert result_middle.comparisons <= 7
        assert result_last.comparisons <= 7

        large_reader.close()
        large_sorted_list.close()

    def test_boundary_conditions(self):
        """Test binary search boundary conditions."""
        binary_search = BinarySearch(self.sorted_list)

        # Test searching for values that would be at boundaries

        # Search for value that would come before first element
        result_before = binary_search.search("aaron")
        assert result_before.found is False

        # Search for value that would come after last element
        result_after = binary_search.search("zulu")
        assert result_after.found is False

        # Search for value that would be between existing elements
        result_between = binary_search.search("caroline")
        assert result_between.found is False

    def test_inheritance_from_algorithm(self):
        """Test that BinarySearch properly inherits from Algorithm."""
        from src.algorithms.algorithm import Algorithm

        binary_search = BinarySearch(self.sorted_list)

        assert isinstance(binary_search, Algorithm)
        assert hasattr(binary_search, "validate_target")
        assert hasattr(binary_search, "search")
        assert hasattr(binary_search, "get_algorithm_name")

    def test_search_result_structure_compliance(self):
        """Test that search results comply with SearchResult structure."""
        binary_search = BinarySearch(self.sorted_list)

        result = binary_search.search("alice")

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

    def test_comparison_count_consistency(self):
        """Test that comparison count is consistent and reasonable."""
        binary_search = BinarySearch(self.sorted_list)

        # Multiple searches for same target should have same comparison count
        result1 = binary_search.search("charlie")
        result2 = binary_search.search("charlie")

        assert result1.comparisons == result2.comparisons

    def test_concurrent_searches(self):
        """Test concurrent search operations."""
        import threading

        binary_search = BinarySearch(self.sorted_list)
        results = []
        errors = []

        def perform_search(target):
            try:
                result = binary_search.search(target)
                results.append((target, result))
            except Exception as e:
                errors.append(e)

        # Create threads for concurrent searches
        targets = ["alice", "bob", "charlie", "diana", "zebra"]
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
        mock_time.side_effect = [0.0, 0.002]  # start, end

        binary_search = BinarySearch(self.sorted_list)
        result = binary_search.search("alice")

        assert result.time_taken == 0.002

    def test_string_representation(self):
        """Test string representation of BinarySearch."""
        binary_search = BinarySearch(self.sorted_list)

        str_repr = str(binary_search)
        assert str_repr == "Binary Search"

    def test_binary_search_specific_edge_cases(self):
        """Test edge cases specific to binary search implementation."""
        binary_search = BinarySearch(self.sorted_list)

        # Test with two-element array
        two_element_filepath = Path(self.temp_dir) / "two_element.dat"
        two_usernames = ["alice", "bob"]

        class TwoGenerator:
            def generate_batch(self, count):
                return two_usernames[:count]

        storage = EfficientUsernameStorage(two_element_filepath)
        storage.store_usernames(TwoGenerator(), 2, 2)

        two_reader = UsernameReader(two_element_filepath)
        two_sorted_filepath = Path(self.temp_dir) / "two_sorted.dat"
        two_sorted_list = SortedList(two_reader, two_sorted_filepath)
        two_sorted_list.sort(chunk_size=2)

        two_binary_search = BinarySearch(two_sorted_list)

        # Test both elements
        result1 = two_binary_search.search("alice")
        result2 = two_binary_search.search("bob")

        assert result1.found is True
        assert result1.index == 0
        assert result2.found is True
        assert result2.index == 1

        two_reader.close()
        two_sorted_list.close()

    def test_algorithm_efficiency_vs_linear(self):
        """Test that binary search is more efficient than linear search would be."""
        binary_search_algo = BinarySearch(self.sorted_list)

        # For worst case in 5 elements:
        # Linear search would need 5 comparisons
        # Binary search should need at most 3 comparisons
        result = binary_search_algo.search("zebra")  # Last element (worst case)

        assert result.comparisons < 5  # Better than linear search
