"""
Integration tests for the LoginChecker system.

Tests cover end-to-end workflows combining multiple components.
"""

import os
import tempfile
from pathlib import Path

import pytest

from data.generate import EfficientUsernameStorage, UsernameGenerator
from data.reader import UsernameReader
from data.writer import UsernameWriter
from src.algorithms.binary_search import BinarySearch
from src.algorithms.linear_search import LinearSearch
from src.data_structures.disk_hashset import DiskHashSet
from src.data_structures.sorted_list import SortedList


class TestIntegrationWorkflows:
    """Integration tests for complete workflows."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

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

    def test_complete_generation_to_search_workflow(self):
        """Test complete workflow from generation to search."""
        # Step 1: Generate usernames
        generator = UsernameGenerator(seed=42)
        data_filepath = Path(self.temp_dir) / "workflow_data.dat"
        storage = EfficientUsernameStorage(data_filepath)

        username_count = 1000
        storage.store_usernames(generator, username_count, 100)

        # Step 2: Read data back
        with UsernameReader(data_filepath) as reader:
            assert len(reader) == username_count

            # Step 3: Test linear search
            linear_search = LinearSearch(reader)

            # Search for first username
            first_username = reader[0]
            result = linear_search.search(first_username)
            assert result.found is True
            assert result.index == 0

            # Search for last username
            last_username = reader[-1]
            result = linear_search.search(last_username)
            assert result.found is True
            assert result.index == username_count - 1

            # Search for non-existent username
            result = linear_search.search("nonexistent_user")
            assert result.found is False

    def test_sorting_and_binary_search_workflow(self):
        """Test workflow with sorting and binary search."""
        # Step 1: Generate test data
        generator = UsernameGenerator(seed=123)
        data_filepath = Path(self.temp_dir) / "sort_workflow_data.dat"
        storage = EfficientUsernameStorage(data_filepath)

        username_count = 500
        storage.store_usernames(generator, username_count, 50)

        # Step 2: Create sorted version
        with UsernameReader(data_filepath) as reader:
            sorted_filepath = Path(self.temp_dir) / "sorted_workflow_data.dat"
            with SortedList(reader, sorted_filepath) as sorted_list:
                sorted_list.sort(chunk_size=100)

                assert len(sorted_list) == username_count

                # Step 3: Verify sorting
                prev_username = sorted_list[0]
                for i in range(1, min(50, len(sorted_list))):
                    current_username = sorted_list[i]
                    assert prev_username <= current_username
                    prev_username = current_username

                # Step 4: Test binary search
                binary_search = BinarySearch(sorted_list)

                # Test searching for various elements
                test_indices = [
                    0,
                    username_count // 4,
                    username_count // 2,
                    3 * username_count // 4,
                    username_count - 1,
                ]

                for idx in test_indices:
                    target_username = sorted_list[idx]
                    result = binary_search.search(target_username)
                    assert result.found is True
                    assert result.index == idx
                    # Binary search should be efficient
                    assert result.comparisons <= 10  # log2(500) ≈ 9

    def test_hashset_workflow(self):
        """Test workflow with disk hashset."""
        # Step 1: Generate usernames
        generator = UsernameGenerator(seed=456)
        usernames = list(generator.generate_batch(200))

        # Step 2: Create and populate hashset
        hashset_path = str(Path(self.temp_dir) / "workflow_hashset.dat")
        hashset = DiskHashSet(hashset_path, num_slots_power=8, create=True)  # 256 slots

        # Add all usernames
        for username in usernames:
            result = hashset.add(username)
            assert result is True  # Should be new

        # Step 3: Test contains functionality
        for username in usernames:
            assert hashset.contains(username) is True

        # Test non-existent usernames
        non_existent = ["fake1", "fake2", "fake3"]
        for fake_username in non_existent:
            assert hashset.contains(fake_username) is False

        # Step 4: Test persistence
        hashset.sync()
        hashset.close()

        # Reopen and verify data persists
        hashset2 = DiskHashSet(hashset_path, create=False)
        for username in usernames[:10]:  # Test subset
            assert hashset2.contains(username) is True

        hashset2.close()

    def test_performance_comparison_workflow(self):
        """Test workflow comparing different search algorithms."""
        # Step 1: Generate test data
        generator = UsernameGenerator(seed=789)
        data_filepath = Path(self.temp_dir) / "perf_data.dat"
        storage = EfficientUsernameStorage(data_filepath)

        username_count = 1000
        storage.store_usernames(generator, username_count, 100)

        with UsernameReader(data_filepath) as reader:
            # Step 2: Create sorted version for binary search
            sorted_filepath = Path(self.temp_dir) / "perf_sorted.dat"
            with SortedList(reader, sorted_filepath) as sorted_list:
                sorted_list.sort(chunk_size=200)

                # Step 3: Set up algorithms
                linear_search = LinearSearch(reader, track_performance=True)
                binary_search = BinarySearch(sorted_list)

                # Step 4: Test same searches on both algorithms
                test_usernames = [reader[i] for i in [0, 100, 500, 800, 999]]

                linear_times = []
                binary_times = []

                total_linear_comparisons = 0
                total_binary_comparisons = 0

                for username in test_usernames:
                    # Linear search
                    linear_result = linear_search.search(username)
                    assert linear_result.found is True
                    linear_times.append(linear_result.time_taken)
                    total_linear_comparisons += linear_result.comparisons

                    # Binary search
                    binary_result = binary_search.search(username)
                    assert binary_result.found is True
                    binary_times.append(binary_result.time_taken)
                    total_binary_comparisons += binary_result.comparisons

                # On average, binary search should use fewer comparisons for larger datasets
                # For small datasets or early elements, linear search might be better for individual searches
                # But binary search should be consistently good across all searches
                assert (
                    total_binary_comparisons <= total_linear_comparisons * 1.5
                )  # Allow some tolerance

                # Step 5: Verify performance characteristics
                linear_stats = linear_search.get_performance_stats()
                assert linear_stats["total_searches"] == len(test_usernames)
                assert linear_stats["total_comparisons"] > 0
                assert linear_stats["avg_comparisons"] > 0

    def test_data_integrity_workflow(self):
        """Test data integrity throughout the workflow."""
        # Step 1: Generate deterministic data
        generator = UsernameGenerator(seed=999)
        original_usernames = list(generator.generate_batch(100))

        # Step 2: Store and read back
        data_filepath = Path(self.temp_dir) / "integrity_data.dat"
        storage = EfficientUsernameStorage(data_filepath)

        # Use generator again (same seed) to store
        generator2 = UsernameGenerator(seed=999)
        storage.store_usernames(generator2, 100, 25)

        # Step 3: Verify data integrity through reader
        with UsernameReader(data_filepath) as reader:
            read_usernames = list(reader)
            assert read_usernames == original_usernames

            # Step 4: Test sorting preserves all data
            sorted_filepath = Path(self.temp_dir) / "integrity_sorted.dat"
            with SortedList(reader, sorted_filepath) as sorted_list:
                sorted_list.sort(chunk_size=30)

                sorted_usernames = [sorted_list[i] for i in range(len(sorted_list))]
                assert len(sorted_usernames) == len(original_usernames)
                assert set(sorted_usernames) == set(original_usernames)
                assert sorted_usernames == sorted(original_usernames)

    def test_error_handling_integration(self):
        """Test error handling in integrated workflows."""
        # Test with corrupted/missing files
        nonexistent_path = Path(self.temp_dir) / "nonexistent.dat"

        # Reader should fail gracefully
        with pytest.raises(FileNotFoundError):
            UsernameReader(nonexistent_path)

        # Create incomplete data (data file without index)
        incomplete_path = Path(self.temp_dir) / "incomplete.dat"
        with open(incomplete_path, "wb") as f:
            f.write(b"test data")

        with pytest.raises(FileNotFoundError):
            UsernameReader(incomplete_path)

    def test_memory_efficiency_workflow(self):
        """Test memory efficiency in workflows."""
        # This test verifies that large datasets can be processed
        # without loading everything into memory

        # Step 1: Generate larger dataset
        generator = UsernameGenerator(seed=1000)
        data_filepath = Path(self.temp_dir) / "memory_test.dat"
        storage = EfficientUsernameStorage(data_filepath)

        username_count = 5000  # Larger dataset
        storage.store_usernames(generator, username_count, 500)

        # Step 2: Process with limited memory operations
        with UsernameReader(data_filepath) as reader:
            # Random access should work efficiently
            test_indices = [0, 1000, 2500, 4000, 4999]
            for idx in test_indices:
                username = reader[idx]
                assert isinstance(username, str)
                assert len(username) > 0

            # Batch processing should work efficiently
            batch_count = 0
            for batch in reader.iter_batch(100):
                batch_count += 1
                assert len(batch) <= 100
                if batch_count >= 10:  # Process first 10 batches
                    break

            assert batch_count == 10

    def test_concurrent_access_workflow(self):
        """Test concurrent access patterns."""
        import threading

        # Step 1: Create shared data
        generator = UsernameGenerator(seed=2000)
        data_filepath = Path(self.temp_dir) / "concurrent_data.dat"
        storage = EfficientUsernameStorage(data_filepath)

        username_count = 1000
        storage.store_usernames(generator, username_count, 200)

        # Step 2: Test concurrent reading
        results = []
        errors = []

        def concurrent_reader(reader_id):
            try:
                with UsernameReader(data_filepath) as reader:
                    # Each thread reads different subset
                    start_idx = reader_id * 100
                    end_idx = min(start_idx + 100, len(reader))

                    thread_results = []
                    for i in range(start_idx, end_idx):
                        username = reader[i]
                        thread_results.append(username)

                    results.append((reader_id, thread_results))
            except Exception as e:
                errors.append(e)

        # Create multiple reader threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_reader, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0
        assert len(results) == 5

        # Verify no data corruption
        for reader_id, thread_results in results:
            assert len(thread_results) == 100  # Each thread should read 100 items

    def test_end_to_end_login_checker_simulation(self):
        """Test complete login checker simulation."""
        # Step 1: Generate user database
        generator = UsernameGenerator(seed=3000)
        db_filepath = Path(self.temp_dir) / "user_database.dat"
        storage = EfficientUsernameStorage(db_filepath)

        db_size = 10000
        storage.store_usernames(generator, db_size, 1000)

        # Step 2: Create multiple data structures for different access patterns
        with UsernameReader(db_filepath) as reader:
            # Linear search for unsorted data
            linear_checker = LinearSearch(reader, track_performance=True)

            # Create sorted version for binary search
            sorted_filepath = Path(self.temp_dir) / "sorted_database.dat"
            with SortedList(reader, sorted_filepath) as sorted_list:
                sorted_list.sort(chunk_size=1000)
                binary_checker = BinarySearch(sorted_list)

                # Create hashset for O(1) lookups
                hashset_path = str(Path(self.temp_dir) / "user_hashset.dat")
                hashset = DiskHashSet(
                    hashset_path, num_slots_power=14, create=True
                )  # 16K slots

                # Populate hashset
                for i in range(0, min(8000, len(reader)), 10):  # Sample every 10th user
                    username = reader[i]
                    hashset.add(username)

                # Step 3: Simulate login attempts
                test_usernames = [reader[i] for i in [0, 100, 1000, 5000, 9999]]
                non_existent = ["fake_user_1", "fake_user_2", "fake_user_3"]

                total_linear_comparisons = 0
                total_binary_comparisons = 0

                for username in test_usernames:
                    # Test all three methods
                    linear_result = linear_checker.search(username)
                    binary_result = binary_checker.search(username)
                    hashset_result = hashset.contains(username)

                    assert linear_result.found is True
                    assert binary_result.found is True
                    # Note: hashset might not contain all users (we sampled)

                    total_linear_comparisons += linear_result.comparisons
                    total_binary_comparisons += binary_result.comparisons

                # On average, binary search should be more efficient for larger datasets
                assert (
                    total_binary_comparisons <= total_linear_comparisons * 1.2
                )  # Allow some tolerance

                for fake_username in non_existent:
                    linear_result = linear_checker.search(fake_username)
                    binary_result = binary_checker.search(fake_username)
                    hashset_result = hashset.contains(fake_username)

                    assert linear_result.found is False
                    assert binary_result.found is False
                    assert hashset_result is False

                # Step 4: Verify performance characteristics
                stats = linear_checker.get_performance_stats()
                total_expected_searches = len(test_usernames) + len(
                    non_existent
                )  # 5 + 3 = 8
                assert stats["total_searches"] == total_expected_searches
                assert stats["avg_comparisons"] > 0

                hashset.close()

    def test_data_format_compatibility(self):
        """Test compatibility between different components."""
        # Step 1: Create data with writer
        writer_filepath = Path(self.temp_dir) / "writer_test.dat"
        writer = UsernameWriter(writer_filepath)

        test_usernames = ["alice", "bob", "charlie", "diana", "eve"]
        writer.write_usernames(iter(test_usernames), batch_size=2)

        # Step 2: Read with reader
        with UsernameReader(writer_filepath) as reader:
            read_usernames = list(reader)
            assert read_usernames == test_usernames

            # Step 3: Test algorithms work with writer-created data
            linear_search = LinearSearch(reader)
            for i, username in enumerate(test_usernames):
                result = linear_search.search(username)
                assert result.found is True
                assert result.index == i
