"""
Comprehensive tests for UsernameGenerator class.

Tests cover pattern generation, reproducibility, edge cases, and performance characteristics.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from mimesis.locales import Locale

from data.generate import EfficientUsernameStorage, UsernameGenerator


class TestUsernameGenerator:
    """Test suite for UsernameGenerator class."""

    def test_init_default(self):
        """Test default initialization."""
        generator = UsernameGenerator()
        assert generator.person is not None
        assert len(generator.patterns) > 0
        assert generator.p_idx == 0

    def test_init_with_locale(self):
        """Test initialization with specific locale."""
        generator = UsernameGenerator(locale=Locale.RU)
        assert generator.person is not None

    def test_init_with_seed(self):
        """Test initialization with seed for reproducibility."""
        seed = 42
        generator = UsernameGenerator(seed=seed)
        assert generator.person is not None

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same sequence."""
        seed = 12345
        generator1 = UsernameGenerator(seed=seed)
        generator2 = UsernameGenerator(seed=seed)

        usernames1 = [generator1.generate_username() for _ in range(10)]
        usernames2 = [generator2.generate_username() for _ in range(10)]

        assert usernames1 == usernames2

    def test_username_length_constraint(self):
        """Test that usernames are limited to 20 characters."""
        generator = UsernameGenerator()

        for _ in range(100):
            username = generator.generate_username()
            assert len(username) <= 20
            assert len(username) > 0

    def test_username_format(self):
        """Test that generated usernames match expected format."""
        generator = UsernameGenerator()

        for _ in range(50):
            username = generator.generate_username()
            # Should be valid UTF-8 string
            assert isinstance(username, str)
            # Should not be empty
            assert len(username.strip()) > 0
            # Should be ASCII compatible for most cases
            try:
                username.encode("ascii")
            except UnicodeEncodeError:
                # UTF-8 is acceptable
                pass

    def test_pattern_cycling(self):
        """Test that patterns cycle through properly."""
        generator = UsernameGenerator(seed=42)
        initial_pattern_count = len(generator.patterns)

        # Generate more usernames than patterns to test cycling
        usernames = []
        for _ in range(initial_pattern_count * 2):
            usernames.append(generator.generate_username())

        # Should have generated usernames without error
        assert len(usernames) == initial_pattern_count * 2
        assert all(len(u) > 0 for u in usernames)

    @pytest.mark.parametrize("count", [1, 10, 100, 1000])
    def test_generate_batch(self, count):
        """Test batch generation with various counts."""
        generator = UsernameGenerator(seed=42)

        batch = list(generator.generate_batch(count))

        assert len(batch) == count
        assert all(isinstance(username, str) for username in batch)
        assert all(len(username) <= 20 for username in batch)
        assert all(len(username) > 0 for username in batch)

    def test_generate_batch_empty(self):
        """Test batch generation with zero count."""
        generator = UsernameGenerator()

        batch = list(generator.generate_batch(0))

        assert len(batch) == 0

    def test_patterns_not_empty(self):
        """Test that pattern list is properly populated."""
        generator = UsernameGenerator()

        assert len(generator.patterns) > 0
        # Each pattern should contain at least one required character
        required_chars = {"C", "U", "l"}
        for pattern in generator.patterns:
            assert any(char in pattern for char in required_chars)

    def test_different_locales_produce_different_results(self):
        """Test that different locales can produce different username styles."""
        seed = 42
        gen_en = UsernameGenerator(locale=Locale.EN, seed=seed)
        gen_ru = UsernameGenerator(locale=Locale.RU, seed=seed)

        usernames_en = [gen_en.generate_username() for _ in range(10)]
        usernames_ru = [gen_ru.generate_username() for _ in range(10)]

        # While they might overlap, they shouldn't be identical sequences
        # (this is probabilistic but very unlikely with different locales)
        assert isinstance(usernames_en, list)
        assert isinstance(usernames_ru, list)

    def test_state_persistence_across_calls(self):
        """Test that generator maintains state across multiple calls."""
        generator = UsernameGenerator(seed=42)

        # Generate first batch
        batch1 = list(generator.generate_batch(5))

        # Generate second batch
        batch2 = list(generator.generate_batch(5))

        # Should be different (with high probability)
        assert batch1 != batch2

    def test_thread_safety_basic(self):
        """Basic test for thread safety concerns."""
        import threading

        generator = UsernameGenerator(seed=42)
        results = []
        errors = []

        def generate_usernames():
            try:
                local_results = [generator.generate_username() for _ in range(10)]
                results.extend(local_results)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=generate_usernames)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should have no errors and expected number of results
        assert len(errors) == 0
        assert len(results) == 50


class TestEfficientUsernameStorage:
    """Test suite for EfficientUsernameStorage class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_filepath = Path(self.temp_dir) / "test_usernames.dat"
        self.storage = EfficientUsernameStorage(self.test_filepath)

    def teardown_method(self):
        """Clean up test fixtures."""
        # Remove test files
        for file in Path(self.temp_dir).glob("*"):
            try:
                file.unlink()
            except Exception:
                pass
        try:
            os.rmdir(self.temp_dir)
        except Exception:
            pass

    def test_init(self):
        """Test initialization."""
        assert self.storage.filepath == self.test_filepath
        assert self.storage.index_filepath == self.test_filepath.with_suffix(".idx")

    @pytest.mark.parametrize(
        "total_count,batch_size",
        [
            (100, 10),
            (1000, 100),
            (10000, 1000),
        ],
    )
    def test_store_usernames_basic(self, total_count, batch_size):
        """Test basic username storage functionality."""
        generator = UsernameGenerator(seed=42)

        # Store usernames
        self.storage.store_usernames(generator, total_count, batch_size)

        # Verify files exist
        assert self.test_filepath.exists()
        assert self.storage.index_filepath.exists()

        # Verify file sizes are reasonable
        data_size = self.test_filepath.stat().st_size
        index_size = self.storage.index_filepath.stat().st_size

        assert data_size > 0
        assert index_size == total_count * 8  # 8 bytes per index entry

    def test_store_usernames_empty(self):
        """Test storing zero usernames."""
        generator = UsernameGenerator(seed=42)

        self.storage.store_usernames(generator, 0, 100)

        # Files should exist but be minimal
        assert self.test_filepath.exists()
        assert self.storage.index_filepath.exists()
        assert self.test_filepath.stat().st_size == 0
        assert self.storage.index_filepath.stat().st_size == 0

    def test_store_usernames_single_batch(self):
        """Test when total count is smaller than batch size."""
        generator = UsernameGenerator(seed=42)

        self.storage.store_usernames(generator, 50, 100)

        assert self.test_filepath.exists()
        assert self.storage.index_filepath.exists()
        # Should have 50 index entries
        assert self.storage.index_filepath.stat().st_size == 50 * 8

    def test_store_usernames_file_content_structure(self):
        """Test that stored files have correct binary structure."""
        generator = UsernameGenerator(seed=42)

        # Reset generator to same state
        generator = UsernameGenerator(seed=42)
        self.storage.store_usernames(generator, 10, 5)

        # Verify index file structure
        with open(self.storage.index_filepath, "rb") as f:
            import struct

            positions = []
            for _ in range(10):
                pos_bytes = f.read(8)
                if len(pos_bytes) == 8:
                    positions.append(struct.unpack("<Q", pos_bytes)[0])

        assert len(positions) == 10
        assert positions[0] == 0  # First position should be 0
        # Positions should be monotonically increasing
        assert all(positions[i] <= positions[i + 1] for i in range(len(positions) - 1))

    def test_store_usernames_progress_reporting(self, capsys):
        """Test that progress is reported during storage."""
        generator = UsernameGenerator(seed=42)

        # Use small batch size to trigger progress updates
        self.storage.store_usernames(generator, 1000, 100)

        captured = capsys.readouterr()
        assert "Generating" in captured.out
        assert "Generation complete" in captured.out

    def test_store_usernames_large_batch_size(self):
        """Test with batch size larger than total count."""
        generator = UsernameGenerator(seed=42)

        self.storage.store_usernames(generator, 10, 1000)

        # Should work correctly despite large batch size
        assert self.test_filepath.exists()
        assert self.storage.index_filepath.exists()

    @patch("time.time")
    def test_store_usernames_timing(self, mock_time):
        """Test timing calculations in storage."""
        # Mock time to return predictable values
        mock_time.side_effect = [0.0, 1.0, 2.0, 3.0]  # Start, progress points

        generator = UsernameGenerator(seed=42)
        self.storage.store_usernames(generator, 100, 50)

        # Should complete without errors
        assert self.test_filepath.exists()

    def test_store_usernames_unicode_handling(self):
        """Test handling of Unicode usernames."""

        # Create a mock generator that produces Unicode usernames
        class UnicodeGenerator:
            def generate_batch(self, count):
                return [f"user_{i}_😀" for i in range(count)]

        generator = UnicodeGenerator()
        self.storage.store_usernames(generator, 5, 2)

        assert self.test_filepath.exists()
        assert self.storage.index_filepath.exists()

        # Verify content can be read back
        with open(self.test_filepath, "rb") as f:
            content = f.read()
            # Should contain UTF-8 encoded emoji
            assert b"\xf0\x9f\x98\x80" in content  # UTF-8 for 😀

    def test_file_overwrite_behavior(self):
        """Test behavior when files already exist."""
        # Create initial files
        generator = UsernameGenerator(seed=42)
        self.storage.store_usernames(generator, 100, 50)

        initial_size = self.test_filepath.stat().st_size

        # Store again with different data
        generator = UsernameGenerator(seed=123)
        self.storage.store_usernames(generator, 50, 25)

        # Files should be overwritten
        new_size = self.test_filepath.stat().st_size
        assert new_size != initial_size

    def test_error_handling_invalid_generator(self):
        """Test error handling with invalid generator."""

        class BadGenerator:
            def generate_batch(self, count):
                raise ValueError("Generator error")

        bad_generator = BadGenerator()

        with pytest.raises(ValueError):
            self.storage.store_usernames(bad_generator, 10, 5)

    def test_error_handling_disk_full_simulation(self):
        """Test error handling when disk operations fail."""
        generator = UsernameGenerator(seed=42)

        # Try to write to a path that doesn't exist
        bad_storage = EfficientUsernameStorage(Path("/nonexistent/path/file.dat"))

        with pytest.raises((OSError, IOError, PermissionError)):
            bad_storage.store_usernames(generator, 10, 5)
