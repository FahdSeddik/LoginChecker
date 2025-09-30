"""
Comprehensive tests for UsernameWriter class.

Tests cover binary format writing, index creation, batch operations, and error handling.
"""

import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from data.writer import UsernameWriter


class TestUsernameWriter:
    """Test suite for UsernameWriter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_filepath = Path(self.temp_dir) / "test_output.dat"
        self.writer = UsernameWriter(self.test_filepath)

    def teardown_method(self):
        """Clean up test fixtures."""
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
        assert self.writer.filepath == self.test_filepath
        assert self.writer.index_filepath == self.test_filepath.with_suffix(".idx")

    def test_write_usernames_basic(self):
        """Test basic username writing functionality."""
        usernames = ["alice", "bob", "charlie", "diana", "eve"]

        self.writer.write_usernames(iter(usernames), batch_size=3)

        # Verify files exist
        assert self.test_filepath.exists()
        assert self.writer.index_filepath.exists()

        # Verify file sizes
        data_size = self.test_filepath.stat().st_size
        index_size = self.writer.index_filepath.stat().st_size

        assert data_size > 0
        assert index_size == len(usernames) * 8  # 8 bytes per index entry

    def test_write_usernames_empty(self):
        """Test writing empty username list."""
        self.writer.write_usernames(iter([]), batch_size=10)

        # Files should exist but be empty
        assert self.test_filepath.exists()
        assert self.writer.index_filepath.exists()
        assert self.test_filepath.stat().st_size == 0
        assert self.writer.index_filepath.stat().st_size == 0

    def test_write_usernames_single(self):
        """Test writing single username."""
        username = ["single_user"]

        self.writer.write_usernames(iter(username), batch_size=10)

        # Verify data file content
        with open(self.test_filepath, "rb") as f:
            content = f.read()
            assert content == b"single_user"

        # Verify index file content
        with open(self.writer.index_filepath, "rb") as f:
            position = struct.unpack("<Q", f.read(8))[0]
            assert position == 0

    def test_write_usernames_file_format(self):
        """Test that files are written in correct binary format."""
        usernames = ["alice", "bob", "charlie"]

        self.writer.write_usernames(iter(usernames), batch_size=2)

        # Verify data file structure
        with open(self.test_filepath, "rb") as f:
            content = f.read()
            expected = b"alicebobcharlie"
            assert content == expected

        # Verify index file structure
        with open(self.writer.index_filepath, "rb") as f:
            positions = []
            while True:
                pos_bytes = f.read(8)
                if len(pos_bytes) != 8:
                    break
                positions.append(struct.unpack("<Q", pos_bytes)[0])

        expected_positions = [0, 5, 8]  # alice(5), bob(3), charlie
        assert positions == expected_positions

    def test_write_usernames_unicode(self):
        """Test writing Unicode usernames."""
        usernames = ["alice", "böb", "charlie_😀"]

        self.writer.write_usernames(iter(usernames), batch_size=2)

        # Verify data file can handle UTF-8
        with open(self.test_filepath, "rb") as f:
            content = f.read()

        # Should contain UTF-8 encoded content
        expected = "alice" + "böb" + "charlie_😀"
        assert content == expected.encode("utf-8")

        # Verify index positions account for UTF-8 byte lengths
        with open(self.writer.index_filepath, "rb") as f:
            positions = []
            while True:
                pos_bytes = f.read(8)
                if len(pos_bytes) != 8:
                    break
                positions.append(struct.unpack("<Q", pos_bytes)[0])

        # Calculate expected positions based on UTF-8 byte lengths
        pos1 = 0
        pos2 = len("alice".encode("utf-8"))
        pos3 = pos2 + len("böb".encode("utf-8"))

        assert positions == [pos1, pos2, pos3]

    @pytest.mark.parametrize("batch_size", [1, 2, 5, 10, 100])
    def test_write_usernames_various_batch_sizes(self, batch_size):
        """Test writing with various batch sizes."""
        usernames = [f"user_{i:03d}" for i in range(10)]

        self.writer.write_usernames(iter(usernames), batch_size=batch_size)

        # Verify all usernames are written correctly regardless of batch size
        assert self.test_filepath.exists()
        assert self.writer.index_filepath.exists()

        # Verify index has correct number of entries
        index_size = self.writer.index_filepath.stat().st_size
        assert index_size == len(usernames) * 8

    def test_write_usernames_large_batch_size(self):
        """Test writing with batch size larger than data."""
        usernames = ["alice", "bob", "charlie"]

        self.writer.write_usernames(iter(usernames), batch_size=100)

        # Should work correctly with large batch size
        assert self.test_filepath.exists()
        assert self.writer.index_filepath.exists()

        with open(self.test_filepath, "rb") as f:
            content = f.read()
            assert content == b"alicebobcharlie"

    def test_write_usernames_progress_reporting(self, capsys):
        """Test progress reporting during writing."""
        # Generate enough usernames to trigger progress reporting
        usernames = [f"user_{i:06d}" for i in range(2000)]

        self.writer.write_usernames(iter(usernames), batch_size=100)

        captured = capsys.readouterr()
        assert "Writing usernames" in captured.out
        assert "Writing complete" in captured.out

    def test_write_usernames_timing_calculation(self):
        """Test that timing calculations work correctly."""
        usernames = ["alice", "bob", "charlie"]

        with patch("time.time") as mock_time:
            # Mock time progression
            mock_time.side_effect = [0.0, 1.0, 2.0]  # start, progress, end

            self.writer.write_usernames(iter(usernames), batch_size=2)

        # Should complete without errors
        assert self.test_filepath.exists()

    def test_write_batch_internal_method(self):
        """Test the internal _write_batch method."""
        batch = ["alice", "bob", "charlie"]

        with (
            open(self.test_filepath, "wb") as data_file,
            open(self.writer.index_filepath, "wb") as index_file,
        ):
            self.writer._write_batch(batch, data_file, index_file, 0)

        # Verify data file
        with open(self.test_filepath, "rb") as f:
            content = f.read()
            assert content == b"alicebobcharlie"

        # Verify index file
        with open(self.writer.index_filepath, "rb") as f:
            positions = []
            for _ in range(3):
                pos_bytes = f.read(8)
                if len(pos_bytes) == 8:
                    positions.append(struct.unpack("<Q", pos_bytes)[0])

        assert positions == [0, 5, 8]

    def test_write_batch_with_offset(self):
        """Test _write_batch with non-zero start position."""
        batch = ["diana", "eve"]
        start_position = 100

        with (
            open(self.test_filepath, "wb") as data_file,
            open(self.writer.index_filepath, "wb") as index_file,
        ):
            self.writer._write_batch(batch, data_file, index_file, start_position)

        # Verify index positions account for offset
        with open(self.writer.index_filepath, "rb") as f:
            positions = []
            for _ in range(2):
                pos_bytes = f.read(8)
                if len(pos_bytes) == 8:
                    positions.append(struct.unpack("<Q", pos_bytes)[0])

        expected_positions = [100, 100 + len("diana")]
        assert positions == expected_positions

    def test_write_usernames_generator_input(self):
        """Test writing from generator input."""

        def username_generator():
            for i in range(5):
                yield f"gen_user_{i}"

        self.writer.write_usernames(username_generator(), batch_size=3)

        # Verify all generated usernames were written
        assert self.test_filepath.exists()

        with open(self.test_filepath, "rb") as f:
            content = f.read().decode("utf-8")
            assert "gen_user_0" in content
            assert "gen_user_4" in content

    def test_write_usernames_long_usernames(self):
        """Test writing usernames at maximum length."""
        # Create usernames at or near 20-character limit
        usernames = [
            "a" * 20,  # exactly 20 chars
            "b" * 19,  # 19 chars
            "c" * 15,  # 15 chars
        ]

        self.writer.write_usernames(iter(usernames), batch_size=2)

        # Verify data is written correctly
        with open(self.test_filepath, "rb") as f:
            content = f.read()
            expected = ("a" * 20) + ("b" * 19) + ("c" * 15)
            assert content == expected.encode("utf-8")

    def test_write_usernames_special_characters(self):
        """Test writing usernames with special characters."""
        usernames = [
            "user.name",
            "user_name",
            "user-name",
            "user@domain",
            "user+tag",
        ]

        self.writer.write_usernames(iter(usernames), batch_size=3)

        # Verify special characters are preserved
        with open(self.test_filepath, "rb") as f:
            content = f.read().decode("utf-8")
            for username in usernames:
                assert username in content

    def test_error_handling_invalid_path(self):
        """Test error handling with invalid file paths."""
        invalid_writer = UsernameWriter(Path("/nonexistent/path/file.dat"))

        with pytest.raises((OSError, IOError, PermissionError)):
            invalid_writer.write_usernames(iter(["test"]), batch_size=1)

    def test_error_handling_disk_full_simulation(self):
        """Test behavior when disk operations might fail."""
        usernames = ["alice", "bob", "charlie"]

        # Create writer with a path in a directory that doesn't exist
        bad_path = Path(self.temp_dir) / "nonexistent" / "file.dat"
        bad_writer = UsernameWriter(bad_path)

        with pytest.raises((OSError, IOError)):
            bad_writer.write_usernames(iter(usernames), batch_size=2)

    def test_write_usernames_empty_strings(self):
        """Test handling of empty string usernames."""
        usernames = ["alice", "", "bob", "", "charlie"]

        self.writer.write_usernames(iter(usernames), batch_size=3)

        # Should handle empty strings correctly
        assert self.test_filepath.exists()

        # Verify index has correct number of entries
        index_size = self.writer.index_filepath.stat().st_size
        assert index_size == len(usernames) * 8

    def test_write_usernames_very_large_count(self):
        """Test writing with simulated large counts for progress reporting."""

        # Create iterator that reports large count for progress testing
        def large_username_generator():
            for i in range(1500):  # Enough to trigger multiple progress reports
                yield f"user_{i:04d}"

        self.writer.write_usernames(large_username_generator(), batch_size=100)

        # Verify successful completion
        assert self.test_filepath.exists()
        assert self.writer.index_filepath.exists()

    def test_cumulative_position_calculation(self):
        """Test that cumulative positions are calculated correctly."""
        usernames = ["a", "bb", "ccc", "dddd"]  # Different lengths

        self.writer.write_usernames(iter(usernames), batch_size=2)

        # Verify cumulative positions
        with open(self.writer.index_filepath, "rb") as f:
            positions = []
            while True:
                pos_bytes = f.read(8)
                if len(pos_bytes) != 8:
                    break
                positions.append(struct.unpack("<Q", pos_bytes)[0])

        # Expected positions: 0, 1, 3, 6
        expected = [0, 1, 3, 6]
        assert positions == expected

    def test_write_usernames_file_permissions(self):
        """Test file creation with appropriate permissions."""
        usernames = ["alice", "bob"]

        self.writer.write_usernames(iter(usernames), batch_size=1)

        # Verify files are readable
        assert os.access(self.test_filepath, os.R_OK)
        assert os.access(self.writer.index_filepath, os.R_OK)

    def test_write_usernames_concurrent_safety(self):
        """Test basic concurrent safety considerations."""
        import threading

        usernames1 = ["alice", "bob", "charlie"]
        usernames2 = ["diana", "eve", "frank"]

        # Use different files to avoid conflicts
        filepath1 = Path(self.temp_dir) / "concurrent1.dat"
        filepath2 = Path(self.temp_dir) / "concurrent2.dat"

        writer1 = UsernameWriter(filepath1)
        writer2 = UsernameWriter(filepath2)

        errors = []

        def write_data(writer, usernames):
            try:
                writer.write_usernames(iter(usernames), batch_size=2)
            except Exception as e:
                errors.append(e)

        # Create threads for concurrent writing
        thread1 = threading.Thread(target=write_data, args=(writer1, usernames1))
        thread2 = threading.Thread(target=write_data, args=(writer2, usernames2))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # Should complete without errors
        assert len(errors) == 0
        assert filepath1.exists()
        assert filepath2.exists()

    def test_write_usernames_batch_boundary_conditions(self):
        """Test batch processing at various boundary conditions."""
        # Test exact multiple of batch size
        usernames = [f"user_{i}" for i in range(10)]

        self.writer.write_usernames(iter(usernames), batch_size=5)

        # Verify all usernames written
        index_size = self.writer.index_filepath.stat().st_size
        assert index_size == len(usernames) * 8

        # Test non-multiple of batch size (remainder)
        filepath2 = Path(self.temp_dir) / "remainder.dat"
        writer2 = UsernameWriter(filepath2)

        usernames2 = [f"user_{i}" for i in range(7)]  # 7 with batch_size 3 = 3+3+1
        writer2.write_usernames(iter(usernames2), batch_size=3)

        index_size2 = filepath2.with_suffix(".idx").stat().st_size
        assert index_size2 == len(usernames2) * 8
