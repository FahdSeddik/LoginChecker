"""
Comprehensive tests for DiskHashSet class.

Tests cover hash operations, collision handling, persistence, and error scenarios.
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.data_structures.disk_hashset import (
    FLAG_EMPTY,
    HEADER_SIZE,
    MAX_KEY_BYTES,
    SLOT_SIZE,
    DiskHashSet,
    fnv1a_64,
)


class TestFNVHash:
    """Test suite for FNV-1a hash function."""

    def test_fnv1a_64_basic(self):
        """Test basic FNV-1a hash functionality."""
        # Test empty string
        assert fnv1a_64(b"") == 0xCBF29CE484222325

        # Test simple strings
        hash1 = fnv1a_64(b"hello")
        hash2 = fnv1a_64(b"world")

        assert hash1 != hash2
        assert isinstance(hash1, int)
        assert isinstance(hash2, int)

    def test_fnv1a_64_consistency(self):
        """Test hash consistency."""
        data = b"test_string"
        hash1 = fnv1a_64(data)
        hash2 = fnv1a_64(data)

        assert hash1 == hash2

    def test_fnv1a_64_different_inputs(self):
        """Test that different inputs produce different hashes."""
        inputs = [b"alice", b"bob", b"charlie", b"diana", b"eve"]
        hashes = [fnv1a_64(inp) for inp in inputs]

        # All hashes should be different (very high probability)
        assert len(set(hashes)) == len(hashes)

    def test_fnv1a_64_unicode(self):
        """Test hash with Unicode data."""
        unicode_data = "测试".encode("utf-8")
        hash_value = fnv1a_64(unicode_data)

        assert isinstance(hash_value, int)
        assert hash_value != 0

    def test_fnv1a_64_large_data(self):
        """Test hash with larger data."""
        large_data = b"x" * 1000
        hash_value = fnv1a_64(large_data)

        assert isinstance(hash_value, int)


class TestDiskHashSet:
    """Test suite for DiskHashSet class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = str(Path(self.temp_dir) / "test_hashset.dat")

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

    def test_init_create_new(self):
        """Test creating a new hashset file."""
        hashset = DiskHashSet(self.test_path, num_slots_power=3, create=True)

        assert hashset.num_slots == 8  # 2^3
        assert hashset.capacity_mask == 7  # num_slots - 1
        assert Path(self.test_path).exists()

        # Verify file size
        expected_size = HEADER_SIZE + hashset.num_slots * SLOT_SIZE
        actual_size = Path(self.test_path).stat().st_size
        assert actual_size == expected_size

        hashset.close()

    def test_init_open_existing(self):
        """Test opening an existing hashset file."""
        # Create hashset
        hashset1 = DiskHashSet(self.test_path, num_slots_power=4, create=True)
        hashset1.add("test_key")
        hashset1.close()

        # Open existing
        hashset2 = DiskHashSet(self.test_path, create=False)
        assert hashset2.contains("test_key")
        assert hashset2.num_slots == 16  # Should read from file

        hashset2.close()

    def test_init_invalid_num_slots_power(self):
        """Test initialization with invalid num_slots_power."""
        with pytest.raises(AssertionError):
            DiskHashSet(self.test_path, num_slots_power=2, create=True)

    def test_add_basic(self):
        """Test basic add functionality."""
        hashset = DiskHashSet(self.test_path, num_slots_power=3, create=True)

        # Add new key
        result1 = hashset.add("alice")
        assert result1 is True

        # Add duplicate key
        result2 = hashset.add("alice")
        assert result2 is False

        hashset.close()

    def test_contains_basic(self):
        """Test basic contains functionality."""
        hashset = DiskHashSet(self.test_path, num_slots_power=3, create=True)

        # Key not present
        assert hashset.contains("alice") is False

        # Add key and test
        hashset.add("alice")
        assert hashset.contains("alice") is True

        # Different key
        assert hashset.contains("bob") is False

        hashset.close()

    def test_remove_basic(self):
        """Test basic remove functionality."""
        hashset = DiskHashSet(self.test_path, num_slots_power=3, create=True)

        # Remove non-existent key
        result1 = hashset.remove("alice")
        assert result1 is False

        # Add and remove key
        hashset.add("alice")
        assert hashset.contains("alice") is True

        result2 = hashset.remove("alice")
        assert result2 is True
        assert hashset.contains("alice") is False

        # Remove again
        result3 = hashset.remove("alice")
        assert result3 is False

        hashset.close()

    def test_add_after_remove(self):
        """Test adding key after removal (tombstone handling)."""
        hashset = DiskHashSet(self.test_path, num_slots_power=3, create=True)

        # Add, remove, add again
        hashset.add("alice")
        hashset.remove("alice")

        result = hashset.add("alice")
        assert result is True
        assert hashset.contains("alice") is True

        hashset.close()

    @pytest.mark.parametrize(
        "keys",
        [
            ["alice", "bob", "charlie"],
            ["a", "bb", "ccc", "dddd"],
            ["user1", "user2", "user3", "user4", "user5"],
        ],
    )
    def test_multiple_keys(self, keys):
        """Test with multiple keys."""
        hashset = DiskHashSet(self.test_path, num_slots_power=4, create=True)

        # Add all keys
        for key in keys:
            result = hashset.add(key)
            assert result is True

        # Verify all keys exist
        for key in keys:
            assert hashset.contains(key) is True

        # Verify duplicates return False
        for key in keys:
            result = hashset.add(key)
            assert result is False

        hashset.close()

    def test_collision_handling(self):
        """Test collision handling with linear probing."""
        # Use very small hashset to force collisions
        hashset = DiskHashSet(self.test_path, num_slots_power=3, create=True)  # 8 slots

        # Add enough keys to likely cause collisions
        keys = [f"key_{i}" for i in range(5)]

        for key in keys:
            hashset.add(key)

        # All keys should be retrievable despite collisions
        for key in keys:
            assert hashset.contains(key) is True

        hashset.close()

    def test_key_length_validation(self):
        """Test key length validation."""
        hashset = DiskHashSet(self.test_path, num_slots_power=3, create=True)

        # Valid key
        hashset.add("valid_key")

        # Key too long (> 20 bytes when encoded)
        long_key = "x" * (MAX_KEY_BYTES + 1)
        with pytest.raises(ValueError, match="key too long"):
            hashset.add(long_key)

        hashset.close()

    def test_unicode_keys(self):
        """Test handling of Unicode keys."""
        hashset = DiskHashSet(self.test_path, num_slots_power=3, create=True)

        unicode_keys = ["alice", "böb", "测试", "🙂"]

        for key in unicode_keys:
            try:
                hashset.add(key)
                assert hashset.contains(key) is True
            except ValueError:
                # Skip if key is too long when UTF-8 encoded
                encoded_len = len(key.encode("utf-8"))
                assert encoded_len > MAX_KEY_BYTES

        hashset.close()

    def test_bulk_load(self):
        """Test bulk loading functionality."""
        hashset = DiskHashSet(self.test_path, num_slots_power=4, create=True)

        keys = [f"bulk_key_{i}" for i in range(10)]
        hashset.bulk_load(keys)

        # Verify all keys were loaded
        for key in keys:
            assert hashset.contains(key) is True

        hashset.close()

    def test_persistence(self):
        """Test that data persists across sessions."""
        keys = ["persistent1", "persistent2", "persistent3"]

        # First session: add data
        hashset1 = DiskHashSet(self.test_path, num_slots_power=3, create=True)
        for key in keys:
            hashset1.add(key)
        hashset1.sync()  # Ensure data is written
        hashset1.close()

        # Second session: verify data exists
        hashset2 = DiskHashSet(self.test_path, create=False)
        for key in keys:
            assert hashset2.contains(key) is True
        hashset2.close()

    def test_sync_functionality(self):
        """Test sync functionality."""
        hashset = DiskHashSet(self.test_path, num_slots_power=3, create=True)

        hashset.add("sync_test")
        hashset.sync()  # Should complete without error

        assert hashset.contains("sync_test") is True

        hashset.close()

    def test_table_full_error(self):
        """Test error when hash table becomes full."""
        # Create very small table
        hashset = DiskHashSet(self.test_path, num_slots_power=3, create=True)  # 8 slots

        # Fill the table
        keys_added = []
        try:
            for i in range(20):  # Try to add more than slots available
                key = f"key_{i:02d}"
                result = hashset.add(key)
                if result:
                    keys_added.append(key)
        except RuntimeError as e:
            assert "Hash table is full" in str(e)

        # Should have added some keys before failing
        assert len(keys_added) > 0

        hashset.close()

    def test_edge_case_empty_key(self):
        """Test handling of empty key."""
        hashset = DiskHashSet(self.test_path, num_slots_power=3, create=True)

        # Empty string should be valid
        result = hashset.add("")
        assert result is True
        assert hashset.contains("") is True

        hashset.close()

    def test_edge_case_max_length_key(self):
        """Test key exactly at maximum length."""
        hashset = DiskHashSet(self.test_path, num_slots_power=3, create=True)

        # Key exactly at max bytes
        max_key = "x" * MAX_KEY_BYTES

        result = hashset.add(max_key)
        assert result is True
        assert hashset.contains(max_key) is True

        hashset.close()

    def test_file_header_validation(self):
        """Test file header validation."""
        # Create invalid file
        with open(self.test_path, "wb") as f:
            f.write(b"INVALID_HEADER" + b"\x00" * (HEADER_SIZE - 14))

        with pytest.raises(ValueError, match="Bad file magic"):
            DiskHashSet(self.test_path, create=False)

    def test_different_slot_powers(self):
        """Test different num_slots_power values."""
        for power in [3, 4, 5, 6]:
            test_path = str(Path(self.temp_dir) / f"test_{power}.dat")
            hashset = DiskHashSet(test_path, num_slots_power=power, create=True)

            expected_slots = 1 << power
            assert hashset.num_slots == expected_slots

            # Add some data
            hashset.add(f"test_key_{power}")
            assert hashset.contains(f"test_key_{power}") is True

            hashset.close()

    def test_error_handling_invalid_path(self):
        """Test error handling with invalid file paths."""
        with pytest.raises((OSError, IOError, PermissionError)):
            DiskHashSet("/nonexistent/path/file.dat", num_slots_power=3, create=True)

    def test_context_manager_behavior(self):
        """Test proper resource cleanup."""
        hashset = DiskHashSet(self.test_path, num_slots_power=3, create=True)

        hashset.add("test")
        assert hashset.contains("test") is True

        # Manual close should work
        hashset.close()

    def test_slot_operations(self):
        """Test internal slot operations."""
        hashset = DiskHashSet(self.test_path, num_slots_power=3, create=True)

        # Test slot offset calculation
        offset0 = hashset._slot_offset(0)
        offset1 = hashset._slot_offset(1)

        assert offset1 == offset0 + SLOT_SIZE

        # Test reading empty slot
        flag, key_len, fingerprint, key_bytes, _ = hashset._read_slot(0)
        assert flag == FLAG_EMPTY
        assert key_len == 0

        hashset.close()

    def test_key_encoding(self):
        """Test internal key encoding."""
        hashset = DiskHashSet(self.test_path, num_slots_power=3, create=True)

        # Test ASCII key
        encoded = hashset._encode_key("ascii")
        assert encoded == b"ascii"

        # Test Unicode key
        encoded_unicode = hashset._encode_key("café")
        assert encoded_unicode == "café".encode("utf-8")

        hashset.close()

    def test_fingerprint_collision_handling(self):
        """Test handling when fingerprints collide but keys differ."""
        hashset = DiskHashSet(self.test_path, num_slots_power=3, create=True)

        # This is hard to test deterministically, but we can verify
        # that the hashset correctly distinguishes between different keys
        # even if they might hash to the same slot

        keys = ["collision1", "collision2", "collision3"]

        for key in keys:
            hashset.add(key)

        # All keys should be distinguishable
        for key in keys:
            assert hashset.contains(key) is True

        # Non-existent similar keys should not be found
        assert hashset.contains("collision4") is False

        hashset.close()

    def test_remove_with_linear_probing(self):
        """Test removal when linear probing was used for insertion."""
        hashset = DiskHashSet(self.test_path, num_slots_power=3, create=True)

        # Add several keys that might probe
        keys = [f"probe_key_{i}" for i in range(5)]

        for key in keys:
            hashset.add(key)

        # Remove middle key
        removed_key = keys[2]
        assert hashset.remove(removed_key) is True
        assert hashset.contains(removed_key) is False

        # Other keys should still be accessible
        for key in keys:
            if key != removed_key:
                assert hashset.contains(key) is True

        hashset.close()

    def test_large_key_set(self):
        """Test with larger number of keys."""
        hashset = DiskHashSet(
            self.test_path, num_slots_power=8, create=True
        )  # 256 slots

        # Add many keys
        keys = [f"large_key_{i:03d}" for i in range(100)]

        for key in keys:
            hashset.add(key)

        # Verify all keys
        for key in keys:
            assert hashset.contains(key) is True

        # Test removal of some keys
        keys_to_remove = keys[::10]  # Every 10th key
        for key in keys_to_remove:
            assert hashset.remove(key) is True
            assert hashset.contains(key) is False

        # Verify remaining keys still exist
        for key in keys:
            if key not in keys_to_remove:
                assert hashset.contains(key) is True

        hashset.close()

    def test_stress_add_remove_cycle(self):
        """Test stress scenario with add/remove cycles."""
        hashset = DiskHashSet(
            self.test_path, num_slots_power=6, create=True
        )  # 64 slots

        # Perform multiple add/remove cycles
        for cycle in range(3):
            keys = [f"cycle_{cycle}_key_{i}" for i in range(20)]

            # Add keys
            for key in keys:
                assert hashset.add(key) is True

            # Verify keys
            for key in keys:
                assert hashset.contains(key) is True

            # Remove half the keys
            keys_to_remove = keys[::2]
            for key in keys_to_remove:
                assert hashset.remove(key) is True

            # Verify removal
            for key in keys_to_remove:
                assert hashset.contains(key) is False

        hashset.close()

    def test_concurrent_access_safety(self):
        """Test basic concurrent access safety."""
        import threading

        hashset = DiskHashSet(self.test_path, num_slots_power=6, create=True)
        errors = []

        def add_keys(start_idx, count):
            try:
                for i in range(count):
                    key = f"concurrent_{start_idx}_{i}"
                    hashset.add(key)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=add_keys, args=(i, 10))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should complete without errors
        assert len(errors) == 0

        # Verify some keys were added
        assert hashset.contains("concurrent_0_0") is True
        assert hashset.contains("concurrent_1_5") is True
        assert hashset.contains("concurrent_2_9") is True

        hashset.close()
