"""
Mathematical Foundation:
- False positive rate: eps <= (2 * b) / 2^f, where b is bucket size, f is fingerprint bits
- Load factor: lf = n / (buckets * b), typical values: ~0.95 for b=4, ~0.84 for b=2
- Two hash functions: h1 = hash(key), h2 = h1 XOR hash(fingerprint)
- Partial-key cuckoo hashing: store only fingerprint, compute alternate via XOR
"""

import math
import mmap
import os
import random
import struct
import tempfile
from typing import Optional, Tuple, Union

import mmh3


class CuckooFilter:
    """
    Cuckoo filter implementation with memory-mapped storage.

    Uses memory-mapped files for persistence and partial-key cuckoo hashing
    for efficient membership testing with deletion support.
    """

    # File format constants
    MAGIC = b"CUCKOOv1"
    HEADER_SIZE = 41  # 4KB aligned header
    VERSION = 1

    # Default parameters
    DEFAULT_BUCKET_SIZE = 4
    DEFAULT_MAX_KICKS = 500
    DEFAULT_LOAD_FACTORS = {2: 0.84, 4: 0.95, 8: 0.98}

    # Fingerprint constants
    FLAG_EMPTY = 0  # Empty slot marker

    def __init__(
        self,
        capacity: int,
        false_positive_rate: float,
        filepath: Optional[str] = None,
        bucket_size: int = DEFAULT_BUCKET_SIZE,
        fingerprint_bits: Optional[int] = None,
        seed: Optional[int] = None,
        max_kicks: int = DEFAULT_MAX_KICKS,
    ):
        """
        Initialize a Cuckoo filter with given capacity and false positive rate.

        Args:
            capacity: Expected number of elements (n)
            false_positive_rate: Desired false positive rate (eps)
            filepath: Optional file path for persistence (creates temp file if None)
            bucket_size: Number of slots per bucket (b), typically 2, 4, or 8
            fingerprint_bits: Bits per fingerprint (f), calculated if None
            seed: Random seed for hash functions (generates random if None)
            max_kicks: Maximum eviction attempts before resize/failure
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if not (0 < false_positive_rate < 1):
            raise ValueError("False positive rate must be between 0 and 1")
        if bucket_size not in [1, 2, 4, 8]:
            raise ValueError("Bucket size must be 1, 2, 4, or 8")

        self.capacity = capacity
        self.false_positive_rate = false_positive_rate
        self.bucket_size = bucket_size
        self.max_kicks = max_kicks
        self.seed: int = (
            seed if seed is not None else int.from_bytes(os.urandom(4), "little")
        )

        self._calculate_parameters(fingerprint_bits)
        self._closed = False
        self.inserted_count = 0  # Will be overridden if loading from existing file

        # Initialize storage (this may load inserted_count from file)
        self._initialize_storage(filepath)

        # Statistics
        self.total_kicks = 0
        self.failed_inserts = 0

    def _calculate_parameters(self, fingerprint_bits: Optional[int]):
        """Calculate optimal parameters: buckets, fingerprint size, storage."""
        target_load = self.DEFAULT_LOAD_FACTORS.get(self.bucket_size, 0.90)
        # Calculate bucket count: buckets = ceil(n / (lf * b))
        self.bucket_count = math.ceil(self.capacity / (target_load * self.bucket_size))

        # Make bucket_count a power of 2 for fast modulo
        self.bucket_count = 2 ** math.ceil(math.log2(self.bucket_count))
        self.bucket_mask = self.bucket_count - 1  # For fast modulo
        if fingerprint_bits is None:
            # Conservative formula: eps <= (2 * b) / 2^f  => f >= log2((2*b)/eps)
            min_fp_bits = math.ceil(
                math.log2((2 * self.bucket_size) / self.false_positive_rate)
            )
            self.fingerprint_bits = max(8, min_fp_bits)
        else:
            self.fingerprint_bits = fingerprint_bits

        # Calculate storage parameters
        self.bytes_per_slot = math.ceil(self.fingerprint_bits / 8)
        self.fingerprint_mask = (1 << self.fingerprint_bits) - 1
        self.payload_bytes = self.bucket_count * self.bucket_size * self.bytes_per_slot
        self.total_bytes = self.HEADER_SIZE + self.payload_bytes

        self.actual_false_positive_rate = (2 * self.bucket_size) / (
            2**self.fingerprint_bits
        )

    def _initialize_storage(self, filepath: Optional[str]):
        """Initialize memory-mapped storage with header and payload."""
        self._temp_file = None
        self._owns_file = False

        if filepath is None:
            self._temp_file = tempfile.NamedTemporaryFile(delete=False)
            filepath = self._temp_file.name
            self._owns_file = True

        self.filepath = filepath

        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                # Empty file, treat as new
                self._create_new_file()
            elif file_size < self.HEADER_SIZE:
                raise ValueError("Invalid file: file too small")
            else:
                try:
                    self._load_existing_file()
                except (ValueError, struct.error) as e:
                    # File exists but is corrupted, raise error
                    raise ValueError(f"Invalid file: {e}")
        else:
            self._create_new_file()

    def _create_new_file(self):
        """Create new file with header and zero-initialized payload."""
        with open(self.filepath, "w+b") as f:
            # Write header
            header = bytearray(self.HEADER_SIZE)
            header[:8] = self.MAGIC
            struct.pack_into("<B", header, 8, self.VERSION)
            struct.pack_into("<Q", header, 9, self.bucket_count)
            struct.pack_into("<I", header, 17, self.bucket_size)
            struct.pack_into("<I", header, 21, self.fingerprint_bits)
            struct.pack_into("<I", header, 25, self.max_kicks)
            struct.pack_into("<Q", header, 29, 0)  # count
            struct.pack_into("<I", header, 37, self.seed)

            f.write(header)

            # Zero-initialize payload
            f.write(b"\x00" * self.payload_bytes)
            f.flush()

        self._open_mmap()

    def _load_existing_file(self):
        """Load existing file and validate header."""
        with open(self.filepath, "r+b") as f:
            header = f.read(self.HEADER_SIZE)
        if len(header) < self.HEADER_SIZE:
            raise ValueError("Invalid file: header too short")

        magic = header[:8]
        if magic != self.MAGIC:
            raise ValueError(f"Invalid file: wrong magic {magic}")
        version = struct.unpack_from("<B", header, 8)[0]
        if version != self.VERSION:
            raise ValueError(f"Unsupported version: {version}")

        self.bucket_count = struct.unpack_from("<Q", header, 9)[0]
        self.bucket_size = struct.unpack_from("<I", header, 17)[0]
        self.fingerprint_bits = struct.unpack_from("<I", header, 21)[0]
        self.max_kicks = struct.unpack_from("<I", header, 25)[0]
        self.inserted_count = struct.unpack_from("<Q", header, 29)[0]
        self.seed = struct.unpack_from("<I", header, 37)[0]

        # Recalculate derived parameters
        self.bucket_mask = self.bucket_count - 1
        self.bytes_per_slot = math.ceil(self.fingerprint_bits / 8)
        self.fingerprint_mask = (1 << self.fingerprint_bits) - 1
        self.payload_bytes = self.bucket_count * self.bucket_size * self.bytes_per_slot

        self._open_mmap()

    def _open_mmap(self):
        """Open memory-mapped file access."""
        self._file = open(self.filepath, "r+b")
        self._mmap = mmap.mmap(self._file.fileno(), 0)
        self._memory_view = memoryview(self._mmap)
        self._header_view = self._memory_view[: self.HEADER_SIZE]
        self._payload_view = self._memory_view[
            self.HEADER_SIZE : self.HEADER_SIZE + self.payload_bytes
        ]

    def _fingerprint(self, key: Union[str, bytes]) -> int:
        """
        Generate f-bit fingerprint from key.

        Args:
            key: The key to fingerprint

        Returns:
            Non-zero fingerprint (f bits)
        """
        if isinstance(key, str):
            key = key.encode("utf-8")
        fp = mmh3.hash(key, seed=self.seed + 1) & self.fingerprint_mask
        if fp == self.FLAG_EMPTY:  # Avoid zero fingerprint (reserved for empty slots)
            fp = 1

        return fp

    def _bucket_index_from_hash(self, h: int) -> int:
        """Convert hash to bucket index."""
        return h & self.bucket_mask

    def _alt_index(self, index: int, fp: int) -> int:
        """Compute alternate bucket index: index XOR hash(fp)."""
        fp_hash = mmh3.hash(
            fp.to_bytes(self.bytes_per_slot, "little"), seed=self.seed + 2
        )
        return index ^ self._bucket_index_from_hash(fp_hash)

    def _two_hashes(self, key: Union[str, bytes]) -> Tuple[int, int]:
        """Generate primary hash and fingerprint for key."""
        if isinstance(key, str):
            key = key.encode("utf-8")

        # Primary hash for first bucket
        h1 = mmh3.hash(key, seed=self.seed)
        bucket1 = self._bucket_index_from_hash(h1)

        # Fingerprint and alternate bucket
        fp = self._fingerprint(key)
        bucket2 = self._alt_index(bucket1, fp)

        return bucket1, bucket2

    def _read_slot(self, bucket: int, slot: int) -> int:
        """Read fingerprint from bucket[slot]."""
        if self._closed:
            raise ValueError("Cannot operate on closed CuckooFilter")

        base_offset = (bucket * self.bucket_size + slot) * self.bytes_per_slot

        if self.bytes_per_slot == 1:
            return self._payload_view[base_offset]
        elif self.bytes_per_slot == 2:
            return struct.unpack_from("<H", self._payload_view, base_offset)[0]
        elif self.bytes_per_slot == 4:
            return struct.unpack_from("<I", self._payload_view, base_offset)[0]
        else:
            # Variable byte length
            fp_bytes = self._payload_view[
                base_offset : base_offset + self.bytes_per_slot
            ]
            return int.from_bytes(fp_bytes, "little")

    def _write_slot(self, bucket: int, slot: int, fp: int):
        """Write fingerprint to bucket[slot]."""
        if self._closed:
            raise ValueError("Cannot operate on closed CuckooFilter")

        base_offset = (bucket * self.bucket_size + slot) * self.bytes_per_slot

        if self.bytes_per_slot == 1:
            self._payload_view[base_offset] = fp & 0xFF
        elif self.bytes_per_slot == 2:
            struct.pack_into("<H", self._payload_view, base_offset, fp & 0xFFFF)
        elif self.bytes_per_slot == 4:
            struct.pack_into("<I", self._payload_view, base_offset, fp & 0xFFFFFFFF)
        else:
            # Variable byte length
            fp_bytes = fp.to_bytes(self.bytes_per_slot, "little")
            self._payload_view[base_offset : base_offset + self.bytes_per_slot] = (
                fp_bytes
            )

    def _is_slot_empty(self, fp: int) -> bool:
        """Check if slot is empty (fingerprint is 0)."""
        return fp == self.FLAG_EMPTY

    def _find_free_slot_in_bucket(self, bucket: int) -> Optional[int]:
        """Find first empty slot in bucket, return slot index or None."""
        for slot in range(self.bucket_size):
            if self._is_slot_empty(self._read_slot(bucket, slot)):
                return slot
        return None

    def _try_insert_in_bucket(self, bucket: int, fp: int) -> bool:
        """Try to insert fingerprint in bucket, return success."""
        slot = self._find_free_slot_in_bucket(bucket)
        if slot is not None:
            self._write_slot(bucket, slot, fp)
            return True
        return False

    def _evict_and_relocate(self, start_bucket: int, fp: int) -> bool:
        """
        Perform cuckoo eviction starting from start_bucket with fingerprint fp.

        Returns True if successfully placed, False if failed after max_kicks.
        """
        current_bucket = start_bucket
        current_fp = fp

        for kick in range(self.max_kicks):
            self.total_kicks += 1

            # Pick random slot to evict
            evict_slot = random.randrange(self.bucket_size)

            # Read evicted fingerprint and write new one
            evicted_fp = self._read_slot(current_bucket, evict_slot)
            self._write_slot(current_bucket, evict_slot, current_fp)

            # If we evicted an empty slot, we're done
            if self._is_slot_empty(evicted_fp):
                return True

            # Try to place evicted fingerprint in its alternate bucket
            current_fp = evicted_fp
            current_bucket = self._alt_index(current_bucket, current_fp)

            # Check if alternate bucket has free space
            if self._try_insert_in_bucket(current_bucket, current_fp):
                return True

        return False

    def add(self, key: Union[str, bytes]) -> bool:
        """
        Add an element to the Cuckoo filter.

        Args:
            key: The element to add (string or bytes)

        Returns:
            True if successfully added, False if failed (filter might be full)
        """
        if self._closed:
            raise ValueError("Cannot operate on closed CuckooFilter")

        fp = self._fingerprint(key)
        bucket1, bucket2 = self._two_hashes(key)

        # Try to insert in either bucket
        if self._try_insert_in_bucket(bucket1, fp):
            self.inserted_count += 1
            self._update_count_in_header()
            return True

        if self._try_insert_in_bucket(bucket2, fp):
            self.inserted_count += 1
            self._update_count_in_header()
            return True

        # Both buckets full, try eviction
        start_bucket = random.choice([bucket1, bucket2])
        if self._evict_and_relocate(start_bucket, fp):
            self.inserted_count += 1
            self._update_count_in_header()
            return True

        # Failed to insert
        self.failed_inserts += 1
        return False

    def contains(self, key: Union[str, bytes]) -> bool:
        """
        Test if an element might be in the set.

        Args:
            key: The element to test (string or bytes)

        Returns:
            False if definitely not in set, True if probably in set
        """
        if self._closed:
            raise ValueError("Cannot operate on closed CuckooFilter")

        fp = self._fingerprint(key)
        bucket1, bucket2 = self._two_hashes(key)

        # Check both candidate buckets
        for bucket in [bucket1, bucket2]:
            for slot in range(self.bucket_size):
                if self._read_slot(bucket, slot) == fp:
                    return True

        return False

    def delete(self, key: Union[str, bytes]) -> bool:
        """
        Delete an element from the filter.

        Args:
            key: The element to delete (string or bytes)

        Returns:
            True if found and deleted, False if not found
        """
        if self._closed:
            raise ValueError("Cannot operate on closed CuckooFilter")

        fp = self._fingerprint(key)
        bucket1, bucket2 = self._two_hashes(key)

        # Search both candidate buckets
        for bucket in [bucket1, bucket2]:
            for slot in range(self.bucket_size):
                if self._read_slot(bucket, slot) == fp:
                    self._write_slot(bucket, slot, self.FLAG_EMPTY)  # Mark as empty
                    self.inserted_count = max(0, self.inserted_count - 1)
                    self._update_count_in_header()
                    return True

        return False

    def __contains__(self, key: Union[str, bytes]) -> bool:
        """Support 'in' operator."""
        return self.contains(key)

    def _update_count_in_header(self):
        """Update inserted count in file header."""
        struct.pack_into("<Q", self._header_view, 29, self.inserted_count)

    def clear(self):
        """Clear all elements from the filter."""
        if self._closed:
            raise ValueError("Cannot operate on closed CuckooFilter")

        # Zero out payload
        for i in range(self.payload_bytes):
            self._payload_view[i] = self.FLAG_EMPTY

        self.inserted_count = 0
        self.total_kicks = 0
        self.failed_inserts = 0
        self._update_count_in_header()

    def flush(self):
        """Flush changes to disk."""
        if not self._closed:
            self._mmap.flush()
            os.fsync(self._file.fileno())

    def close(self):
        """Close the filter and clean up resources."""
        if self._closed:
            return

        self._closed = True

        # Clean up views first
        if hasattr(self, "_payload_view"):
            del self._payload_view
        if hasattr(self, "_header_view"):
            del self._header_view
        if hasattr(self, "_memory_view"):
            del self._memory_view

        if hasattr(self, "_mmap") and self._mmap:
            try:
                self._mmap.close()
                self._mmap = None
            except (ValueError, BufferError):
                pass

        if hasattr(self, "_file") and self._file:
            try:
                self._file.close()
                self._file = None
            except (ValueError, OSError):
                pass

        # Clean up temporary file if we own it
        if (
            hasattr(self, "_owns_file")
            and self._owns_file
            and hasattr(self, "_temp_file")
            and self._temp_file
        ):
            try:
                os.unlink(self._temp_file.name)
            except (OSError, FileNotFoundError):
                pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor."""
        try:
            self.close()
        except Exception:
            pass

    def get_stats(self) -> dict:
        """
        Get comprehensive statistics about the filter.

        Returns:
            Dictionary with filter statistics
        """
        # Calculate bucket occupancy distribution
        occupancy_hist = [0] * (self.bucket_size + 1)
        used_slots = 0

        for bucket in range(self.bucket_count):
            bucket_count = 0
            for slot in range(self.bucket_size):
                if not self._is_slot_empty(self._read_slot(bucket, slot)):
                    bucket_count += 1
                    used_slots += 1
            occupancy_hist[bucket_count] += 1

        load_factor = used_slots / (self.bucket_count * self.bucket_size)

        # Calculate current false positive rate estimate
        if used_slots > 0:
            estimated_fpr = min(
                1.0, (2 * self.bucket_size) / (2**self.fingerprint_bits)
            )
        else:
            estimated_fpr = 0.0

        return {
            "capacity": self.capacity,
            "inserted_count": self.inserted_count,
            "false_positive_rate": self.false_positive_rate,
            "actual_false_positive_rate": self.actual_false_positive_rate,
            "estimated_current_fpr": estimated_fpr,
            "bucket_count": self.bucket_count,
            "bucket_size": self.bucket_size,
            "fingerprint_bits": self.fingerprint_bits,
            "total_slots": self.bucket_count * self.bucket_size,
            "used_slots": used_slots,
            "load_factor": load_factor,
            "occupancy_histogram": occupancy_hist,
            "total_kicks": self.total_kicks,
            "failed_inserts": self.failed_inserts,
            "avg_kicks_per_insert": self.total_kicks / max(1, self.inserted_count),
            "memory_usage_mb": self.total_bytes / (1024 * 1024),
            "filepath": self.filepath,
        }

    @classmethod
    def calculate_parameters(
        cls,
        capacity: int,
        false_positive_rate: float,
        bucket_size: int = DEFAULT_BUCKET_SIZE,
    ) -> dict:
        """
        Calculate optimal parameters for given capacity and false positive rate.

        Args:
            capacity: Expected number of elements
            false_positive_rate: Desired false positive rate
            bucket_size: Number of slots per bucket

        Returns:
            Dictionary with calculated parameters
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if not (0 < false_positive_rate < 1):
            raise ValueError("False positive rate must be between 0 and 1")
        if bucket_size not in [1, 2, 4, 8]:
            raise ValueError("Bucket size must be 1, 2, 4, or 8")

        # Get target load factor
        target_load = cls.DEFAULT_LOAD_FACTORS.get(bucket_size, 0.90)

        # Calculate bucket count
        bucket_count = math.ceil(capacity / (target_load * bucket_size))
        bucket_count = 2 ** math.ceil(math.log2(bucket_count))  # Round to power of 2

        # Calculate fingerprint bits
        fingerprint_bits = max(
            8, math.ceil(math.log2((2 * bucket_size) / false_positive_rate))
        )

        # Storage calculations
        bytes_per_slot = math.ceil(fingerprint_bits / 8)
        payload_bytes = bucket_count * bucket_size * bytes_per_slot
        total_bytes = cls.HEADER_SIZE + payload_bytes

        # Actual false positive rate
        actual_fpr = (2 * bucket_size) / (2**fingerprint_bits)

        return {
            "capacity": capacity,
            "false_positive_rate": false_positive_rate,
            "actual_false_positive_rate": actual_fpr,
            "bucket_count": bucket_count,
            "bucket_size": bucket_size,
            "fingerprint_bits": fingerprint_bits,
            "bytes_per_slot": bytes_per_slot,
            "total_slots": bucket_count * bucket_size,
            "target_load_factor": target_load,
            "payload_bytes": payload_bytes,
            "total_bytes": total_bytes,
            "memory_usage_mb": total_bytes / (1024 * 1024),
            "memory_usage_gb": total_bytes / (1024 * 1024 * 1024),
        }


if __name__ == "__main__":
    # Example: Create a filter for 1000 elements with 1% FPR
    print("Cuckoo Filter Example")
    print("=" * 50)

    # Calculate parameters first
    params = CuckooFilter.calculate_parameters(1000, 0.01, bucket_size=4)
    print("Calculated Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()

    # Create and test filter
    cf = CuckooFilter(capacity=1000, false_positive_rate=0.01, seed=42)
    try:
        # Add some elements
        test_elements = [f"user_{i}" for i in range(500)]
        success_count = 0

        for element in test_elements:
            if cf.add(element):
                success_count += 1

        print(f"Successfully added: {success_count}/{len(test_elements)} elements")

        # Test membership
        print(f"\n'user_42' in filter: {'user_42' in cf}")
        print(f"'user_999999' in filter: {'user_999999' in cf}")

        # Test deletion
        deleted = cf.delete("user_42")
        print(f"\nDeleted 'user_42': {deleted}")
        print(f"'user_42' in filter after deletion: {'user_42' in cf}")

        # Print statistics
        stats = cf.get_stats()
        print("\nFilter Statistics:")
        print(f"  Inserted: {stats['inserted_count']:,}")
        print(f"  Load factor: {stats['load_factor']:.4f}")
        print(f"  Memory: {stats['memory_usage_mb']:.2f} MB")
        print(f"  Total kicks: {stats['total_kicks']:,}")
        print(f"  Failed inserts: {stats['failed_inserts']:,}")
        print(f"  Avg kicks per insert: {stats['avg_kicks_per_insert']:.2f}")
        print(f"  Estimated FPR: {stats['estimated_current_fpr']:.6f}")
        print(f"  Occupancy histogram: {stats['occupancy_histogram']}")

    finally:
        cf.close()
