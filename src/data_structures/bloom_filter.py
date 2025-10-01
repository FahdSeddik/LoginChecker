"""
Mathematical Foundation:
- False positive rate: eps = (1 - e^(-k*n/m))^k
- Optimal k: k_opt = (m/n) * ln(2)
- Required bits: m = -(n * ln(eps)) / (ln(2))^2
- Bits per element: bpe = -ln(eps) / (ln(2))^2
"""

import hashlib
import math
import mmap
import os
import struct
import tempfile
from typing import Optional, Tuple, Union


class BloomFilter:
    """
    Bloom filter implementation with memory-mapped storage.
    Uses memory-mapped files for persistence and the Kirsch-Mitzenmacher
    two-hash optimization for efficiency.
    """

    def __init__(
        self,
        capacity: int,
        false_positive_rate: float,
        filepath: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize a Bloom filter with given capacity and false positive rate.

        Args:
            capacity: Expected number of elements (n)
            false_positive_rate: Desired false positive rate (eps)
            filepath: Optional file path for persistence (creates temp file if None)
            seed: Random seed for hash functions (generates random if None)
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if not (0 < false_positive_rate < 1):
            raise ValueError("False positive rate must be between 0 and 1")

        self.capacity = capacity
        self.false_positive_rate = false_positive_rate
        if isinstance(seed, int):
            self.seed = seed.to_bytes(8, "little", signed=False)
        else:
            self.seed = os.urandom(8)

        self._calculate_parameters()
        self._initialize_storage(filepath)
        self._closed = False
        self.inserted_count = 0

    def _calculate_parameters(self):
        """Calculate optimal m (bits) and k (hash functions) parameters."""
        ln2 = math.log(2)
        ln2_squared = ln2 * ln2

        # Required number of bits: m = -(n * ln(eps)) / (ln(2))^2
        self.num_bits = int(
            math.ceil(
                -(self.capacity * math.log(self.false_positive_rate)) / ln2_squared
            )
        )
        self.num_bytes = math.ceil(self.num_bits / 8)

        # Optimal number of hash functions: k = (m/n) * ln(2)
        self.num_hashes = int(round((self.num_bits / self.capacity) * ln2))
        self.num_hashes = max(1, self.num_hashes)

        self.actual_false_positive_rate = math.pow(
            1 - math.exp(-self.num_hashes * self.capacity / self.num_bits),
            self.num_hashes,
        )
        self.bits_per_element = self.num_bits / self.capacity

    def _initialize_storage(self, filepath: Optional[str]):
        """Initialize memory-mapped storage for the bit array."""
        if filepath is None:
            # Create temporary file
            self._temp_file = tempfile.NamedTemporaryFile(delete=False)
            filepath = self._temp_file.name
            self._owns_file = True
        else:
            self._temp_file = None
            self._owns_file = False

        self.filepath = filepath

        # Create or open the file with correct size
        with open(filepath, "w+b") as f:
            # Ensure file is large enough
            f.seek(self.num_bytes - 1)
            f.write(b"\x00")
            f.flush()

        # Memory map the file
        self._file = open(filepath, "r+b")
        self._mmap = mmap.mmap(self._file.fileno(), self.num_bytes)

        # Create memoryview for efficient access
        self._memory_view = memoryview(self._mmap)

    def _two_hashes(self, key: Union[str, bytes]) -> Tuple[int, int]:
        """
        Generate two independent hash values using BLAKE2b.
        Args:
            key: The key to hash (string or bytes)

        Returns:
            Tuple of two 64-bit hash values (h1, h2)
        """
        # Convert string to bytes if needed
        if isinstance(key, str):
            key = key.encode("utf-8")
        hasher = hashlib.blake2b(key=self.seed, digest_size=16)
        hasher.update(key)
        digest = hasher.digest()

        h1, h2 = struct.unpack_from("<QQ", digest)

        # avoid the degenerate case h2 == 0 -> use h2 |= 1 (cheap & effective)
        h2 |= 1

        return h1, h2

    def _get_bit_positions(self, key: Union[str, bytes]) -> list[int]:
        """
        Generate k bit positions using Kirsch-Mitzenmacher optimization.
        References:
            1. https://www.eecs.harvard.edu/~michaelm/postscripts/tr-02-05.pdf
            2. https://stackoverflow.com/questions/11954086/which-hash-functions-to-use-in-a-bloom-filter

        Instead of computing k independent hashes, we compute just 2 hashes
        and derive k positions as: pos_i = (h1 + i * h2) % m

        Args:
            key: The key to hash

        Returns:
            List of k bit positions
        """
        h1, h2 = self._two_hashes(key)

        positions = []
        for i in range(self.num_hashes):
            # Kirsch-Mitzenmacher: g_i(x) = (h1 + i * h2) mod m
            position = (h1 + i * h2) % self.num_bits
            positions.append(position)

        return positions

    def _set_bit(self, bit_index: int):
        """Set a bit at the given index to 1."""
        if self._closed:
            raise ValueError("Cannot operate on closed BloomFilter")

        byte_index = bit_index // 8
        bit_in_byte = bit_index % 8
        self._memory_view[byte_index] |= 1 << bit_in_byte

    def _get_bit(self, bit_index: int) -> bool:
        """Get the value of a bit at the given index."""
        if self._closed:
            raise ValueError("Cannot operate on closed BloomFilter")

        byte_index = bit_index // 8
        bit_in_byte = bit_index % 8

        # Test the bit using bitwise AND
        return bool((self._memory_view[byte_index] >> bit_in_byte) & 1)

    def add(self, key: Union[str, bytes]):
        """
        Add an element to the Bloom filter.

        Args:
            key: The element to add (string or bytes)
        """
        positions = self._get_bit_positions(key)

        # Set all k bits for this key
        for position in positions:
            self._set_bit(position)

        self.inserted_count += 1

    def contains(self, key: Union[str, bytes]) -> bool:
        """
        Test if an element might be in the set.

        Args:
            key: The element to test (string or bytes)

        Returns:
            False if definitely not in set, True if probably in set
        """
        positions = self._get_bit_positions(key)

        for position in positions:
            if not self._get_bit(position):
                return False

        return True

    def __contains__(self, key: Union[str, bytes]) -> bool:
        """Support 'in' operator."""
        return self.contains(key)

    def get_stats(self) -> dict:
        """
        Get comprehensive statistics about the filter.

        Returns:
            Dictionary with filter statistics
        """
        set_bits = sum(bin(byte).count("1") for byte in self._memory_view)
        fill_ratio = set_bits / self.num_bits
        if self.inserted_count > 0:
            estimated_fpr = math.pow(fill_ratio, self.num_hashes)
        else:
            estimated_fpr = 0.0

        return {
            "capacity": self.capacity,
            "inserted_count": self.inserted_count,
            "false_positive_rate": self.false_positive_rate,
            "actual_false_positive_rate": self.actual_false_positive_rate,
            "estimated_current_fpr": estimated_fpr,
            "num_bits": self.num_bits,
            "num_bytes": self.num_bytes,
            "num_hashes": self.num_hashes,
            "bits_per_element": self.bits_per_element,
            "set_bits": set_bits,
            "fill_ratio": fill_ratio,
            "memory_usage_mb": self.num_bytes / (1024 * 1024),
            "filepath": self.filepath,
        }

    def clear(self):
        """Clear all elements from the filter."""
        # Zero out all bytes
        for i in range(self.num_bytes):
            self._memory_view[i] = 0

            self.inserted_count = 0

    def flush(self):
        """Flush changes to disk."""
        self._mmap.flush()
        os.fsync(self._file.fileno())

    def close(self):
        """Close the filter and clean up resources."""
        if self._closed:
            return

        self._closed = True

        # Clean up memoryview first to release references
        if hasattr(self, "_memory_view"):
            del self._memory_view

        if hasattr(self, "_mmap") and self._mmap:
            try:
                self._mmap.close()
                self._mmap = None
            except (ValueError, BufferError):
                pass  # Already closed or has exported pointers

        if hasattr(self, "_file") and self._file:
            try:
                self._file.close()
                self._file = None
            except (ValueError, OSError):
                pass  # Already closed

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
            pass  # Ignore all exceptions during cleanup

    @classmethod
    def calculate_parameters(cls, capacity: int, false_positive_rate: float) -> dict:
        """
        Calculate optimal parameters for given capacity and false positive rate.

        Static method for parameter estimation without creating a filter.

        Args:
            capacity: Expected number of elements
            false_positive_rate: Desired false positive rate

        Returns:
            Dictionary with calculated parameters
        """
        ln2 = math.log(2)
        ln2_squared = ln2 * ln2

        # Calculate parameters
        num_bits = int(
            math.ceil(-(capacity * math.log(false_positive_rate)) / ln2_squared)
        )
        num_hashes = max(1, int(round((num_bits / capacity) * ln2)))
        num_bytes = math.ceil(num_bits / 8)

        # Calculate actual false positive rate
        actual_fpr = math.pow(
            1 - math.exp(-num_hashes * capacity / num_bits), num_hashes
        )

        # Calculate storage requirements
        bits_per_element = num_bits / capacity
        memory_mb = num_bytes / (1024 * 1024)
        memory_gb = memory_mb / 1024

        return {
            "capacity": capacity,
            "false_positive_rate": false_positive_rate,
            "actual_false_positive_rate": actual_fpr,
            "num_bits": num_bits,
            "num_bytes": num_bytes,
            "num_hashes": num_hashes,
            "bits_per_element": bits_per_element,
            "memory_usage_mb": memory_mb,
            "memory_usage_gb": memory_gb,
        }


if __name__ == "__main__":
    # Example: Create a filter for 1 billion elements with 10% FPR
    bf = BloomFilter(capacity=1_000, false_positive_rate=0.1, seed=42)
    try:
        # Add some elements
        test_elements = [f"user_{i}" for i in range(1000)]

        for element in test_elements:
            bf.add(element)

        # Test membership
        print(f"\n'user_42' in filter: {'user_42' in bf}")
        print(f"'user_999999' in filter: {'user_999999' in bf}")

        # Print statistics
        stats = bf.get_stats()
        print("\nFilter Statistics:")
        print(f"  Inserted: {stats['inserted_count']:,}")
        print(f"  Memory: {stats['memory_usage_mb']:.2f} MB")
        print(f"  Fill ratio: {stats['fill_ratio']:.4f}")
        print(f"  Estimated FPR: {stats['estimated_current_fpr']:.6f}")
    finally:
        bf.close()
