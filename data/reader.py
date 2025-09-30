import mmap
import struct
from pathlib import Path
from typing import Iterator, Optional, Union


class UsernameReader:
    """Memory-efficient reader for binary username data with cumulative position index."""

    def __init__(self, filepath: Union[str, Path], limit: Optional[int] = None):
        self.filepath = Path(filepath)
        self.index_filepath = self.filepath.with_suffix(".idx")

        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {self.filepath}")
        if not self.index_filepath.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_filepath}")

        # Load the entire index file into memory for O(1) random access
        self._load_index()

        # Set the limit for the number of usernames to consider
        self._limit = limit

        # Open data file with memory mapping for efficient access
        self._data_file = open(self.filepath, "rb")
        data_file_size = self.filepath.stat().st_size
        if data_file_size > 0:
            self._mmapped_data = mmap.mmap(
                self._data_file.fileno(), 0, access=mmap.ACCESS_READ
            )
            self.eof_idx = len(self._mmapped_data)
        else:
            self._mmapped_data = None
            self.eof_idx = 0

    def _load_index(self) -> None:
        """Load the cumulative position index into memory efficiently, avoiding crashes for large files."""
        index_size = self.index_filepath.stat().st_size
        self.total_count = (
            index_size // 8
        )  # Each entry is 8 bytes (unsigned 64-bit integer)

        # Use memory mapping for the index file to avoid loading all into RAM
        self._index_file = open(self.index_filepath, "rb")
        if index_size > 0:
            self._mmapped_index = mmap.mmap(
                self._index_file.fileno(), 0, access=mmap.ACCESS_READ
            )
        else:
            self._mmapped_index = None

        class PositionsProxy:
            def __init__(self, mmapped_index, count):
                self.mmapped_index = mmapped_index
                self.count = count

            def __getitem__(self, idx):
                if not 0 <= idx < self.count:
                    raise IndexError(f"Index {idx} out of range [0, {self.count})")
                if self.mmapped_index is None:
                    raise IndexError("No data available in empty file")
                offset = idx * 8
                return struct.unpack_from("<Q", self.mmapped_index, offset)[0]

            def __len__(self):
                return self.count

        self.positions = PositionsProxy(self._mmapped_index, self.total_count)

    def __len__(self) -> int:
        """Return total number of usernames (limited by configured limit)."""
        if self._limit is not None:
            return min(self._limit, self.total_count)
        return self.total_count

    @property
    def effective_count(self) -> int:
        """Return the effective count considering the limit."""
        if self._limit is not None:
            return min(self._limit, self.total_count)
        return self.total_count

    def set_limit(self, limit: Optional[int]) -> None:
        """Set a new limit for the number of usernames to consider."""
        if limit is not None and limit < 0:
            raise ValueError("Limit must be non-negative or None")
        self._limit = limit

    def get_limit(self) -> Optional[int]:
        """Get the current limit."""
        return self._limit

    def clear_limit(self) -> None:
        """Remove the limit to access all usernames."""
        self._limit = None

    def __getitem__(self, index: int) -> str:
        """Get username at specific index using O(1) random access."""
        effective_count = self.effective_count

        # Handle negative indices
        if index < 0:
            index = effective_count + index

        if not 0 <= index < effective_count:
            raise IndexError(f"Index {index} out of range [0, {effective_count})")

        # Handle empty file case
        if self._mmapped_data is None:
            raise IndexError("Cannot access data in empty file")

        # Get start and end positions
        start_pos = self.positions[index]
        if index + 1 < self.total_count:
            end_pos = self.positions[index + 1]
        else:
            # Last username: read until end of data file
            end_pos = self.eof_idx

        # Calculate length and read username
        length = end_pos - start_pos
        username_bytes = self._mmapped_data[start_pos : start_pos + length]
        return username_bytes.decode("utf-8")

    def __iter__(self) -> Iterator[str]:
        """Iterate over all usernames efficiently."""
        return self.iter_usernames()

    def iter_usernames(
        self, start: int = 0, end: Optional[int] = None
    ) -> Iterator[str]:
        """Iterate over usernames in given range using direct position access."""
        effective_count = self.effective_count

        if end is None:
            end = effective_count

        if not 0 <= start < effective_count:
            raise ValueError(f"Start index {start} out of range")
        if not start <= end <= effective_count:
            raise ValueError(f"End index {end} out of range")

        for i in range(start, end):
            yield self[i]

    def iter_batch(
        self, batch_size: int = 10000, start: int = 0, end: Optional[int] = None
    ) -> Iterator[list[str]]:
        """Iterate over usernames in batches."""
        effective_count = self.effective_count

        if end is None:
            end = effective_count

        current = start
        while current < end:
            batch_end = min(current + batch_size, end)
            batch = [self[i] for i in range(current, batch_end)]
            yield batch
            current = batch_end

    def search(
        self, pattern: str, case_sensitive: bool = False
    ) -> Iterator[tuple[int, str]]:
        """Search for usernames matching a pattern."""
        if not case_sensitive:
            pattern = pattern.lower()

        effective_count = self.effective_count
        for i in range(effective_count):
            username = self[i]
            check_username = username if case_sensitive else username.lower()
            if pattern in check_username:
                yield i, username

    def get_stats(self) -> dict:
        """Get statistics about the username dataset."""
        data_size = self.filepath.stat().st_size
        index_size = self.index_filepath.stat().st_size
        effective_count = self.effective_count

        return {
            "total_usernames": self.total_count,
            "effective_usernames": effective_count,
            "limit": self._limit,
            "data_file_size": data_size,
            "index_file_size": index_size,
            "total_size": data_size + index_size,
            "avg_username_length": data_size / self.total_count
            if self.total_count > 0
            else 0,
            "data_file_size_gb": data_size / (1024**3),
            "index_file_size_mb": index_size / (1024**2),
            "total_size_gb": (data_size + index_size) / (1024**3),
            "index_entry_size": 8,  # 64-bit integers
            "index_entries": self.total_count,
        }

    def sample(self, count: int = 10) -> list[str]:
        """Get a random sample of usernames using O(1) access."""
        import random

        effective_count = self.effective_count
        if count > effective_count:
            count = effective_count

        indices = random.sample(range(effective_count), count)
        return [self[i] for i in indices]

    def close(self) -> None:
        """Close the memory-mapped file and data file."""
        if hasattr(self, "_mmapped_data") and self._mmapped_data is not None:
            self._mmapped_data.close()
        if hasattr(self, "_mmapped_index") and self._mmapped_index is not None:
            self._mmapped_index.close()
        if hasattr(self, "_index_file"):
            self._index_file.close()
        if hasattr(self, "_data_file"):
            self._data_file.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self.close()
