import heapq
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

from data.reader import UsernameReader
from data.writer import UsernameWriter


class SortedList:
    """Sorts usernames from reader and provides sorted access."""

    def __init__(
        self, reader: UsernameReader, sorted_filepath: Optional[Union[str, Path]] = None
    ):
        self.reader = reader
        self.sorted_filepath = Path(sorted_filepath) if sorted_filepath else None
        self._is_sorted = False
        self._sorted_reader: Optional[UsernameReader] = None

    def sort(self, chunk_size: int = 1_000_000, temp_dir: Optional[str] = None) -> None:
        """Sort usernames using external merge sort."""
        print(f"Starting sort of {len(self.reader):,} usernames...")
        start_time = time.time()

        if self.sorted_filepath is None:
            self.sorted_filepath = self.reader.filepath.with_name(
                self.reader.filepath.stem + "_sorted.dat"
            )

        # Create temporary directory for chunks
        with tempfile.TemporaryDirectory(dir=temp_dir) as temp_dir_path:
            temp_path = Path(temp_dir_path)

            # Phase 1: Create sorted chunks
            chunk_files = self._create_sorted_chunks(chunk_size, temp_path)

            # Phase 2: Merge chunks
            self._merge_chunks(chunk_files)

        # Create reader for sorted data
        self._sorted_reader = UsernameReader(self.sorted_filepath)
        self._is_sorted = True

        elapsed = time.time() - start_time
        print(f"Sort complete in {elapsed:.1f}s")

    def _create_sorted_chunks(self, chunk_size: int, temp_path: Path) -> list[Path]:
        """Create sorted chunks from the original data."""
        chunk_files: list[Path] = []
        total_usernames = len(self.reader)

        for chunk_start in range(0, total_usernames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_usernames)
            print(
                f"Creating chunk {len(chunk_files) + 1}: {chunk_start:,}-{chunk_end:,}"
            )

            # Load chunk into memory and sort
            chunk = [self.reader[i] for i in range(chunk_start, chunk_end)]
            chunk.sort()

            # Write sorted chunk
            chunk_file = temp_path / f"chunk_{len(chunk_files):04d}.dat"
            writer = UsernameWriter(chunk_file)
            writer.write_usernames(iter(chunk))
            chunk_files.append(chunk_file)

        return chunk_files

    def _merge_chunks(self, chunk_files: list[Path]) -> None:
        """Merge sorted chunks into final sorted file."""
        print(f"Merging {len(chunk_files)} chunks...")

        # Open readers for all chunks
        readers = [UsernameReader(chunk_file) for chunk_file in chunk_files]

        # Initialize heap with first username from each chunk
        heap: list[tuple[str, int, int]] = []
        for i, reader in enumerate(readers):
            if len(reader) > 0:
                heapq.heappush(
                    heap, (reader[0], i, 0)
                )  # (username, reader_idx, position)

        # Merge using heap
        assert self.sorted_filepath is not None
        writer = UsernameWriter(self.sorted_filepath)

        def merged_usernames():
            while heap:
                username, reader_idx, pos = heapq.heappop(heap)
                yield username

                # Add next username from same reader if available
                if pos + 1 < len(readers[reader_idx]):
                    next_username = readers[reader_idx][pos + 1]
                    heapq.heappush(heap, (next_username, reader_idx, pos + 1))

        writer.write_usernames(merged_usernames())

        # Close all readers
        for reader in readers:
            reader.close()

    def __getitem__(self, index: int) -> str:
        """Get username at index from sorted data."""
        if not self._is_sorted or self._sorted_reader is None:
            raise RuntimeError("Must call sort() before accessing sorted data")
        return self._sorted_reader[index]

    def __len__(self) -> int:
        """Get length of sorted data."""
        if not self._is_sorted or self._sorted_reader is None:
            raise RuntimeError("Must call sort() before accessing sorted data")
        return len(self._sorted_reader)

    def close(self) -> None:
        """Close sorted reader."""
        if self._sorted_reader:
            self._sorted_reader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
