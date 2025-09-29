import mmap
import struct
from pathlib import Path
from typing import Iterator, Optional, Union


class UsernameReader:
    """Memory-efficient reader for binary username data with length-only index."""

    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self.index_filepath = self.filepath.with_suffix(".idx")

        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {self.filepath}")
        if not self.index_filepath.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_filepath}")

        # Calculate total count from index file size
        index_size = self.index_filepath.stat().st_size
        self.total_count = index_size  # Each entry is 1 byte (length only)

    def __len__(self) -> int:
        """Return total number of usernames."""
        return self.total_count

    def __getitem__(self, index: int) -> str:
        """Get username at specific index by reading sequentially from start."""
        # Handle negative indices
        if index < 0:
            index = self.total_count + index

        if not 0 <= index < self.total_count:
            raise IndexError(f"Index {index} out of range [0, {self.total_count})")

        # Calculate position by reading and summing lengths one by one
        position = 0
        target_length = 0

        with open(self.index_filepath, "rb") as index_file:
            for i in range(index + 1):
                length_bytes = index_file.read(1)
                if not length_bytes:
                    raise IndexError(f"Index {index} out of range")
                length = struct.unpack("<B", length_bytes)[0]

                if i == index:
                    target_length = length
                else:
                    position += length

        # Read the username from data file
        with open(self.filepath, "rb") as data_file:
            data_file.seek(position)
            username_bytes = data_file.read(target_length)
            return username_bytes.decode("utf-8")

    def __iter__(self) -> Iterator[str]:
        """Iterate over all usernames efficiently."""
        return self.iter_usernames()

    def iter_usernames(
        self, start: int = 0, end: Optional[int] = None
    ) -> Iterator[str]:
        """Iterate over usernames in given range using sequential reading."""
        if end is None:
            end = self.total_count

        if not 0 <= start < self.total_count:
            raise ValueError(f"Start index {start} out of range")
        if not start <= end <= self.total_count:
            raise ValueError(f"End index {end} out of range")

        with (
            open(self.index_filepath, "rb") as index_file,
            open(self.filepath, "rb") as data_file,
        ):
            # Calculate starting position by reading lengths one by one
            position = 0
            if start > 0:
                for _ in range(start):
                    length_bytes = index_file.read(1)
                    if not length_bytes:
                        raise ValueError(f"Start index {start} out of range")
                    length = struct.unpack("<B", length_bytes)[0]
                    position += length

            # Memory map the data file for efficient access
            with mmap.mmap(
                data_file.fileno(), 0, access=mmap.ACCESS_READ
            ) as mmapped_data:
                # Read usernames in the range
                for _ in range(start, end):
                    length_bytes = index_file.read(1)
                    if not length_bytes:
                        break
                    length = struct.unpack("<B", length_bytes)[0]

                    username_bytes = mmapped_data[position : position + length]
                    yield username_bytes.decode("utf-8")
                    position += length

    def iter_batch(
        self, batch_size: int = 10000, start: int = 0, end: Optional[int] = None
    ) -> Iterator[list[str]]:
        """Iterate over usernames in batches."""
        if end is None:
            end = self.total_count

        current = start
        while current < end:
            batch_end = min(current + batch_size, end)
            batch = list(self.iter_usernames(current, batch_end))
            yield batch
            current = batch_end

    def search(
        self, pattern: str, case_sensitive: bool = False
    ) -> Iterator[tuple[int, str]]:
        """Search for usernames matching a pattern."""
        if not case_sensitive:
            pattern = pattern.lower()

        for i, username in enumerate(self.iter_usernames()):
            check_username = username if case_sensitive else username.lower()
            if pattern in check_username:
                yield i, username

    def get_stats(self) -> dict:
        """Get statistics about the username dataset."""
        data_size = self.filepath.stat().st_size
        index_size = self.index_filepath.stat().st_size

        return {
            "total_usernames": self.total_count,
            "data_file_size": data_size,
            "index_file_size": index_size,
            "total_size": data_size + index_size,
            "avg_username_length": data_size / self.total_count
            if self.total_count > 0
            else 0,
            "data_file_size_gb": data_size / (1024**3),
            "index_file_size_mb": index_size / (1024**2),
            "total_size_gb": (data_size + index_size) / (1024**3),
        }

    def sample(self, count: int = 10) -> list[str]:
        """Get a random sample of usernames."""
        import random

        if count > self.total_count:
            count = self.total_count

        indices = random.sample(range(self.total_count), count)
        return [self[i] for i in indices]


def main():
    """Demonstrate the username reader functionality."""

    data_file = Path(__file__).parent / "usernames.dat"

    if not data_file.exists():
        print(f"Username data file not found: {data_file}")
        print("Please run generate.py first to create the data.")
        return

    # Initialize reader
    reader = UsernameReader(data_file)

    # Display statistics
    stats = reader.get_stats()
    print("Username Dataset Statistics:")
    print(f"  Total usernames: {stats['total_usernames']:,}")
    print(f"  Data file size: {stats['data_file_size_gb']:.2f} GB")
    print(f"  Index file size: {stats['index_file_size_mb']:.2f} MB")
    print(f"  Average username length: {stats['avg_username_length']:.1f} characters")
    print()

    # Show first 10 usernames
    print("First 10 usernames:")
    for i, username in enumerate(reader.iter_usernames(0, 10)):
        print(f"  {i}: {username}")
    print()

    # Show last 10 usernames
    print("Last 10 usernames:")
    start_idx = len(reader) - 10
    for i, username in enumerate(reader.iter_usernames(start_idx)):
        print(f"  {start_idx + i}: {username}")
    print()

    # Random sample
    print("Random sample of 5 usernames:")
    for username in reader.sample(5):
        print(f"  {username}")
    print()

    # Demonstrate batch iteration (first 3 batches)
    print("First 3 batches of 5 usernames each:")
    for batch_num, batch in enumerate(reader.iter_batch(5, 0, 15)):
        print(f"  Batch {batch_num + 1}: {batch}")
    print()

    # Search example
    print("Searching for usernames containing 'admin':")
    admin_count = 0
    for index, username in reader.search("admin"):
        if admin_count < 5:  # Show first 5 matches
            print(f"  Index {index}: {username}")
        admin_count += 1
        if admin_count >= 100:  # Limit search to avoid long output
            break
    print(f"  Found {admin_count} usernames containing 'admin' (showing first 5)")


if __name__ == "__main__":
    main()
