import struct
import time
from pathlib import Path
from typing import Iterator, Optional

from mimesis import Person
from mimesis.locales import Locale


class UsernameGenerator:
    """Generates usernames efficiently using mimesis with various patterns."""

    def __init__(self, locale: Locale = Locale.EN, seed: Optional[int] = None):
        self.person = Person(locale=locale, seed=seed)
        self.basic = ["C", "c", "U", "u", "L", "l", "D", "d", ""]
        self.connectors = [".", "_", "-", ""]
        # Required characters for mimesis username function
        required_chars = {"C", "U", "l"}
        patterns_set = set()
        for first in self.basic:
            for second in self.basic:
                for third in self.basic:
                    for connector1 in self.connectors:
                        for connector2 in self.connectors:
                            pattern = f"{first}{connector1}{second}{connector2}{third}"
                            # Only add patterns that are non-empty and contain at least one required character
                            if pattern and any(
                                char in pattern for char in required_chars
                            ):
                                patterns_set.add(pattern)
        self.patterns = list(patterns_set)
        self.p_idx = 0

    def generate_username(self) -> str:
        """Generate a single username without specifying patterns."""
        pattern = self.patterns[self.p_idx]
        self.p_idx = (self.p_idx + 1) % len(self.patterns)
        return self.person.username(mask=pattern, drange=(0, 9999))[:20]

    def generate_batch(self, count: int) -> Iterator[str]:
        """Generate a batch of usernames."""
        for _ in range(count):
            yield self.generate_username()


class EfficientUsernameStorage:
    """Stores usernames in an efficient binary format with length-only index."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.index_filepath = filepath.with_suffix(".idx")

    def store_usernames(
        self, generator: UsernameGenerator, total_count: int, batch_size: int = 100_000
    ) -> None:
        """Store usernames in binary format with length index for sequential access."""

        print(f"Generating {total_count:,} usernames...")
        start_time = time.time()

        with (
            open(self.filepath, "wb") as data_file,
            open(self.index_filepath, "wb") as index_file,
        ):
            for batch_start in range(0, total_count, batch_size):
                batch_end = min(batch_start + batch_size, total_count)
                current_batch_size = batch_end - batch_start

                # Generate batch of usernames
                usernames = list(generator.generate_batch(current_batch_size))

                # Write usernames and build index
                for username in usernames:
                    username_bytes = username.encode("utf-8")
                    length = len(username_bytes)

                    # Write to index: length (1 byte)
                    index_file.write(struct.pack("<B", length))

                    # Write username to data file
                    data_file.write(username_bytes)

                # Progress update
                if batch_start % (batch_size * 10) == 0:
                    elapsed = time.time() - start_time
                    rate = batch_end / elapsed if elapsed > 0 else 0
                    eta = (total_count - batch_end) / rate if rate > 0 else 0
                    print(
                        f"Progress: {batch_end:,}/{total_count:,} "
                        f"({100 * batch_end / total_count:.1f}%) - "
                        f"Rate: {rate:,.0f} usernames/sec - "
                        f"ETA: {eta:.0f}s"
                    )

        elapsed = time.time() - start_time
        print(
            f"Generation complete! {total_count:,} usernames in {elapsed:.1f}s "
            f"({total_count / elapsed:,.0f} usernames/sec)"
        )

        # Print file sizes
        data_size = self.filepath.stat().st_size
        index_size = self.index_filepath.stat().st_size
        print(f"Data file: {data_size:,} bytes ({data_size / 1024**3:.2f} GB)")
        print(f"Index file: {index_size:,} bytes ({index_size / 1024**2:.2f} MB)")


def main():
    """Generate 1 billion usernames efficiently."""

    # Configuration
    TOTAL_USERNAMES = 1_000_000_000  # 1 billion
    BATCH_SIZE = 100_000
    SEED = 42  # For reproducible results

    # Setup paths
    data_dir = Path(__file__).parent
    output_file = data_dir / "usernames.dat"

    # Initialize generator with seed for reproducibility
    generator = UsernameGenerator(locale=Locale.EN, seed=SEED)

    # Initialize storage
    storage = EfficientUsernameStorage(output_file)

    # Check if files already exist
    if output_file.exists():
        response = input(f"Output file {output_file} exists. Overwrite? (y/N): ")
        if response.lower() != "y":
            print("Aborted.")
            return

    # Generate and store usernames
    try:
        storage.store_usernames(generator, TOTAL_USERNAMES, BATCH_SIZE)
        print("Username generation completed successfully!")

    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
    except Exception as e:
        print(f"Error during generation: {e}")
        raise


if __name__ == "__main__":
    main()
