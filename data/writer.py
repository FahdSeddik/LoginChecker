import struct
import time
from pathlib import Path
from typing import Iterator


class UsernameWriter:
    """Writes usernames in binary format with cumulative position index."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.index_filepath = filepath.with_suffix(".idx")

    def write_usernames(
        self, usernames: Iterator[str], batch_size: int = 100_000
    ) -> None:
        """Write usernames in binary format with cumulative position index."""
        print(f"Writing usernames to {self.filepath}...")
        start_time = time.time()

        count = 0
        with (
            open(self.filepath, "wb") as data_file,
            open(self.index_filepath, "wb") as index_file,
        ):
            cumulative_position = 0
            batch = []

            for username in usernames:
                batch.append(username)
                count += 1

                if len(batch) >= batch_size:
                    self._write_batch(batch, data_file, index_file, cumulative_position)
                    cumulative_position += sum(len(u.encode("utf-8")) for u in batch)
                    batch.clear()

                    if count % (batch_size * 10) == 0:
                        elapsed = time.time() - start_time
                        rate = count / elapsed if elapsed > 0 else 0
                        print(
                            f"Wrote {count:,} usernames - Rate: {rate:,.0f} usernames/sec"
                        )

            # Write remaining batch
            if batch:
                self._write_batch(batch, data_file, index_file, cumulative_position)

        elapsed = time.time() - start_time
        print(f"Writing complete! {count:,} usernames in {elapsed:.1f}s")

    def _write_batch(self, batch, data_file, index_file, start_position):
        """Write a batch of usernames."""
        cumulative_position = start_position

        for username in batch:
            username_bytes = username.encode("utf-8")

            # Write to index: cumulative position
            index_file.write(struct.pack("<Q", cumulative_position))

            # Write username to data file
            data_file.write(username_bytes)

            cumulative_position += len(username_bytes)
