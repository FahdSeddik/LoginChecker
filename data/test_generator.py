"""Test script for username generator and reader with smaller dataset."""

import sys
from pathlib import Path

from mimesis.locales import Locale

# Add parent directory to path for imports when running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from data.generate import EfficientUsernameStorage, UsernameGenerator
    from data.reader import UsernameReader
except ImportError:
    # Fallback for when running as module
    from .generate import EfficientUsernameStorage, UsernameGenerator
    from .reader import UsernameReader


def test_small_dataset():
    """Test with a smaller dataset to verify functionality."""

    # Test configuration
    TEST_COUNT = 100_000  # 100K usernames for testing
    BATCH_SIZE = 10_000
    SEED = 42

    # Setup paths
    data_dir = Path(__file__).parent
    test_file = data_dir / "test_usernames.dat"

    print(f"Testing with {TEST_COUNT:,} usernames...")

    # Clean up existing test files
    if test_file.exists():
        test_file.unlink()
    if test_file.with_suffix(".idx").exists():
        test_file.with_suffix(".idx").unlink()

    # Generate test data
    generator = UsernameGenerator(locale=Locale.EN, seed=SEED)
    storage = EfficientUsernameStorage(test_file)

    try:
        storage.store_usernames(generator, TEST_COUNT, BATCH_SIZE)

        # Test reader
        reader = UsernameReader(test_file)

        # Verify count
        assert len(reader) == TEST_COUNT, f"Expected {TEST_COUNT}, got {len(reader)}"
        print(f"[OK] Count verification passed: {len(reader):,} usernames")

        # Test random access
        first_username = reader[0]
        last_username = reader[-1]
        middle_username = reader[TEST_COUNT // 2]

        print("[OK] Random access test passed:")
        print(f"  First: {first_username}")
        print(f"  Middle: {middle_username}")
        print(f"  Last: {last_username}")

        # Test iteration
        iteration_count = 0
        for _ in reader.iter_usernames(0, 10):
            iteration_count += 1
        assert iteration_count == 10, f"Expected 10, got {iteration_count}"
        print("[OK] Iteration test passed")

        # Test batch iteration
        batch_count = 0
        total_in_batches = 0
        for batch in reader.iter_batch(1000, 0, 5000):
            batch_count += 1
            total_in_batches += len(batch)
        assert total_in_batches == 5000, f"Expected 5000, got {total_in_batches}"
        print(f"[OK] Batch iteration test passed: {batch_count} batches")

        # Show stats
        stats = reader.get_stats()
        print("[OK] Dataset stats:")
        print(f"  Total usernames: {stats['total_usernames']:,}")
        print(f"  Data file: {stats['data_file_size'] / 1024**2:.1f} MB")
        print(f"  Index file: {stats['index_file_size'] / 1024:.1f} KB")
        print(f"  Avg length: {stats['avg_username_length']:.1f} chars")

        print("\n[OK] All tests passed! Generator and reader working correctly.")

        # Clean up test files
        test_file.unlink()
        test_file.with_suffix(".idx").unlink()
        print("Test files cleaned up.")

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        raise


if __name__ == "__main__":
    test_small_dataset()
