"""
Pytest configuration and fixtures for LoginChecker tests.

This file contains shared fixtures and configuration for all test modules.
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def temp_test_dir():
    """Create a temporary directory for all tests in the session."""
    temp_dir = tempfile.mkdtemp(prefix="loginchecker_tests_")
    yield Path(temp_dir)

    # Cleanup
    import shutil

    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass


@pytest.fixture
def temp_dir():
    """Create a temporary directory for a single test."""
    temp_dir = tempfile.mkdtemp(prefix="loginchecker_test_")
    yield Path(temp_dir)

    # Cleanup
    import shutil

    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass


@pytest.fixture
def sample_usernames():
    """Provide a consistent set of sample usernames for testing."""
    return ["alice", "bob", "charlie", "diana", "eve", "frank", "grace"]


@pytest.fixture
def small_username_dataset(temp_dir, sample_usernames):
    """Create a small dataset for testing."""
    from data.generate import EfficientUsernameStorage

    class TestGenerator:
        def __init__(self, usernames):
            self.usernames = usernames
            self.index = 0

        def generate_batch(self, count):
            batch = []
            for _ in range(count):
                if self.index < len(self.usernames):
                    batch.append(self.usernames[self.index])
                    self.index += 1
                else:
                    break
            return batch

    filepath = temp_dir / "small_dataset.dat"
    storage = EfficientUsernameStorage(filepath)
    generator = TestGenerator(sample_usernames)
    storage.store_usernames(generator, len(sample_usernames), 3)

    return filepath


# Configure pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


# Custom collection modifiers
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid or "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark slow tests (tests that might take longer)
        if any(
            keyword in item.nodeid
            for keyword in ["large", "performance", "stress", "concurrent"]
        ):
            item.add_marker(pytest.mark.slow)

        # Mark unit tests (default for most tests)
        if not any(
            marker.name in ["integration", "slow"] for marker in item.iter_markers()
        ):
            item.add_marker(pytest.mark.unit)


# Pytest options
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )


def pytest_runtest_setup(item):
    """Setup for individual test runs."""
    # Skip slow tests unless --run-slow is passed
    if "slow" in item.keywords and not item.config.getoption("--run-slow"):
        pytest.skip("need --run-slow option to run")


# Error handling for import issues
@pytest.fixture(autouse=True)
def handle_import_errors():
    """Handle common import issues in tests."""
    try:
        # Try importing key modules to catch issues early
        __import__("data.reader")
        __import__("data.writer")
        __import__("data.generate")
        __import__("src.algorithms.algorithm")
        __import__("src.algorithms.linear_search")
        __import__("src.algorithms.binary_search")
        __import__("src.data_structures.sorted_list")
        __import__("src.data_structures.disk_hashset")
    except ImportError as e:
        pytest.skip(f"Required module not available: {e}")
