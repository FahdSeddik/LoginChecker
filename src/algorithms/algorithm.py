from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchResult:
    """
    Result of a search operation.

    Attributes:
        found: Whether the item was found
        index: Index of the item if found, -1 otherwise
        comparisons: Number of comparisons performed
        time_taken: Time taken for the search in seconds
        hash_operations: Number of hash operations performed (for hash-based algorithms)
        false_positive: Whether this is a false positive (for probabilistic structures)
        additional_info: Any additional algorithm-specific information
    """

    found: bool
    index: int
    comparisons: int
    time_taken: float
    hash_operations: int = 0
    false_positive: bool = False
    additional_info: Optional[dict] = None


class Algorithm(ABC):
    """
    Abstract base class for search algorithms.

    This class defines the interface that all search algorithms must implement
    to search for a target username using a given reader.
    """

    def __init__(self, reader):
        """
        Initialize the algorithm with a reader.

        Args:
            reader: A reader object that provides access to username data.
                   Expected to have methods like __getitem__, __len__, etc.
        """
        self.reader = reader

    @abstractmethod
    def search(self, target: str) -> SearchResult:
        """
        Search for the target username in the data.

        Args:
            target: The username to search for

        Returns:
            A SearchResult object containing the search outcome
        """
        pass

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """
        Get the name of the algorithm.

        Returns:
            A string representing the algorithm name
        """
        pass

    def validate_target(self, target: str) -> bool:
        """
        Validate that the target is a non-empty string.

        Args:
            target: The target username to validate

        Returns:
            True if target is valid, False otherwise
        """
        return isinstance(target, str) and len(target.strip()) > 0

    def __str__(self) -> str:
        """String representation of the algorithm."""
        return f"{self.get_algorithm_name()}"
