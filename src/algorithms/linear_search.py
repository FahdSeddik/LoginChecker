import time

from data.reader import UsernameReader

from .algorithm import Algorithm, SearchResult


class LinearSearch(Algorithm):
    """
    Linear Search Algorithm Implementation

    A comprehensive implementation of linear search with performance tracking
    and statistical analysis capabilities.

    Time Complexity: O(n) - worst case, best case O(1), average case O(n/2)
    Space Complexity: O(1) - constant extra space
    """

    def __init__(self, reader: UsernameReader, track_performance: bool = True):
        """
        Initialize the Linear Search instance.

        Args:
            reader: UsernameReader instance to search through
            track_performance: Whether to track performance metrics
        """
        super().__init__(reader)
        self.track_performance = track_performance
        self.total_comparisons = 0
        self.total_searches = 0
        self.total_time = 0.0

    def search(self, target: str) -> SearchResult:
        """
        Search for the target username using the Algorithm interface.

        Args:
            target: Username to search for

        Returns:
            A SearchResult object containing the search outcome
        """
        if not self.validate_target(target):
            return SearchResult(found=False, index=-1, comparisons=0, time_taken=0.0)
        if not self.reader or len(self.reader) == 0:
            raise ValueError("Reader cannot be None or empty")

        start_time = time.perf_counter()
        comparisons = 0

        # Linear search implementation
        for i in range(len(self.reader)):
            comparisons += 1
            element = self.reader[i]
            if element == target:
                end_time = time.perf_counter()
                time_taken = end_time - start_time

                if self.track_performance:
                    self._update_statistics(comparisons, time_taken)

                return SearchResult(
                    found=True, index=i, comparisons=comparisons, time_taken=time_taken
                )

        # Element not found
        end_time = time.perf_counter()
        time_taken = end_time - start_time

        if self.track_performance:
            self._update_statistics(comparisons, time_taken)

        return SearchResult(
            found=False, index=-1, comparisons=comparisons, time_taken=time_taken
        )

    def get_algorithm_name(self) -> str:
        """
        Get the name of the algorithm.

        Returns:
            A string representing the algorithm name
        """
        return "LinearSearch"

    def _update_statistics(self, comparisons: int, time_taken: float) -> None:
        """
        Update internal performance statistics.

        Args:
            comparisons: Number of comparisons in this search
            time_taken: Time taken for this search
        """
        self.total_comparisons += comparisons
        self.total_searches += 1
        self.total_time += time_taken

    def get_performance_stats(self) -> dict:
        """
        Get performance statistics for all searches performed.

        Returns:
            Dictionary containing performance metrics
        """
        if self.total_searches == 0:
            return {
                "total_searches": 0,
                "total_comparisons": 0,
                "total_time": 0.0,
                "avg_comparisons": 0.0,
                "avg_time": 0.0,
                "comparisons_per_second": 0.0,
            }

        return {
            "total_searches": self.total_searches,
            "total_comparisons": self.total_comparisons,
            "total_time": self.total_time,
            "avg_comparisons": self.total_comparisons / self.total_searches,
            "avg_time": self.total_time / self.total_searches,
            "comparisons_per_second": self.total_comparisons / self.total_time
            if self.total_time > 0
            else 0.0,
        }

    def reset_statistics(self) -> None:
        """Reset all performance statistics."""
        self.total_comparisons = 0
        self.total_searches = 0
        self.total_time = 0.0
