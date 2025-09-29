import time

from ..algorithms.algorithm import Algorithm, SearchResult
from ..data_structures.sorted_list import SortedList


class BinarySearch(Algorithm):
    """Binary search on sorted username data."""

    def __init__(self, sorted_list: SortedList):
        self.sorted_list = sorted_list
        super().__init__(sorted_list)

    def search(self, target: str) -> SearchResult:
        """Search for target using binary search."""
        if not self.validate_target(target):
            return SearchResult(found=False, index=-1, comparisons=0, time_taken=0.0)

        start_time = time.perf_counter()
        comparisons = 0

        left, right = 0, len(self.sorted_list) - 1
        result_index = -1

        while left <= right:
            mid = (left + right) // 2
            comparisons += 1

            mid_value = self.sorted_list[mid]

            if mid_value == target:
                result_index = mid
                break
            elif mid_value < target:
                left = mid + 1
            else:
                right = mid - 1

        end_time = time.perf_counter()
        time_taken = end_time - start_time

        return SearchResult(
            found=result_index != -1,
            index=result_index,
            comparisons=comparisons,
            time_taken=time_taken,
        )

    def get_algorithm_name(self) -> str:
        return "Binary Search"
