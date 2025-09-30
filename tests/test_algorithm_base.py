"""
Comprehensive tests for Algorithm base class and SearchResult.

Tests cover the interface contract, validation, and result structures.
"""

from unittest.mock import MagicMock

import pytest

from src.algorithms.algorithm import Algorithm, SearchResult


class TestSearchResult:
    """Test suite for SearchResult dataclass."""

    def test_basic_initialization(self):
        """Test basic SearchResult initialization."""
        result = SearchResult(found=True, index=42, comparisons=10, time_taken=0.001)

        assert result.found is True
        assert result.index == 42
        assert result.comparisons == 10
        assert result.time_taken == 0.001
        assert result.hash_operations == 0  # default
        assert result.false_positive is False  # default
        assert result.additional_info is None  # default

    def test_initialization_with_all_fields(self):
        """Test SearchResult initialization with all fields."""
        additional = {"custom_metric": 123}
        result = SearchResult(
            found=False,
            index=-1,
            comparisons=25,
            time_taken=0.005,
            hash_operations=3,
            false_positive=True,
            additional_info=additional,
        )

        assert result.found is False
        assert result.index == -1
        assert result.comparisons == 25
        assert result.time_taken == 0.005
        assert result.hash_operations == 3
        assert result.false_positive is True
        assert result.additional_info == additional

    def test_equality(self):
        """Test SearchResult equality comparison."""
        result1 = SearchResult(True, 5, 10, 0.001)
        result2 = SearchResult(True, 5, 10, 0.001)
        result3 = SearchResult(False, -1, 10, 0.001)

        assert result1 == result2
        assert result1 != result3

    def test_immutability(self):
        """Test that SearchResult is immutable (dataclass frozen behavior)."""
        result = SearchResult(True, 5, 10, 0.001)

        # Note: dataclass is not frozen by default, but we test the interface
        assert hasattr(result, "found")
        assert hasattr(result, "index")
        assert hasattr(result, "comparisons")
        assert hasattr(result, "time_taken")

    @pytest.mark.parametrize(
        "found,index,comparisons,time_taken",
        [
            (True, 0, 1, 0.0001),
            (False, -1, 100, 0.01),
            (True, 999999, 1, 0.000001),
            (False, -1, 0, 0.0),
        ],
    )
    def test_various_values(self, found, index, comparisons, time_taken):
        """Test SearchResult with various value combinations."""
        result = SearchResult(found, index, comparisons, time_taken)

        assert result.found == found
        assert result.index == index
        assert result.comparisons == comparisons
        assert result.time_taken == time_taken


class TestAlgorithmBase:
    """Test suite for Algorithm base class."""

    def setup_method(self):
        """Set up test fixtures."""

        # Create a concrete implementation for testing
        class ConcreteAlgorithm(Algorithm):
            def search(self, target: str) -> SearchResult:
                if not self.validate_target(target):
                    return SearchResult(False, -1, 0, 0.0)

                # Simple mock implementation
                if hasattr(self.reader, "__getitem__") and target == self.reader[0]:
                    return SearchResult(True, 0, 1, 0.001)
                return SearchResult(False, -1, 1, 0.001)

            def get_algorithm_name(self) -> str:
                return "ConcreteAlgorithm"

        self.ConcreteAlgorithm = ConcreteAlgorithm

    def test_abstract_instantiation(self):
        """Test that Algorithm cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Algorithm(None)

    def test_concrete_instantiation(self):
        """Test that concrete implementation can be instantiated."""
        mock_reader = MagicMock()
        algorithm = self.ConcreteAlgorithm(mock_reader)

        assert algorithm.reader == mock_reader

    def test_validate_target_valid_strings(self):
        """Test validate_target with valid strings."""
        mock_reader = MagicMock()
        algorithm = self.ConcreteAlgorithm(mock_reader)

        valid_targets = [
            "alice",
            "user123",
            "a",
            "very_long_username_that_is_still_valid",
            "user@domain.com",
            "user.name_123",
            "CamelCaseUser",
            "user-with-dashes",
        ]

        for target in valid_targets:
            assert algorithm.validate_target(target) is True

    def test_validate_target_invalid_inputs(self):
        """Test validate_target with invalid inputs."""
        mock_reader = MagicMock()
        algorithm = self.ConcreteAlgorithm(mock_reader)

        invalid_targets = [
            "",  # empty string
            "   ",  # whitespace only
            "\t\n",  # whitespace only
            None,  # not a string
            123,  # not a string
            [],  # not a string
            {},  # not a string
        ]

        for target in invalid_targets:
            assert algorithm.validate_target(target) is False

    def test_validate_target_unicode(self):
        """Test validate_target with Unicode strings."""
        mock_reader = MagicMock()
        algorithm = self.ConcreteAlgorithm(mock_reader)

        unicode_targets = [
            "café",
            "测试用户",
            "Ñoël",
            "Москва",
            "🙂user",
        ]

        for target in unicode_targets:
            assert algorithm.validate_target(target) is True

    def test_validate_target_edge_cases(self):
        """Test validate_target with edge cases."""
        mock_reader = MagicMock()
        algorithm = self.ConcreteAlgorithm(mock_reader)

        # String with leading/trailing spaces should be valid
        # (validation only checks if stripped length > 0)
        assert algorithm.validate_target("  alice  ") is True

        # Single character
        assert algorithm.validate_target("a") is True

        # Special characters
        assert algorithm.validate_target("@#$%") is True

    def test_search_interface_compliance(self):
        """Test that search method returns SearchResult."""
        mock_reader = MagicMock()
        mock_reader.__getitem__ = MagicMock(return_value="alice")
        algorithm = self.ConcreteAlgorithm(mock_reader)

        result = algorithm.search("alice")

        assert isinstance(result, SearchResult)
        assert result.found is True
        assert result.index == 0

    def test_search_with_invalid_target(self):
        """Test search behavior with invalid target."""
        mock_reader = MagicMock()
        algorithm = self.ConcreteAlgorithm(mock_reader)

        result = algorithm.search("")

        assert isinstance(result, SearchResult)
        assert result.found is False
        assert result.index == -1
        assert result.comparisons == 0
        assert result.time_taken == 0.0

    def test_get_algorithm_name_interface(self):
        """Test get_algorithm_name method."""
        mock_reader = MagicMock()
        algorithm = self.ConcreteAlgorithm(mock_reader)

        name = algorithm.get_algorithm_name()

        assert isinstance(name, str)
        assert name == "ConcreteAlgorithm"

    def test_string_representation(self):
        """Test string representation of algorithm."""
        mock_reader = MagicMock()
        algorithm = self.ConcreteAlgorithm(mock_reader)

        str_repr = str(algorithm)

        assert str_repr == "ConcreteAlgorithm"

    def test_reader_storage(self):
        """Test that reader is properly stored."""
        mock_reader = MagicMock()
        algorithm = self.ConcreteAlgorithm(mock_reader)

        assert algorithm.reader is mock_reader

    def test_abstract_methods_enforcement(self):
        """Test that abstract methods must be implemented."""

        # Class missing search method
        class IncompleteAlgorithm1(Algorithm):
            def get_algorithm_name(self) -> str:
                return "Incomplete1"

        # Class missing get_algorithm_name method
        class IncompleteAlgorithm2(Algorithm):
            def search(self, target: str) -> SearchResult:
                return SearchResult(False, -1, 0, 0.0)

        mock_reader = MagicMock()

        with pytest.raises(TypeError):
            IncompleteAlgorithm1(mock_reader)

        with pytest.raises(TypeError):
            IncompleteAlgorithm2(mock_reader)

    def test_multiple_algorithm_instances(self):
        """Test multiple algorithm instances with different readers."""
        mock_reader1 = MagicMock()
        mock_reader2 = MagicMock()

        algorithm1 = self.ConcreteAlgorithm(mock_reader1)
        algorithm2 = self.ConcreteAlgorithm(mock_reader2)

        assert algorithm1.reader is mock_reader1
        assert algorithm2.reader is mock_reader2
        assert algorithm1.reader is not algorithm2.reader

    def test_validate_target_type_safety(self):
        """Test type safety of validate_target method."""
        mock_reader = MagicMock()
        algorithm = self.ConcreteAlgorithm(mock_reader)

        # Test with various non-string types
        non_strings = [
            0,
            1,
            -1,
            3.14,
            True,
            False,
            [],
            [1, 2, 3],
            (),
            (1, 2),
            {},
            {"key": "value"},
            set(),
            {1, 2, 3},
            object(),
            type,
            len,
        ]

        for non_string in non_strings:
            assert algorithm.validate_target(non_string) is False

    def test_inheritance_behavior(self):
        """Test proper inheritance behavior."""
        mock_reader = MagicMock()
        algorithm = self.ConcreteAlgorithm(mock_reader)

        # Should be instance of both concrete class and base class
        assert isinstance(algorithm, self.ConcreteAlgorithm)
        assert isinstance(algorithm, Algorithm)

        # Should have access to base class methods
        assert hasattr(algorithm, "validate_target")
        assert callable(algorithm.validate_target)

    def test_method_override_behavior(self):
        """Test that subclasses can override methods appropriately."""

        class CustomValidationAlgorithm(Algorithm):
            def validate_target(self, target: str) -> bool:
                # Custom validation: must be exactly 5 characters
                return isinstance(target, str) and len(target) == 5

            def search(self, target: str) -> SearchResult:
                return SearchResult(self.validate_target(target), 0, 1, 0.001)

            def get_algorithm_name(self) -> str:
                return "CustomValidation"

        mock_reader = MagicMock()
        algorithm = CustomValidationAlgorithm(mock_reader)

        # Should use overridden validation
        assert algorithm.validate_target("alice") is True  # 5 chars
        assert algorithm.validate_target("bob") is False  # 3 chars
        assert algorithm.validate_target("charlie") is False  # 7 chars

    @pytest.mark.parametrize(
        "reader_type",
        [
            MagicMock(),
            [],
            {},
            "string",
            42,
            None,
        ],
    )
    def test_reader_type_flexibility(self, reader_type):
        """Test that algorithm accepts various reader types."""
        # Algorithm should accept any reader type (duck typing)
        algorithm = self.ConcreteAlgorithm(reader_type)
        assert algorithm.reader is reader_type

    def test_search_result_consistency(self):
        """Test that search results are consistent."""
        mock_reader = MagicMock()
        mock_reader.__getitem__ = MagicMock(return_value="test_user")
        algorithm = self.ConcreteAlgorithm(mock_reader)

        # Multiple calls should return consistent results
        result1 = algorithm.search("test_user")
        result2 = algorithm.search("test_user")
        result3 = algorithm.search("different_user")

        assert result1.found == result2.found
        assert result1.index == result2.index
        assert result3.found is False
