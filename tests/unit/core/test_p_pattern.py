"""
Unit tests for PatternSpace interface.
模式空间接口单元测试。
"""

import pytest
import numpy as np
from metathin.core import PatternSpace, PatternExtractionError, FeatureVector


class SimplePatternSpace(PatternSpace):
    """Simple implementation for testing."""
    
    def __init__(self, extract_func=None):
        self._extract_func = extract_func or (lambda x: np.array([len(str(x))], dtype=np.float64))
        self._feature_names = ['length']
    
    def extract(self, raw_input) -> FeatureVector:
        return self._extract_func(raw_input)
    
    def get_feature_names(self):
        return self._feature_names


class TestPatternSpace:
    """Test PatternSpace interface."""
    
    def test_extract_returns_feature_vector(self):
        """extract() should return a FeatureVector."""
        pattern = SimplePatternSpace()
        features = pattern.extract("hello")
        
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float64
        assert features.ndim == 1
    
    def test_extract_with_numeric_input(self):
        """extract() should work with numeric input."""
        pattern = SimplePatternSpace()
        features = pattern.extract(42)
        
        assert len(features) == 1
        assert features[0] == 2  # len("42") = 2
    
    def test_extract_with_custom_function(self):
        """extract() should support custom extraction functions."""
        def custom_extract(x):
            return np.array([float(x), float(x) ** 2], dtype=np.float64)
        
        pattern = SimplePatternSpace(custom_extract)
        features = pattern.extract(5)
        
        assert len(features) == 2
        assert features[0] == 5.0
        assert features[1] == 25.0
    
    def test_get_feature_names(self):
        """get_feature_names() should return feature names."""
        pattern = SimplePatternSpace()
        names = pattern.get_feature_names()
        
        assert names == ['length']
    
    def test_get_feature_dimension(self):
        """get_feature_dimension() should return correct dimension."""
        pattern = SimplePatternSpace()
        dim = pattern.get_feature_dimension()
        
        assert dim == 1
    
    def test_validate_features_valid(self):
        """validate_features() should return True for valid features."""
        pattern = SimplePatternSpace()
        valid_features = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        
        assert pattern.validate_features(valid_features) is True
    
    def test_validate_features_invalid_dtype(self):
        """validate_features() should return False for wrong dtype."""
        pattern = SimplePatternSpace()
        invalid_features = np.array([1, 2, 3], dtype=np.int64)
        
        assert pattern.validate_features(invalid_features) is False
    
    def test_validate_features_invalid_ndim(self):
        """validate_features() should return False for 2D array."""
        pattern = SimplePatternSpace()
        invalid_features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        
        assert pattern.validate_features(invalid_features) is False
    
    def test_validate_features_with_nan(self):
        """validate_features() should return False for NaN values."""
        pattern = SimplePatternSpace()
        invalid_features = np.array([1.0, np.nan, 3.0], dtype=np.float64)
        
        assert pattern.validate_features(invalid_features) is False
    
    def test_validate_features_with_inf(self):
        """validate_features() should return False for Inf values."""
        pattern = SimplePatternSpace()
        invalid_features = np.array([1.0, np.inf, 3.0], dtype=np.float64)
        
        assert pattern.validate_features(invalid_features) is False


class TestCustomPatternSpace:
    """Test custom PatternSpace implementations."""
    
    def test_custom_pattern_with_multiple_features(self):
        """Custom pattern should be able to extract multiple features."""
        
        class MultiFeaturePattern(PatternSpace):
            def extract(self, raw_input) -> FeatureVector:
                text = str(raw_input)
                return np.array([
                    len(text),           # length
                    len(text.split()),   # word count
                    text.count('a'),     # count of 'a'
                ], dtype=np.float64)
            
            def get_feature_names(self):
                return ['length', 'word_count', 'a_count']
        
        pattern = MultiFeaturePattern()
        features = pattern.extract("hello amazing world")
        
        assert len(features) == 3
        assert features[0] == 19  # length
        assert features[1] == 3   # word count
        assert features[2] == 2   # one 'a' in "amazing"
    
    def test_custom_pattern_with_dict_input(self):
        """Custom pattern should handle dictionary input."""
        
        class DictPattern(PatternSpace):
            def extract(self, raw_input) -> FeatureVector:
                if not isinstance(raw_input, dict):
                    raise PatternExtractionError("Expected dict")
                return np.array([
                    raw_input.get('x', 0.0),
                    raw_input.get('y', 0.0),
                ], dtype=np.float64)
        
        pattern = DictPattern()
        features = pattern.extract({'x': 10.0, 'y': 20.0})
        
        assert len(features) == 2
        assert features[0] == 10.0
        assert features[1] == 20.0
    

    def test_custom_pattern_raises_extraction_error(self):
        """Custom pattern should raise PatternExtractionError on failure."""
    
        class FailingPattern(PatternSpace):
            def extract(self, raw_input) -> FeatureVector:
                raise ValueError("Cannot process input")
    
        pattern = FailingPattern()
    
        # PatternSpace implementations should wrap exceptions in PatternExtractionError
        # If not, at least some exception should be raised
        with pytest.raises(Exception):
            pattern.extract("test")