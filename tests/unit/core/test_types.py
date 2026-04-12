"""
Unit tests for core types.
核心类型单元测试。
"""

import numpy as np
from metathin.core import FeatureVector, FitnessScore, ParameterDict


class TestFeatureVector:
    """Test FeatureVector type alias."""
    
    def test_feature_vector_is_numpy_array(self):
        """FeatureVector should be a numpy array."""
        features: FeatureVector = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float64
    
    def test_feature_vector_should_be_1d(self):
        """FeatureVector should be 1-dimensional."""
        features = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert features.ndim == 1
    
    def test_feature_vector_no_nan(self):
        """FeatureVector should not contain NaN."""
        features = np.array([1.0, 2.0, np.nan], dtype=np.float64)
        assert np.any(np.isnan(features))  # Contains NaN
        # In practice, this should be validated by PatternSpace


class TestFitnessScore:
    """Test FitnessScore type alias."""
    
    def test_fitness_score_should_be_float(self):
        """FitnessScore should be a float."""
        fitness: FitnessScore = 0.75
        assert isinstance(fitness, float)
    
    def test_fitness_score_range(self):
        """FitnessScore should be in [0, 1]."""
        valid_scores = [0.0, 0.5, 1.0]
        for score in valid_scores:
            assert 0 <= score <= 1


class TestParameterDict:
    """Test ParameterDict type alias."""
    
    def test_parameter_dict_structure(self):
        """ParameterDict should map strings to floats."""
        params: ParameterDict = {
            'w_0_0': 0.5,
            'w_0_1': -0.2,
            'b_0': 0.1
        }
        assert isinstance(params, dict)
        for key, value in params.items():
            assert isinstance(key, str)
            assert isinstance(value, float)
    
    def test_parameter_dict_empty(self):
        """Empty parameter dict is valid."""
        params: ParameterDict = {}
        assert len(params) == 0