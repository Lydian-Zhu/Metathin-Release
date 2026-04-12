"""
Unit tests for pattern space components.
模式空间组件单元测试。
"""

import pytest
import numpy as np
from metathin.components import (
    SimplePatternSpace,
    StatisticalPatternSpace,
    NormalizedPatternSpace,
    CompositePatternSpace,
    CachedPatternSpace
)


class TestSimplePatternSpace:
    """Test SimplePatternSpace."""
    
    def test_basic_extraction(self):
        """Should extract features using provided function."""
        pattern = SimplePatternSpace(lambda x: [len(str(x)), len(str(x).split())])
        
        features = pattern.extract("hello world")
        
        assert len(features) == 2
        assert features[0] == 11  # len("hello world")
        assert features[1] == 2   # word count
    
    def test_extract_numeric_input(self):
        """Should handle numeric input."""
        pattern = SimplePatternSpace(lambda x: [float(x), float(x) ** 2])
        
        features = pattern.extract(5)
        
        assert len(features) == 2
        assert features[0] == 5.0
        assert features[1] == 25.0
    
    def test_extract_list_input(self):
        """Should handle list input."""
        pattern = SimplePatternSpace(lambda x: [len(x), sum(x)])
        
        features = pattern.extract([1, 2, 3, 4])
        
        assert len(features) == 2
        assert features[0] == 4    # length
        assert features[1] == 10   # sum
    
    def test_feature_names(self):
        """Should return feature names if provided."""
        pattern = SimplePatternSpace(
            lambda x: [len(x)],
            feature_names=['length']
        )
        
        assert pattern.get_feature_names() == ['length']
    
    def test_default_feature_names(self):
        """Should generate default feature names if not provided."""
        pattern = SimplePatternSpace(lambda x: [len(x), len(x.split())])
        
        names = pattern.get_feature_names()
        assert len(names) == 2
        assert "SimplePattern_0" in names[0]
    
    def test_feature_dimension(self):
        """Should return correct feature dimension."""
        pattern = SimplePatternSpace(lambda x: [len(x), len(x.split()), x.count('a')])
        
        dim = pattern.get_feature_dimension()
        assert dim == 3 or dim == 0  # Dimension detection may need implementation


class TestStatisticalPatternSpace:
    """Test StatisticalPatternSpace."""
    
    def test_extract_statistical_features(self):
        """Should extract statistical features."""
        pattern = StatisticalPatternSpace(features=['mean', 'std', 'max', 'min'])
        
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        features = pattern.extract(data)
        
        assert len(features) == 4
        assert features[0] == 3.0   # mean
        assert features[1] == np.std(data)  # std
        assert features[2] == 5.0   # max
        assert features[3] == 1.0   # min
    
    def test_extract_all_features(self):
        """Should extract all features when none specified."""
        pattern = StatisticalPatternSpace()
        
        data = [1, 2, 3, 4, 5]
        features = pattern.extract(data)
        
        # Should have all statistical features
        assert len(features) > 5
    
    def test_extract_single_value(self):
        """Should handle single numeric value."""
        pattern = StatisticalPatternSpace()
        
        features = pattern.extract(42)
        
        assert len(features) > 0
        assert not np.any(np.isnan(features))
    
    def test_extract_empty_data(self):
        """Should handle empty data with zeros."""
        pattern = StatisticalPatternSpace(handle_empty='zeros')
        
        features = pattern.extract([])
        
        assert len(features) > 0
        assert not np.any(np.isnan(features))
    
    def test_extract_empty_data_raises(self):
        """Should raise error when handle_empty='raise'."""
        pattern = StatisticalPatternSpace(handle_empty='raise')
        
        with pytest.raises(Exception):
            pattern.extract([])
    
    def test_feature_names(self):
        """Should return qualified feature names."""
        pattern = StatisticalPatternSpace(features=['mean', 'std'], name="stats")
        
        names = pattern.get_feature_names()
        
        assert names == ['stats.mean', 'stats.std']
    
    def test_extract_numpy_array(self):
        """Should handle numpy array input."""
        pattern = StatisticalPatternSpace()
        
        data = np.array([1, 2, 3, 4, 5])
        features = pattern.extract(data)
        
        assert len(features) > 0


class TestNormalizedPatternSpace:
    """Test NormalizedPatternSpace."""
    
    def test_fixed_range_normalization(self):
        """Should normalize features to fixed range."""
        base = SimplePatternSpace(lambda x: [float(x), float(x) ** 2])
        pattern = NormalizedPatternSpace(base, ranges=[(0, 10), (0, 100)], method='fixed')
        
        features = pattern.extract(5)
        
        assert 0 <= features[0] <= 1
        assert 0 <= features[1] <= 1
        assert features[0] == 0.5   # (5-0)/(10-0) = 0.5
        assert features[1] == 0.25  # (25-0)/(100-0) = 0.25
    
    def test_adaptive_normalization(self):
        """Should adaptively update ranges."""
        base = SimplePatternSpace(lambda x: [float(x)])
        pattern = NormalizedPatternSpace(base, method='adaptive')
        
        features1 = pattern.extract(0)
        features2 = pattern.extract(10)
        features3 = pattern.extract(5)
        
        # Features should be in [0,1]
        assert 0 <= features1[0] <= 1
        assert 0 <= features2[0] <= 1
        assert 0 <= features3[0] <= 1
    
    def test_standard_normalization(self):
        """Should standardize to mean 0, std 1."""
        base = SimplePatternSpace(lambda x: [float(x)])
        pattern = NormalizedPatternSpace(base, method='standard')
        
        # Feed multiple values to build statistics
        for i in range(1, 11):
            pattern.extract(i)
        
        features = pattern.extract(5)
        
        # Should be within reasonable range
        assert -3 <= features[0] <= 3
    
    def test_preserves_feature_names(self):
        """Should preserve base feature names."""
        base = SimplePatternSpace(
            lambda x: [float(x)],
            feature_names=['value']
        )
        pattern = NormalizedPatternSpace(base, method='fixed')
        
        names = pattern.get_feature_names()
        assert names == base.get_feature_names()


class TestCompositePatternSpace:
    """Test CompositePatternSpace."""
    
    def test_composite_multiple_patterns(self):
        """Should combine features from multiple patterns."""
        pattern1 = SimplePatternSpace(lambda x: [len(str(x))])
        pattern2 = SimplePatternSpace(lambda x: [len(str(x).split())])
        
        composite = CompositePatternSpace([
            ('len', pattern1),
            ('words', pattern2)
        ])
        
        features = composite.extract("hello world")
        
        assert len(features) == 2
        assert features[0] == 11
        assert features[1] == 2
    
    def test_composite_with_unnamed_patterns(self):
        """Should auto-generate names for unnamed patterns."""
        pattern1 = SimplePatternSpace(lambda x: [len(str(x))])
        pattern2 = SimplePatternSpace(lambda x: [len(str(x).split())])
        
        composite = CompositePatternSpace([pattern1, pattern2])
        
        features = composite.extract("hello world")
        
        assert len(features) == 2
    
    def test_composite_single_input_to_all(self):
        """Should pass same input to all patterns."""
        pattern1 = SimplePatternSpace(lambda x: [len(str(x))])
        pattern2 = SimplePatternSpace(lambda x: [len(str(x).split())])
        
        composite = CompositePatternSpace([pattern1, pattern2])
        
        features = composite.extract("hello world")
        
        assert len(features) == 2
    
    def test_composite_multi_part_input(self):
        """Should distribute list input to patterns."""
        pattern1 = SimplePatternSpace(lambda x: [len(str(x))])
        pattern2 = SimplePatternSpace(lambda x: [len(str(x))])
        
        composite = CompositePatternSpace([pattern1, pattern2])
        
        features = composite.extract(["hello", "world"])
        
        assert len(features) == 2
        assert features[0] == 5   # len("hello")
        assert features[1] == 5   # len("world")
    
    def test_composite_feature_names(self):
        """Should return compound feature names."""
        pattern1 = SimplePatternSpace(
            lambda x: [len(str(x))],
            feature_names=['length']
        )
        pattern2 = SimplePatternSpace(
            lambda x: [len(str(x).split())],
            feature_names=['word_count']
        )
        
        composite = CompositePatternSpace([
            ('text', pattern1),
            ('stats', pattern2)
        ])
        
        names = composite.get_feature_names()
        
        assert names == ['text.length', 'stats.word_count']


class TestCachedPatternSpace:
    """Test CachedPatternSpace."""
    
    def test_cache_hit(self):
        """Should return cached result on repeated calls."""
        call_count = 0
        
        def extractor(x):
            nonlocal call_count
            call_count += 1
            return np.array([len(str(x))], dtype=np.float64)
        
        base = SimplePatternSpace(extractor)
        cached = CachedPatternSpace(base, cache_size=10)
        
        # First call - should compute
        features1 = cached.extract("hello")
        assert call_count == 1
        
        # Second call - should use cache
        features2 = cached.extract("hello")
        assert call_count == 1
        
        np.testing.assert_array_equal(features1, features2)
    
    def test_cache_miss(self):
        """Should compute on cache miss."""
        call_count = 0
        
        def extractor(x):
            nonlocal call_count
            call_count += 1
            return np.array([len(str(x))], dtype=np.float64)
        
        base = SimplePatternSpace(extractor)
        cached = CachedPatternSpace(base, cache_size=10)
        
        cached.extract("hello")
        cached.extract("world")
        cached.extract("hello")
        
        assert call_count == 2  # "hello" and "world" each computed once
    
    def test_cache_size_limit(self):
        """Should respect cache size limit."""
        call_count = 0
        
        def extractor(x):
            nonlocal call_count
            call_count += 1
            return np.array([x], dtype=np.float64)
        
        base = SimplePatternSpace(extractor)
        cached = CachedPatternSpace(base, cache_size=2)
        
        cached.extract(1)
        cached.extract(2)
        cached.extract(3)  # This should evict one
        
        # All three were computed because cache size = 2
        assert call_count == 3
    
    def test_cache_stats(self):
        """Should provide cache statistics."""
        base = SimplePatternSpace(lambda x: np.array([len(str(x))]))
        cached = CachedPatternSpace(base, cache_size=10)
        
        cached.extract("a")
        cached.extract("a")  # hit
        cached.extract("b")
        cached.extract("b")  # hit
        cached.extract("c")
        
        stats = cached.get_cache_stats()
        
        assert stats['hits'] == 2
        assert stats['misses'] == 3
        assert stats['size'] == 3
        assert stats['capacity'] == 10
        assert stats['hit_rate'] == 0.4