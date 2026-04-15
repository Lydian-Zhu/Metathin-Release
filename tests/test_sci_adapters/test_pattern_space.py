# tests/test_sci_adapters/test_pattern_space.py
"""
Test Scientific Discovery Pattern Space Adapter | 测试科学发现模式空间适配器

Tests for SciPatternSpace - perception layer adapter.
测试 SciPatternSpace - 感知层适配器。
"""

import pytest
import numpy as np
from metathin_plus.sci.adapters.pattern_space import SciPatternSpace


class TestSciPatternSpace:
    """Test scientific discovery pattern space | 测试科学发现模式空间"""
    
    def test_initialization(self):
        """Test pattern space initialization | 测试模式空间初始化"""
        pattern = SciPatternSpace(window_size=50, name="test_pattern")
        
        assert pattern.window_size == 50
        assert pattern._name == "test_pattern"  # Use _name instead of name | 使用 _name 而不是 name
        assert pattern.buffer_size == 0
        assert pattern.is_ready is False
    
    def test_extract_single_value(self):
        """Test single value extraction (streaming mode) | 测试单值提取（流式模式）"""
        pattern = SciPatternSpace(window_size=10, n_negative=5, n_positive=5)
        
        # Add values less than window size | 添加不足窗口大小的值
        for i in range(5):
            features = pattern.extract(float(i))
            # Should return zero vector when data insufficient | 数据不足时应返回零向量
            assert np.all(features == 0)
        
        assert pattern.buffer_size == 5
        assert pattern.is_ready is False
    
    def test_extract_after_buffer_full(self):
        """Test extraction after buffer is full | 测试缓冲区满后提取"""
        pattern = SciPatternSpace(window_size=5, n_negative=2, n_positive=2)
        
        # Add enough values | 添加足够的值
        for i in range(5):
            features = pattern.extract(float(i))
        
        # Should have non-zero vector after 5th call | 第5次后应该有非零向量
        assert pattern.is_ready is True
        assert not np.all(features == 0)
        assert len(features) == pattern.get_feature_dimension()
    
    def test_extract_batch(self):
        """Test batch extraction | 测试批量提取"""
        pattern = SciPatternSpace(window_size=10, n_negative=5, n_positive=5)
        
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        features = pattern.extract(data)
        
        assert len(features) == pattern.get_feature_dimension()
        assert not np.all(features == 0)
    
    def test_extract_numpy_array(self):
        """Test numpy array extraction | 测试 numpy 数组提取"""
        pattern = SciPatternSpace(window_size=10, n_negative=5, n_positive=5)
        
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        features = pattern.extract(data)
        
        assert len(features) == pattern.get_feature_dimension()
    
    def test_extract_from_xy(self):
        """Test extraction from (x, y) pairs | 测试从 (x, y) 对提取"""
        pattern = SciPatternSpace(n_negative=5, n_positive=5)
        
        x_vals = np.linspace(0, 10, 20)
        y_vals = 2 * x_vals + 1
        
        features = pattern.extract_from_xy(x_vals, y_vals)
        assert len(features) == pattern.get_feature_dimension()
    
    def test_extract_from_xy_mismatch(self):
        """Test (x, y) length mismatch raises error | 测试 (x, y) 长度不匹配时抛出错误"""
        pattern = SciPatternSpace()
        
        x_vals = np.array([1, 2, 3])
        y_vals = np.array([1, 2])
        
        with pytest.raises(ValueError):
            pattern.extract_from_xy(x_vals, y_vals)
    
    def test_extract_none(self):
        """Test None input returns zero vector | 测试 None 输入返回零向量"""
        pattern = SciPatternSpace()
        features = pattern.extract(None)
        
        assert len(features) == pattern.get_feature_dimension()
        assert np.all(features == 0)
    
    def test_get_feature_names(self):
        """Test getting feature names | 测试获取特征名称"""
        pattern = SciPatternSpace(name="test", n_negative=2, n_positive=2)
        names = pattern.get_feature_names()
        
        assert len(names) == 5  # 2 + 2 + 1
        assert names[0] == "test_coeff_-2"
        assert names[2] == "test_coeff_0"
        assert names[4] == "test_coeff_2"
    
    def test_get_feature_dimension(self):
        """Test getting feature dimension | 测试获取特征维度"""
        pattern = SciPatternSpace(n_negative=10, n_positive=10)
        assert pattern.get_feature_dimension() == 21
        
        pattern_small = SciPatternSpace(n_negative=2, n_positive=2)
        assert pattern_small.get_feature_dimension() == 5
    
    def test_reset(self):
        """Test pattern space reset | 测试模式空间重置"""
        pattern = SciPatternSpace(window_size=10)
        
        # Add some data | 添加一些数据
        for i in range(5):
            pattern.extract(float(i))
        
        assert pattern.buffer_size == 5
        
        pattern.reset()
        assert pattern.buffer_size == 0
        assert pattern.is_ready is False
    
    def test_get_stats(self):
        """Test getting statistics | 测试获取统计信息"""
        pattern = SciPatternSpace(window_size=50, name="test")
        stats = pattern.get_stats()
        
        assert stats['name'] == "test"
        assert stats['window_size'] == 50
        assert stats['buffer_size'] == 0
        assert stats['is_ready'] is False
        assert stats['feature_dimension'] == 41
    
    def test_cache(self):
        """Test result caching | 测试结果缓存"""
        pattern = SciPatternSpace(window_size=5, use_cache=True)
        
        # Add data until buffer full | 添加数据直到缓冲区满
        for i in range(5):
            features1 = pattern.extract(float(i))
        
        # Extract same values again (should come from cache) | 再次提取相同的值（应该从缓存获取）
        pattern.reset()
        for i in range(5):
            features2 = pattern.extract(float(i))
        
        # Results should be identical | 结果应该相同
        np.testing.assert_array_equal(features1, features2)
    
    def test_repr(self):
        """Test string representation | 测试字符串表示"""
        pattern = SciPatternSpace(window_size=50, name="test")
        repr_str = repr(pattern)
        
        assert "SciPatternSpace" in repr_str
        assert "window=50" in repr_str