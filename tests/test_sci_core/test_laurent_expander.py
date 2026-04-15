# tests/test_sci_core/test_laurent_expander.py
"""
Test Laurent Series Expander | 测试洛朗级数展开器

Tests for LaurentExpander class and convenience functions.
测试 LaurentExpander 类和便捷函数。
"""

import pytest
import numpy as np
from metathin_plus.sci.core.laurent_expander import (
    LaurentExpander,
    create_default_expander,
    expand_function
)


class TestLaurentExpander:
    """Test Laurent expander functionality | 测试洛朗展开器功能"""
    
    def test_default_creation(self):
        """Test default expander creation | 测试默认展开器创建"""
        expander = LaurentExpander()
        assert expander.config.n_negative == 20
        assert expander.config.n_positive == 20
        assert expander.config.dimension == 41
    
    def test_custom_creation(self):
        """Test custom expander creation | 测试自定义展开器创建"""
        expander = LaurentExpander(n_negative=10, n_positive=15, center=2.0)
        assert expander.config.n_negative == 10
        assert expander.config.n_positive == 15
        assert expander.config.center == 2.0
    
    def test_expand_data(self):
        """Test expansion from data points | 测试从数据点展开"""
        expander = LaurentExpander(n_negative=2, n_positive=2)
        
        # Linear function y = 2x + 1 | 线性函数
        x_vals = np.linspace(-5, 5, 20)
        y_vals = 2 * x_vals + 1
        
        vector = expander.expand_data(x_vals, y_vals)
        assert len(vector) == 5
        
        # Reconstruct and verify | 重建并验证
        func = expander.reconstruct(vector)
        test_x = 3.0
        original = 2 * test_x + 1
        reconstructed = func(test_x)
        assert abs(reconstructed - original) < 0.1
    
    def test_expand_data_with_center(self):
        """Test expansion with custom center point | 测试带自定义中心点的展开"""
        expander = LaurentExpander(n_negative=2, n_positive=2, center=1.0)
        
        # Data centered at x=1 | 以 x=1 为中心的数据
        x_vals = np.linspace(0, 2, 20)
        y_vals = (x_vals - 1) ** 2
        
        vector = expander.expand_data(x_vals, y_vals)
        assert len(vector) == 5
    
    def test_compare_cosine(self):
        """Test cosine similarity comparison | 测试余弦相似度比较"""
        expander = LaurentExpander(n_negative=2, n_positive=2)
        
        vec1 = expander.expand_series(np.array([1, 0, 0, 0, 0]))
        vec2 = expander.expand_series(np.array([1, 0, 0, 0, 0]))
        
        similarity = expander.compare(vec1, vec2, metric='cosine')
        assert similarity == 1.0
    
    def test_compare_euclidean(self):
        """Test Euclidean distance comparison | 测试欧氏距离比较"""
        expander = LaurentExpander(n_negative=2, n_positive=2)
        
        vec1 = expander.expand_series(np.array([1, 0, 0, 0, 0]))
        vec2 = expander.expand_series(np.array([2, 0, 0, 0, 0]))
        
        distance = expander.compare(vec1, vec2, metric='euclidean')
        assert distance == 1.0
    
    def test_compare_manhattan(self):
        """Test Manhattan distance comparison | 测试曼哈顿距离比较"""
        expander = LaurentExpander(n_negative=2, n_positive=2)
        
        vec1 = expander.expand_series(np.array([1, 0, 0, 0, 0]))
        vec2 = expander.expand_series(np.array([2, 3, 0, 0, 0]))
        
        distance = expander.compare(vec1, vec2, metric='manhattan')
        assert distance == 4.0
    
    def test_cache(self):
        """Test result caching | 测试结果缓存"""
        expander = LaurentExpander(use_cache=True)
        stats = expander.get_cache_stats()
        assert stats['use_cache'] is True
        
        expander.clear_cache()
        stats = expander.get_cache_stats()
        assert stats['cache_size'] == 0
    
    def test_create_default_expander(self):
        """Test default expander factory function | 测试默认展开器工厂函数"""
        expander = create_default_expander()
        assert expander.config.n_negative == 20
        assert expander.config.n_positive == 20
    
    def test_expand_function_numerical(self):
        """Test numerical function expansion | 测试数值函数展开"""
        def f(x):
            return x ** 2
        
        vector = expand_function(f, x_min=-5, x_max=5, n_samples=100)
        assert len(vector) == 41


@pytest.mark.slow
class TestLaurentExpanderSymbolic:
    """Test symbolic expansion (requires SymPy) | 测试符号展开（需要 SymPy）"""
    
    def test_expand_symbolic_sin(self):
        """Test sin(x) symbolic expansion | 测试 sin(x) 符号展开"""
        try:
            from sympy import symbols, sin
            expander = LaurentExpander(n_negative=5, n_positive=5)
            x = symbols('x')
            
            vector = expander.expand_symbolic(sin(x), x)
            assert len(vector) == 11
            
        except ImportError:
            pytest.skip("SymPy not installed | SymPy 未安装")
    
    def test_expand_symbolic_exp(self):
        """Test exp(x) symbolic expansion | 测试 exp(x) 符号展开"""
        try:
            from sympy import symbols, exp
            expander = LaurentExpander(n_negative=5, n_positive=5)
            x = symbols('x')
            
            vector = expander.expand_symbolic(exp(x), x)
            assert len(vector) == 11
            
        except ImportError:
            pytest.skip("SymPy not installed | SymPy 未安装")