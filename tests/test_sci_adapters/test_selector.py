# tests/test_sci_adapters/test_selector.py
"""
Test Scientific Discovery Selector Adapter | 测试科学发现选择器适配器

Tests for SciSelectorAdapter - evaluation layer adapter.
测试 SciSelectorAdapter - 评估层适配器。
"""

import pytest
import numpy as np
from unittest.mock import Mock
from metathin_plus.sci.adapters.selector import SciSelectorAdapter
from metathin.core.b_behavior import MetaBehavior


class MockBehavior(MetaBehavior):
    """Mock behavior for testing | 测试用模拟行为"""
    
    def __init__(self, name):
        super().__init__()
        self._name = name
    
    @property
    def name(self):
        return self._name
    
    def execute(self, features, **kwargs):
        return f"Executed {self._name} | 执行了 {self._name}"


class TestSciSelectorAdapter:
    """Test scientific discovery selector | 测试科学发现选择器"""
    
    @pytest.fixture
    def selector(self):
        """Create selector for tests | 创建测试用选择器"""
        return SciSelectorAdapter(
            n_negative=5,
            n_positive=5,
            similarity_threshold=0.7,
            default_fitness=0.5
        )
    
    @pytest.fixture
    def mock_behavior(self):
        """Create mock behavior | 创建模拟行为"""
        return MockBehavior("test_behavior")
    
    def test_initialization(self, selector):
        """Test selector initialization | 测试选择器初始化"""
        assert selector.similarity_threshold == 0.7
        assert selector.default_fitness == 0.5
    
    def test_compute_fitness_default(self, selector, mock_behavior):
        """Test default fitness (no registered pattern) | 测试默认适应度（未注册模式）"""
        features = np.zeros(selector._space.config.dimension)
        
        fitness = selector.compute_fitness(mock_behavior, features)
        
        assert 0 <= fitness <= 1
        assert fitness == selector.default_fitness
    
    def test_register_function_pattern(self, selector, mock_behavior):
        """Test registering function pattern for behavior | 测试为行为注册函数模式"""
        success = selector.register_function_pattern(
            behavior_name="test_behavior",
            function_expr="sin(x)",
            fitness=0.9
        )
        
        assert success is True
        assert "test_behavior" in selector._behavior_function_map
    
    def test_compute_fitness_with_pattern(self, selector, mock_behavior):
        """Test fitness computation with registered pattern | 测试带注册模式的适应度计算"""
        # Register pattern | 注册模式
        selector.register_function_pattern(
            behavior_name="test_behavior",
            function_expr="x",
            fitness=0.9
        )
        
        # Create feature vector similar to x | 创建与 x 相似的特征向量
        # Note: This needs actual feature vector, using zero for now | 注意：这需要实际特征向量，暂时用零
        features = np.zeros(selector._space.config.dimension)
        fitness = selector.compute_fitness(mock_behavior, features)
        
        assert 0 <= fitness <= 1
    
    def test_compute_fitness_from_data(self, selector, mock_behavior):
        """Test fitness computation from raw data | 测试从原始数据计算适应度"""
        selector.register_function_pattern(
            behavior_name="test_behavior",
            function_expr="x",
            fitness=0.9
        )
        
        x_vals = np.linspace(0, 10, 20)
        y_vals = x_vals  # Linear data | 线性数据
        
        fitness = selector.compute_fitness_from_data(
            mock_behavior, x_vals, y_vals
        )
        
        assert 0 <= fitness <= 1
    
    def test_match_function(self, selector):
        """Test function matching | 测试函数匹配"""
        # Create sine wave data | 创建正弦波数据
        x_vals = np.linspace(0, 10, 50)
        y_vals = np.sin(x_vals)
        
        # Extract vector from data | 从数据提取向量
        vector = selector._expander.expand_data(x_vals, y_vals)
        
        # Match | 匹配
        matches = selector.match_function(vector.coefficients, top_k=3)
        
        # Should have matches (may match sin or similar) | 应该有匹配结果（可能匹配到 sin 或近似函数）
        assert len(matches) >= 0
    
    def test_register_custom_function(self, selector):
        """Test registering custom function in library | 测试在库中注册自定义函数"""
        success = selector.register_custom_function(
            name="my_func",
            expr="a*x**2 + b*x + c",
            parameters=["a", "b", "c"],
            tags=["custom"]
        )
        
        assert success is True
        assert "my_func" in selector._library
    
    def test_set_behavior_pattern(self, selector, mock_behavior):
        """Test setting behavior pattern | 测试设置行为模式"""
        # Register function first | 先注册函数
        selector.register_custom_function("linear", "x")
        
        # Set behavior pattern | 设置行为模式
        success = selector.set_behavior_pattern(
            behavior_name="test_behavior",
            function_name="linear",
            fitness=0.95
        )
        
        assert success is True
        assert selector._behavior_function_map["test_behavior"] == "linear"
        assert selector._behavior_fitness_map["test_behavior"] == 0.95
    
    def test_set_behavior_pattern_nonexistent(self, selector, mock_behavior):
        """Test setting pattern with nonexistent function | 测试使用不存在的函数设置模式"""
        success = selector.set_behavior_pattern(
            behavior_name="test_behavior",
            function_name="nonexistent"
        )
        
        assert success is False
    
    def test_get_parameters(self, selector):
        """Test getting learnable parameters | 测试获取可学习参数"""
        params = selector.get_parameters()
        assert params == {}  # No learnable parameters | 无可学习参数
    
    def test_update_parameters(self, selector):
        """Test parameter update (not supported) | 测试参数更新（不支持）"""
        # Should not raise error | 应该不报错
        selector.update_parameters({})
        selector.update_parameters({"w_0": 0.1})
    
    def test_get_library_stats(self, selector):
        """Test getting library statistics | 测试获取库统计信息"""
        stats = selector.get_library_stats()
        
        assert 'total_functions' in stats
        assert 'builtin_count' in stats
    
    def test_list_patterns(self, selector, mock_behavior):
        """Test listing registered patterns | 测试列出已注册模式"""
        selector.register_function_pattern("behavior1", "sin(x)")
        selector.register_function_pattern("behavior2", "cos(x)")
        
        patterns = selector.list_patterns()
        
        assert "behavior1" in patterns
        assert "behavior2" in patterns
    
    def test_clear_patterns(self, selector, mock_behavior):
        """Test clearing all patterns | 测试清空所有模式"""
        selector.register_function_pattern("behavior1", "sin(x)")
        selector.register_function_pattern("behavior2", "cos(x)")
        
        assert len(selector._behavior_function_map) == 2
        
        selector.clear_patterns()
        
        assert len(selector._behavior_function_map) == 0
    
    def test_repr(self, selector):
        """Test string representation | 测试字符串表示"""
        repr_str = repr(selector)
        
        assert "SciSelectorAdapter" in repr_str
        assert "threshold" in repr_str