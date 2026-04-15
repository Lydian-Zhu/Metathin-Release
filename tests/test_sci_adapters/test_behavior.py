# tests/test_sci_adapters/test_behavior.py
"""
Test Scientific Discovery Behavior Adapter | 测试科学发现行为适配器

Tests for SciDiscoveryBehavior - action layer adapter.
测试 SciDiscoveryBehavior - 行动层适配器。
"""

import pytest
import numpy as np
from metathin_plus.sci.adapters.behavior import SciDiscoveryBehavior


# Fixture for behavior | behavior 的 fixture
@pytest.fixture
def behavior():
    """Create a SciDiscoveryBehavior instance for testing | 创建用于测试的 SciDiscoveryBehavior 实例"""
    return SciDiscoveryBehavior(
        name="test_behavior",
        window_size=20,
        error_threshold=0.2,
        n_negative=10,
        n_positive=10
    )


class TestSciDiscoveryBehavior:
    """Test scientific discovery behavior | 测试科学发现行为"""
    
    def test_initialization(self):
        """Test behavior initialization | 测试行为初始化"""
        behavior = SciDiscoveryBehavior(
            name="test_behavior",
            window_size=20,
            error_threshold=0.1
        )
        
        assert behavior.name == "test_behavior"
        assert behavior.agent.window_size == 20
        assert behavior.agent.error_threshold == 0.1
    
    def test_execute_predict_only(self, behavior):
        """Test predict-only mode | 测试仅预测模式"""
        # Add some data first | 先添加一些数据
        for i in range(10):
            behavior.add_point(float(i), float(i))
        
        # Predict mode | 预测模式
        result = behavior.execute(
            features=np.array([0]),  # Not used | 未使用
            x=10.0,
            predict_only=True
        )
        
        assert isinstance(result, float) or isinstance(result, dict)
    
    def test_execute_with_y(self, behavior):
        """Test execution with y value (training mode) | 测试带 y 值的执行（训练模式）"""
        result = behavior.execute(
            features=np.array([0]),
            x=0.0,
            y=0.0
        )
        
        # Should return prediction | 应该返回预测值
        assert result is not None
    
    def test_execute_with_return_result(self, behavior):
        """Test execution returning full result | 测试返回完整结果的执行"""
        # Add some data | 添加一些数据
        for i in range(10):
            behavior.add_point(float(i), float(i))
        
        result = behavior.execute(
            features=np.array([0]),
            x=10.0,
            return_result=True
        )
        
        assert isinstance(result, dict)
        assert 'prediction' in result
        assert 'phase' in result
    
    def test_execute_missing_x(self, behavior):
        """Test execution with missing x parameter | 测试缺少 x 参数的执行"""
        result = behavior.execute(features=np.array([0]))
        
        assert isinstance(result, dict)
        assert 'error' in result
    
    def test_add_point(self, behavior):
        """Test adding data point | 测试添加数据点"""
        result = behavior.add_point(1.0, 2.0)
        
        assert result.x == 1.0
        assert result.y == 2.0
    
    def test_validate(self, behavior):
        """Test validation | 测试验证"""
        # Add data | 添加数据
        for i in range(10):
            behavior.add_point(float(i), float(i))
        
        # Validate | 验证
        triggered = behavior.validate(10.0, 10.0)
        assert triggered is False
    
    def test_predict(self, behavior):
        """Test prediction | 测试预测"""
        # Add data | 添加数据
        for i in range(10):
            behavior.add_point(float(i), float(i))
        
        prediction = behavior.predict(10.0)
        assert isinstance(prediction, float)
    
    def test_register_function(self, behavior):
        """Test registering custom function | 测试注册自定义函数"""
        success = behavior.register_function(
            "test_func",
            "a*x + b",
            parameters=["a", "b"],
            tags=["test"]
        )
        
        assert success is True
    
    def test_get_status(self, behavior):
        """Test getting status | 测试获取状态"""
        status = behavior.get_status()
        
        assert status['name'] == "test_behavior"
        assert 'phase' in status
        assert 'buffer_size' in status
        assert 'window_size' in status
    
    def test_get_statistics(self, behavior):
        """Test getting statistics | 测试获取统计信息"""
        stats = behavior.get_statistics()
        
        assert 'total_points' in stats
    
    def test_get_matched_function(self, behavior):
        """Test getting matched function name | 测试获取匹配的函数名称"""
        # Add data | 添加数据
        for i in range(20):
            behavior.add_point(float(i), float(i))
        
        func = behavior.get_matched_function()
        # May be None or function name | 可能为 None 或函数名
        assert func is None or isinstance(func, str)
    
    def test_get_current_parameters(self, behavior):
        """Test getting current fitted parameters | 测试获取当前拟合参数"""
        params = behavior.get_current_parameters()
        assert isinstance(params, dict)
    
    def test_reset(self, behavior):
        """Test behavior reset | 测试行为重置"""
        # Add data | 添加数据
        for i in range(5):
            behavior.add_point(float(i), float(i))
        
        assert behavior.agent.buffer_size > 0
        
        behavior.reset()
        
        assert behavior.agent.buffer_size == 0
    
    def test_clear_history(self, behavior):
        """Test clearing history | 测试清空历史"""
        for i in range(5):
            behavior.add_point(float(i), float(i))
        
        assert len(behavior.agent.history) > 0
        
        behavior.clear_history()
        
        assert len(behavior.agent.history) == 0
    
    def test_can_execute(self, behavior):
        """Test can_execute method | 测试 can_execute 方法"""
        assert behavior.can_execute(np.array([0])) is True
    
    def test_get_complexity(self, behavior):
        """Test complexity calculation | 测试复杂度计算"""
        complexity = behavior.get_complexity()
        assert complexity > 0
    
    def test_repr(self, behavior):
        """Test string representation | 测试字符串表示"""
        repr_str = repr(behavior)
        assert "SciDiscoveryBehavior" in repr_str
        assert behavior.name in repr_str


class TestSciDiscoveryBehaviorWithData:
    """Test behavior with real data | 使用真实数据测试行为"""
    
    @pytest.fixture
    def behavior(self):
        """Create behavior for data tests | 创建用于数据测试的行为"""
        return SciDiscoveryBehavior(
            name="test",
            window_size=20,
            error_threshold=0.2,
            n_negative=10,
            n_positive=10
        )
    
    def test_linear_data(self, behavior):
        """Test with linear data | 测试线性数据"""
        # Add linear data | 添加线性数据
        for i in range(20):
            x = float(i)
            y = 2 * x + 1
            behavior.add_point(x, y)
        
        # Predict next point | 预测下一个点
        prediction = behavior.predict(20.0)
        # Prediction should be close to 41 | 预测应该接近 41
        assert abs(prediction - 41) < 5
    
    def test_quadratic_data(self, behavior):
        """Test with quadratic data | 测试二次数据"""
        for i in range(20):
            x = float(i)
            y = x ** 2
            behavior.add_point(x, y)
        
        # Predict next point | 预测下一个点
        prediction = behavior.predict(20.0)
        # Prediction should be close to 400 | 预测应该接近 400
        assert abs(prediction - 400) < 50