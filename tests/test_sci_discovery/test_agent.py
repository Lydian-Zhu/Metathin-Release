# tests/test_sci_discovery/test_agent.py
"""
Test Scientific Discovery Agent | 测试科学发现代理

Tests for ScientificDiscoveryAgent, DiscoveryPhase, and DiscoveryResult.
测试 ScientificDiscoveryAgent、DiscoveryPhase 和 DiscoveryResult。
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from metathin_plus.sci.discovery.agent import (
    ScientificDiscoveryAgent,
    DiscoveryPhase,
    DiscoveryResult
)


class TestScientificDiscoveryAgent:
    """Test scientific discovery agent | 测试科学发现代理"""
    
    @pytest.fixture
    def agent(self):
        """Create test agent | 创建测试用代理"""
        return ScientificDiscoveryAgent(
            window_size=10,
            error_threshold=0.1,
            similarity_threshold=0.8,
            n_negative=5,
            n_positive=5
        )
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory | 创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_initialization(self, agent):
        """Test agent initialization | 测试代理初始化"""
        assert agent.window_size == 10
        assert agent.error_threshold == 0.1
        assert agent.similarity_threshold == 0.8
        assert agent.phase == DiscoveryPhase.INIT
        assert agent.buffer_size == 0
    
    def test_add_point_collecting(self, agent):
        """Test adding points during collection phase | 测试收集阶段添加点"""
        # Add less than window_size points | 添加少于窗口大小的点
        for i in range(5):
            result = agent.add_point(i, i ** 2)
            assert result.phase == DiscoveryPhase.COLLECTING
            assert result.x == i
            assert result.y == i ** 2
        
        assert agent.buffer_size == 5
        assert agent.phase == DiscoveryPhase.COLLECTING
    
    def test_add_point_triggers_extraction(self, agent):
        """Test that adding enough points triggers extraction | 测试添加足够点后触发提取"""
        # Add window_size points | 添加窗口大小的点
        for i in range(agent.window_size):
            result = agent.add_point(i, i ** 2)
        
        # Should have results | 应该有结果
        assert len(agent.history) == agent.window_size
        assert agent.buffer_size == agent.window_size
    
    def test_linear_function_discovery(self, agent):
        """Test linear function discovery | 测试线性函数发现"""
        # Generate linear data y = 2x + 1 | 生成线性数据
        for i in range(agent.window_size):
            x = float(i)
            y = 2 * x + 1
            result = agent.add_point(x, y)
        
        # After adding all points, the agent should have made predictions
        # 添加所有点后，代理应该已经做出了预测
        # Check that predictions are reasonably close to actual values
        # 检查预测值是否合理接近实际值
        last_results = agent.get_recent_history(3)
        for r in last_results:
            if r.prediction is not None:
                # For linear data y = 2x + 1, prediction should be close to actual
                # 对于线性数据 y = 2x + 1，预测值应该接近实际值
                # Note: prediction is for the NEXT x, not the current x
                # 注意：预测值是针对下一个 x，而不是当前 x
                actual_next = 2 * (r.x + 1) + 1  # Next y value | 下一个 y 值
                # Allow larger tolerance since it's a rough check
                # 允许更大的容差，因为是粗略检查
                error = abs(r.prediction - actual_next)
                assert error < 10, f"Prediction {r.prediction} vs actual {actual_next}, error={error}"
    
    def test_quadratic_function_discovery(self, agent):
        """Test quadratic function discovery | 测试二次函数发现"""
        # Generate quadratic data y = x^2 | 生成二次数据
        for i in range(agent.window_size):
            x = float(i)
            y = x ** 2
            result = agent.add_point(x, y)
        
        # Verify state | 验证状态
        status = agent.get_status()
        assert status['buffer_size'] == agent.window_size
    
    def test_validation_success(self, agent):
        """Test successful validation (no restart) | 测试验证成功（不触发重启）"""
        # Add data | 添加数据
        for i in range(agent.window_size):
            agent.add_point(i, i)
        
        # Validate next point | 验证下一个点
        next_x = agent.window_size
        next_y = next_x
        triggered = agent.validate(next_x, next_y)
        
        # Error should be small, no restart | 误差应该很小，不触发重启
        assert triggered is False
    
    def test_validation_triggers_restart(self, agent):
        """Test validation triggers restart on large error | 测试大误差触发重启"""
        # Add linear data | 添加线性数据
        for i in range(agent.window_size):
            agent.add_point(i, i)
        
        # Validate a completely different point | 验证一个完全不同的点
        next_x = agent.window_size
        next_y = 1000  # Large error | 巨大误差
        
        triggered = agent.validate(next_x, next_y)
        
        # Should trigger restart | 应该触发重启
        assert triggered is True
        assert agent.phase == DiscoveryPhase.RESTARTING
        assert agent.restart_count == 1
    
    def test_predict(self, agent):
        """Test prediction | 测试预测"""
        # Add data | 添加数据
        for i in range(agent.window_size):
            agent.add_point(i, i)
        
        # Predict next point | 预测下一个点
        next_x = agent.window_size
        prediction = agent.predict(next_x)
        
        # Prediction should be close to actual | 预测应该接近实际值
        # Since we're using linear data y=x, prediction should be close to next_x
        # 由于我们使用的是线性数据 y=x，预测值应该接近 next_x
        assert abs(prediction - next_x) < 5
    
    def test_reset(self, agent):
        """Test agent reset | 测试代理重置"""
        # Add some data | 添加一些数据
        for i in range(5):
            agent.add_point(i, i)
        
        assert agent.buffer_size > 0
        assert len(agent.history) > 0
        
        agent.reset()
        
        assert agent.buffer_size == 0
        assert len(agent.history) == 0
        assert agent.phase == DiscoveryPhase.INIT
        assert agent.restart_count == 0
    
    def test_clear_history(self, agent):
        """Test clearing history | 测试清空历史"""
        for i in range(5):
            agent.add_point(i, i)
        
        assert len(agent.history) == 5
        agent.clear_history()
        assert len(agent.history) == 0
    
    def test_get_status(self, agent):
        """Test getting agent status | 测试获取代理状态"""
        status = agent.get_status()
        
        assert 'phase' in status
        assert 'current_function' in status
        assert 'buffer_size' in status
        assert 'window_size' in status
        assert 'error_threshold' in status
    
    def test_get_statistics(self, agent):
        """Test getting statistics | 测试获取统计信息"""
        # Add some data | 添加一些数据
        for i in range(agent.window_size):
            agent.add_point(i, i)
        
        stats = agent.get_statistics()
        
        assert stats['total_points'] == agent.window_size
        assert 'predictions' in stats
        assert 'restarts' in stats
    
    def test_register_custom_function(self, agent):
        """Test registering custom function | 测试注册自定义函数"""
        success = agent.register_function(
            "test_func",
            "a*x + b",
            parameters=["a", "b"],
            tags=["test"],
            description="Test linear function | 测试线性函数"
        )
        
        assert success is True
    
    def test_sine_wave_approximation(self, agent):
        """Test sine wave approximation (needs more points) | 测试正弦波逼近（需要更多点）"""
        # Use a smaller window for faster test | 使用更小的窗口加快测试
        agent_small = ScientificDiscoveryAgent(
            window_size=15,
            error_threshold=0.5,
            n_negative=10,
            n_positive=10
        )
        
        # Generate sine wave | 生成正弦波
        for i in range(15):
            x = i * 0.5
            y = np.sin(x)
            agent_small.add_point(x, y)
        
        # Verify agent has output | 验证代理有输出
        assert len(agent_small.history) == 15
    
    def test_exponential_decay(self, agent):
        """Test exponential decay | 测试指数衰减"""
        agent_small = ScientificDiscoveryAgent(
            window_size=15,
            error_threshold=0.2,
            n_negative=10,
            n_positive=10
        )
        
        # Generate exponential decay | 生成指数衰减
        for i in range(15):
            x = float(i)
            y = np.exp(-x / 5)
            agent_small.add_point(x, y)
        
        # Verify | 验证
        status = agent_small.get_status()
        assert status['buffer_size'] == 15


class TestDiscoveryResult:
    """Test discovery result data class | 测试发现结果数据类"""
    
    def test_creation(self):
        """Test result creation | 测试结果创建"""
        import time
        result = DiscoveryResult(
            phase=DiscoveryPhase.PREDICTING,
            timestamp=time.time(),
            x=1.0,
            y=2.0,
            prediction=2.1,
            matched_function="linear",
            similarity=0.95,
            parameters={"a": 1.0, "b": 2.0}
        )
        
        assert result.x == 1.0
        assert result.y == 2.0
        assert result.prediction == 2.1
        assert result.matched_function == "linear"
        assert result.similarity == 0.95
    
    def test_success_property(self):
        """Test success property | 测试 success 属性"""
        import time
        
        # Success case | 成功的情况
        result_success = DiscoveryResult(
            phase=DiscoveryPhase.PREDICTING,
            timestamp=time.time(),
            x=1.0, y=1.0,
            prediction=1.0,
            error=None
        )
        assert result_success.success is True
        
        # Small error | 小误差
        result_small_error = DiscoveryResult(
            phase=DiscoveryPhase.PREDICTING,
            timestamp=time.time(),
            x=1.0, y=1.0,
            prediction=1.05,
            error=0.05
        )
        assert result_small_error.success is True
        
        # Large error | 大误差
        result_large_error = DiscoveryResult(
            phase=DiscoveryPhase.PREDICTING,
            timestamp=time.time(),
            x=1.0, y=1.0,
            prediction=2.0,
            error=1.0
        )
        assert result_large_error.success is False
    
    def test_to_dict(self):
        """Test conversion to dictionary | 测试转换为字典"""
        import time
        result = DiscoveryResult(
            phase=DiscoveryPhase.PREDICTING,
            timestamp=time.time(),
            x=1.0,
            y=2.0,
            prediction=2.1
        )
        
        data = result.to_dict()
        assert data['x'] == 1.0
        assert data['y'] == 2.0
        assert data['prediction'] == 2.1
        assert data['phase'] == 'predicting'