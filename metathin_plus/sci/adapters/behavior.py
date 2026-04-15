"""
Scientific Discovery Behavior | 科学发现行为
=============================================

Behavior adapter that wraps the ScientificDiscoveryAgent as a Metathin MetaBehavior.
将科学发现代理包装为 Metathin 元行为的适配器。

This behavior serves as the action layer (B) in Metathin,
executing scientific discovery and prediction tasks.
此行为作为 Metathin 中的行动层 (B)，
执行科学发现和预测任务。
"""

import numpy as np
from typing import Any, Dict, Optional, List, Union
import logging

from metathin.core.b_behavior import MetaBehavior
from metathin.core.types import FeatureVector

from ..discovery.agent import ScientificDiscoveryAgent, DiscoveryPhase, DiscoveryResult


class SciDiscoveryBehavior(MetaBehavior):
    """
    Scientific Discovery Behavior | 科学发现行为
    
    Wraps the ScientificDiscoveryAgent as a Metathin behavior.
    将科学发现代理包装为 Metathin 行为。
    
    This behavior enables the agent to:
        - Discover mathematical patterns in data | 发现数据中的数学模式
        - Predict future values | 预测未来值
        - Adapt to changing patterns | 适应变化的模式
        - Learn from errors | 从错误中学习
    
    Example | 示例:
        >>> behavior = SciDiscoveryBehavior(name="discoverer")
        >>> 
        >>> # Register custom function | 注册自定义函数
        >>> behavior.register_function("my_func", "a*sin(b*x)")
        >>> 
        >>> # Use in agent | 在代理中使用
        >>> agent = Metathin(behaviors=[behavior])
        >>> result = agent.think(x, y=y)  # y is actual value
    """
    
    def __init__(self,
                 name: str = "sci_discovery",
                 window_size: int = 50,
                 error_threshold: float = 0.1,
                 similarity_threshold: float = 0.9,
                 n_negative: int = 20,
                 n_positive: int = 20,
                 enable_special_functions: bool = False,
                 data_dir: Optional[str] = None):
        """
        Initialize scientific discovery behavior | 初始化科学发现行为
        
        Args:
            name: Behavior name | 行为名称
            window_size: Fitting window size | 拟合窗口大小
            error_threshold: Error threshold for restart | 重启误差阈值
            similarity_threshold: Function matching threshold | 函数匹配阈值
            n_negative: Number of negative power terms | 负幂项数量
            n_positive: Number of positive power terms | 正幂项数量
            enable_special_functions: Whether to enable special functions | 是否启用特殊函数
            data_dir: Directory for function library | 函数库目录
        """
        super().__init__()
        
        self._name = name
        self._window_size = window_size
        self._error_threshold = error_threshold
        self._similarity_threshold = similarity_threshold
        
        # Initialize the scientific discovery agent | 初始化科学发现代理
        self._agent = ScientificDiscoveryAgent(
            window_size=window_size,
            error_threshold=error_threshold,
            similarity_threshold=similarity_threshold,
            n_negative=n_negative,
            n_positive=n_positive,
            enable_special_functions=enable_special_functions,
            data_dir=data_dir
        )
        
        self._logger = logging.getLogger(f"metathin_sci.behavior.{name}")
        self._logger.info(f"Initialized: window={window_size}, threshold={error_threshold}")
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def agent(self) -> ScientificDiscoveryAgent:
        """Get the underlying scientific discovery agent | 获取底层的科学发现代理"""
        return self._agent
    
    def execute(self, features: FeatureVector, **kwargs) -> Any:
        """
        Execute scientific discovery | 执行科学发现
        
        Supported kwargs | 支持的参数:
            - x: X coordinate (required for prediction) | X 坐标
            - y: Actual Y value (optional, for training) | 实际 Y 值（可选，用于训练）
            - predict_only: If True, only predict without updating | 仅预测不更新
            - return_result: If True, return full DiscoveryResult | 返回完整结果
        
        Args:
            features: Feature vector (not used directly) | 特征向量（不直接使用）
            **kwargs: Additional parameters | 额外参数
            
        Returns:
            Any: Prediction value or DiscoveryResult | 预测值或发现结果
        """
        x = kwargs.get('x')
        y = kwargs.get('y')
        predict_only = kwargs.get('predict_only', False)
        return_result = kwargs.get('return_result', False)
        
        # Validate input | 验证输入
        if x is None:
            self._logger.warning("Missing 'x' parameter")
            return {'error': "Missing 'x' parameter"}
        
        # Predict only mode | 仅预测模式
        if predict_only:
            prediction = self._agent.predict(x)
            if return_result:
                return {
                    'prediction': prediction,
                    'matched_function': self._agent.current_function,
                    'phase': self._agent.phase.value
                }
            return prediction
        
        # Training mode (with y) | 训练模式（有 y）
        if y is not None:
            result = self._agent.add_point(x, y)
            
            # Validate with next point if available | 如果有下一个点则验证
            next_x = kwargs.get('next_x')
            next_y = kwargs.get('next_y')
            if next_x is not None and next_y is not None:
                self._agent.validate(next_x, next_y)
            
            if return_result:
                return {
                    'prediction': result.prediction,
                    'matched_function': result.matched_function,
                    'similarity': result.similarity,
                    'parameters': result.parameters,
                    'phase': result.phase.value,
                    'error': result.error
                }
            return result.prediction
        
        # Prediction mode (no y) | 预测模式（无 y）
        prediction = self._agent.predict(x)
        if return_result:
            return {
                'prediction': prediction,
                'matched_function': self._agent.current_function,
                'phase': self._agent.phase.value
            }
        return prediction
    
    def can_execute(self, features: FeatureVector) -> bool:
        """
        Check if behavior can execute | 检查行为是否可执行
        
        Always returns True for this behavior.
        此行为始终返回 True。
        
        Args:
            features: Feature vector (not used) | 特征向量（未使用）
            
        Returns:
            bool: Always True | 始终为 True
        """
        return True
    
    def get_complexity(self) -> float:
        """
        Get behavior complexity | 获取行为复杂度
        
        Complexity scales with window size.
        复杂度与窗口大小成正比。
        
        Returns:
            float: Complexity value | 复杂度值
        """
        return self._window_size / 10.0
    
    def before_execute(self, features: FeatureVector) -> None:
        """Pre-execution hook | 执行前钩子"""
        self._logger.debug(f"Preparing for discovery, buffer size: {self._agent.buffer_size}")
    
    def after_execute(self, result: Any, execution_time: float) -> None:
        """Post-execution hook | 执行后钩子"""
        if isinstance(result, dict) and result.get('matched_function'):
            self._logger.debug(f"Discovered: {result['matched_function']}")
    
    def on_error(self, error: Exception) -> None:
        """Error handling hook | 错误处理钩子"""
        self._logger.error(f"Discovery failed: {error}")
        self._agent.reset()
    
    # ============================================================
    # Public API | 公共接口
    # ============================================================
    
    def add_point(self, x: float, y: float) -> DiscoveryResult:
        """
        Add a data point and perform discovery | 添加数据点并执行发现
        
        Args:
            x: X coordinate | X 坐标
            y: Y value | Y 值
            
        Returns:
            DiscoveryResult: Discovery result | 发现结果
        """
        return self._agent.add_point(x, y)
    
    def validate(self, x: float, y: float) -> bool:
        """
        Validate prediction and check for restart | 验证预测并检查重启
        
        Args:
            x: X coordinate | X 坐标
            y: Actual Y value | 实际 Y 值
            
        Returns:
            bool: Whether restart was triggered | 是否触发重启
        """
        return self._agent.validate(x, y)
    
    def predict(self, x: float) -> float:
        """
        Predict next value | 预测下一个值
        
        Args:
            x: Next X coordinate | 下一个 X 坐标
            
        Returns:
            float: Predicted value | 预测值
        """
        return self._agent.predict(x)
    
    def register_function(self, 
                          name: str, 
                          expr, 
                          parameters: List[str] = None,
                          tags: List[str] = None,
                          description: str = "") -> bool:
        """
        Register a custom function | 注册自定义函数
        
        Args:
            name: Function name | 函数名称
            expr: Function expression | 函数表达式
            parameters: Parameter names | 参数名称
            tags: Search tags | 搜索标签
            description: Function description | 函数描述
            
        Returns:
            bool: Success status | 成功状态
        """
        return self._agent.register_function(
            name, expr, 
            parameters=parameters, 
            tags=tags, 
            description=description
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get behavior status | 获取行为状态"""
        status = self._agent.get_status()
        status.update({
            'name': self._name,
            'window_size': self._window_size,
            'error_threshold': self._error_threshold,
            'similarity_threshold': self._similarity_threshold
        })
        return status
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics | 获取详细统计"""
        return self._agent.get_statistics()
    
    def get_matched_function(self) -> Optional[str]:
        """Get currently matched function name | 获取当前匹配的函数名称"""
        return self._agent.current_function
    
    def get_current_parameters(self) -> Dict[str, float]:
        """Get current fitted parameters | 获取当前拟合的参数"""
        return self._agent.current_parameters
    
    def reset(self) -> None:
        """Reset the behavior | 重置行为"""
        self._agent.reset()
        self._logger.info("Behavior reset")
    
    def clear_history(self) -> None:
        """Clear discovery history | 清空发现历史"""
        self._agent.clear_history()
    
    def save_library(self) -> bool:
        """Save user function library | 保存用户函数库"""
        return self._agent.save_library()
    
    def __repr__(self) -> str:
        return (f"SciDiscoveryBehavior(name='{self._name}', "
                f"window={self._window_size}, "
                f"current_func={self._agent.current_function}, "
                f"phase={self._agent.phase.value})")