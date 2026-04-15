# metathin_plus/chaos/base.py
"""
Base Classes for Chaos Module | 混沌模块基类
=============================================

Defines core data structures used across all chaos components.
定义所有混沌组件使用的核心数据结构。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypeVar, Generic
import time
import numpy as np


T = TypeVar('T')
"""
Type variable for system state | 系统状态类型变量
Can be float, List[float], Dict, or custom class
可以是 float、List[float]、Dict 或自定义类
"""


@dataclass
class SystemState(Generic[T]):
    """
    System state container | 系统状态容器
    
    Wraps the raw system state with metadata for Metathin processing.
    包装原始系统状态并附带元数据供 Metathin 处理。
    
    Attributes:
        data: Raw state data | 原始状态数据
        timestamp: Time of observation | 观测时间戳
        metadata: Additional information | 额外信息
    """
    data: T
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_value(self) -> float:
        """
        Extract numeric prediction target from state.
        从状态中提取数值预测目标。
        
        Returns:
            float: Prediction target value | 预测目标值
        """
        # If data is numeric | 如果数据是数值
        if isinstance(self.data, (int, float)):
            return float(self.data)
        
        # If data is list/tuple | 如果是列表/元组
        if isinstance(self.data, (list, tuple)) and len(self.data) > 0:
            try:
                return float(self.data[0])
            except (TypeError, ValueError):
                pass
        
        # If data is dict | 如果是字典
        if isinstance(self.data, dict):
            for v in self.data.values():
                if isinstance(v, (int, float)):
                    return float(v)
        
        # If has value attribute | 如果有 value 属性
        if hasattr(self.data, 'value'):
            try:
                return float(self.data.value)
            except (TypeError, ValueError):
                pass
        
        return 0.0
    
    def to_features(self) -> List[float]:
        """
        Convert state to feature list.
        将状态转换为特征列表。
        
        Returns:
            List[float]: Feature vector | 特征向量
        """
        features = [self.get_value()]
        
        # Add timestamp offset (normalized) | 添加时间戳偏移（归一化）
        if 'start_time' in self.metadata:
            offset = self.timestamp - self.metadata['start_time']
            features.append(offset)
        
        # Add any numeric metadata | 添加数值元数据
        for k, v in self.metadata.items():
            if isinstance(v, (int, float)) and k != 'start_time':
                features.append(float(v))
        
        return features


@dataclass
class PredictionResult:
    """
    Prediction result container | 预测结果容器
    
    Attributes:
        value: Predicted value | 预测值
        confidence: Confidence level (0-1) | 置信度
        method: Name of the predictor | 预测器名称
        error: Actual error (if known) | 实际误差
        metadata: Additional info | 额外信息
    """
    value: float
    confidence: float = 0.5
    method: str = "unknown"
    error: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Whether prediction was successful | 预测是否成功"""
        return self.error is None or self.error < 0.1


class ChaosModel(Generic[T]):
    """
    Abstract base for chaos system models.
    混沌系统模型抽象基类。
    
    Users can implement their own system dynamics.
    用户可以自行实现系统动力学。
    """
    
    def dynamics(self, t: float, state: T) -> T:
        """
        System dynamics: dx/dt = f(t, x)
        系统动力学方程
        
        Args:
            t: Current time | 当前时间
            state: Current state | 当前状态
            
        Returns:
            T: State derivative | 状态导数
        """
        raise NotImplementedError
    
    def get_parameters(self) -> Dict[str, float]:
        """Get model parameters | 获取模型参数"""
        return {}
    
    def get_feature_names(self) -> List[str]:
        """Get feature names | 获取特征名称"""
        return ['state_value', 'time_offset']