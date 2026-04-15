# metathin_plus/chaos/pattern_space.py
"""
Chaos Pattern Space (P) | 混沌模式空间 (P)
===========================================

Converts system state to feature vectors for Metathin processing.
将系统状态转换为特征向量供 Metathin 处理。
"""

import numpy as np
from typing import Any, List, Optional
from collections import deque

from metathin.core.p_pattern import PatternSpace
from metathin.core.types import FeatureVector

from .base import SystemState, T


class ChaosPatternSpace(PatternSpace):
    """
    Chaos Pattern Space - P component for chaos prediction.
    混沌模式空间 - 混沌预测的感知层组件。
    
    Extracts features from time series of system states.
    从系统状态的时间序列中提取特征。
    
    Features extracted:
        - Current value | 当前值
        - Velocity (first difference) | 速度（一阶差分）
        - Acceleration (second difference) | 加速度（二阶差分）
        - Recent mean (window) | 近期均值
        - Recent variance | 近期方差
        - Recent min/max | 近期最小/最大值
    """
    
    def __init__(
        self,
        window_size: int = 10,
        include_velocity: bool = True,
        include_acceleration: bool = True,
        include_stats: bool = True,
        name: str = "ChaosPattern"
    ):
        """
        Initialize chaos pattern space.
        初始化混沌模式空间。
        
        Args:
            window_size: Sliding window for statistics | 统计滑动窗口
            include_velocity: Include first difference | 包含一阶差分
            include_acceleration: Include second difference | 包含二阶差分
            include_stats: Include statistical features | 包含统计特征
            name: Pattern space name | 模式空间名称
        """
        super().__init__()
        
        self.window_size = window_size
        self.include_velocity = include_velocity
        self.include_acceleration = include_acceleration
        self.include_stats = include_stats
        self._name = name
        
        # History buffer | 历史缓冲区
        self._history: List[float] = []
        self._timestamps: List[float] = []
        
        self._logger.debug(f"ChaosPatternSpace initialized: window={window_size}")
    
    @property
    def name(self) -> str:
        return self._name
    
    def extract(self, raw_input: Any) -> FeatureVector:
        """
        Extract feature vector from raw input.
        从原始输入提取特征向量。
        
        Args:
            raw_input: SystemState or numeric value | 系统状态或数值
            
        Returns:
            FeatureVector: Feature vector | 特征向量
        """
        # Convert to SystemState if needed | 转换为 SystemState
        if not isinstance(raw_input, SystemState):
            state = SystemState(data=raw_input)
        else:
            state = raw_input
        
        # Get current value | 获取当前值
        current = state.get_value()
        
        # Update history | 更新历史
        self._history.append(current)
        self._timestamps.append(state.timestamp)
        
        # Limit history size | 限制历史大小
        max_history = self.window_size * 3
        if len(self._history) > max_history:
            self._history = self._history[-max_history:]
            self._timestamps = self._timestamps[-max_history:]
        
        # Build feature vector | 构建特征向量
        features = []
        
        # 1. Current value | 当前值
        features.append(current)
        
        # 2. Velocity (if enough history) | 速度
        if self.include_velocity and len(self._history) >= 2:
            dt = self._timestamps[-1] - self._timestamps[-2]
            if dt > 0:
                velocity = (self._history[-1] - self._history[-2]) / dt
            else:
                velocity = self._history[-1] - self._history[-2]
            features.append(np.clip(velocity, -10, 10))
        else:
            features.append(0.0)
        
        # 3. Acceleration (if enough history) | 加速度
        if self.include_acceleration and len(self._history) >= 3:
            dt1 = self._timestamps[-1] - self._timestamps[-2]
            dt2 = self._timestamps[-2] - self._timestamps[-3]
            if dt1 > 0 and dt2 > 0:
                v1 = (self._history[-1] - self._history[-2]) / dt1
                v2 = (self._history[-2] - self._history[-3]) / dt2
                acceleration = (v1 - v2) / ((dt1 + dt2) / 2)
            else:
                acceleration = 0.0
            features.append(np.clip(acceleration, -10, 10))
        else:
            features.append(0.0)
        
        # 4. Statistical features | 统计特征
        if self.include_stats and len(self._history) >= self.window_size:
            recent = self._history[-self.window_size:]
            features.append(np.mean(recent))
            features.append(np.std(recent))
            features.append(np.min(recent))
            features.append(np.max(recent))
        else:
            features.extend([current, 0.0, current, current])
        
        return np.array(features, dtype=np.float64)
    
    def reset(self) -> None:
        """Reset history buffer | 重置历史缓冲区"""
        self._history.clear()
        self._timestamps.clear()
        self._logger.debug("ChaosPatternSpace reset")
    
    def get_feature_names(self) -> List[str]:
        """Get feature names | 获取特征名称"""
        names = [f"{self._name}_current"]
        
        if self.include_velocity:
            names.append(f"{self._name}_velocity")
        
        if self.include_acceleration:
            names.append(f"{self._name}_acceleration")
        
        if self.include_stats:
            names.extend([
                f"{self._name}_mean",
                f"{self._name}_std",
                f"{self._name}_min",
                f"{self._name}_max"
            ])
        
        return names
    
    def get_feature_dimension(self) -> int:
        """Get feature dimension | 获取特征维度"""
        dim = 1  # current
        if self.include_velocity:
            dim += 1
        if self.include_acceleration:
            dim += 1
        if self.include_stats:
            dim += 4
        return dim