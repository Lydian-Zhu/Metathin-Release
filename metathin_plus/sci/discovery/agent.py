"""
Scientific Discovery Agent | 科学发现代理
===========================================

Adaptive function discovery and prediction agent.
自适应函数发现与预测代理。

Design Principles | 设计原则:
    - Independent of Metathin core | 独立于 Metathin 核心
    - State machine driven | 状态机驱动
    - Observable (full history) | 可观测（完整历史）
    - Configurable thresholds | 可配置阈值

Workflow | 工作流程:
    1. Collect data points | 收集数据点
    2. Extract function vector | 提取函数向量
    3. Match against library | 匹配库中的函数
    4. Fit parameters | 拟合参数
    5. Predict next value | 预测下一个值
    6. Validate and restart if needed | 验证，必要时重启
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
import logging

from sympy import sympify

from ..core.function_space import FunctionSpace, FunctionVector, VectorSpaceConfig
from ..core.function_library import FunctionLibrary, FunctionEntry
from ..core.laurent_expander import LaurentExpander


class DiscoveryPhase(Enum):
    """
    Discovery phase enumeration | 发现阶段枚举
    
    States:
        INIT: Initialization | 初始化
        COLLECTING: Collecting data | 收集数据
        EXTRACTING: Extracting vector | 提取向量
        MATCHING: Matching function | 匹配函数
        FITTING: Fitting parameters | 拟合参数
        PREDICTING: Predicting | 预测
        VALIDATING: Validating | 验证
        RESTARTING: Restarting | 重启
        COMPLETED: Completed | 完成
    """
    INIT = "init"
    COLLECTING = "collecting"
    EXTRACTING = "extracting"
    MATCHING = "matching"
    FITTING = "fitting"
    PREDICTING = "predicting"
    VALIDATING = "validating"
    RESTARTING = "restarting"
    COMPLETED = "completed"


@dataclass
class DiscoveryResult:
    """
    Discovery result data class | 发现结果数据类
    
    Attributes:
        phase: Current discovery phase | 当前发现阶段
        timestamp: Result timestamp | 结果时间戳
        x: X coordinate | X 坐标
        y: Y value | Y 值
        prediction: Predicted value (if any) | 预测值（如果有）
        matched_function: Name of matched function | 匹配的函数名称
        similarity: Similarity score | 相似度分数
        parameters: Fitted parameters | 拟合的参数
        error: Prediction error (if validated) | 预测误差（如果已验证）
        metadata: Additional metadata | 附加元数据
    """
    phase: DiscoveryPhase
    timestamp: float
    x: float
    y: float
    prediction: Optional[float] = None
    matched_function: Optional[str] = None
    similarity: float = 0.0
    parameters: Dict[str, float] = field(default_factory=dict)
    error: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Whether the prediction was successful | 预测是否成功"""
        return self.error is None or self.error < 0.1
    
    def to_dict(self) -> dict:
        """Convert to dictionary | 转换为字典"""
        return {
            'phase': self.phase.value,
            'timestamp': self.timestamp,
            'x': self.x,
            'y': self.y,
            'prediction': self.prediction,
            'matched_function': self.matched_function,
            'similarity': self.similarity,
            'parameters': self.parameters,
            'error': self.error,
            'metadata': self.metadata
        }


class ScientificDiscoveryAgent:
    """
    Scientific Discovery Agent | 科学发现代理
    
    Adaptive function discovery and prediction using Laurent series vector space.
    使用洛朗级数向量空间的自适应函数发现与预测。
    
    Example | 示例:
        >>> agent = ScientificDiscoveryAgent(window_size=50, error_threshold=0.1)
        >>> 
        >>> # Online learning | 在线学习
        >>> for x, y in data_stream:
        ...     result = agent.add_point(x, y)
        ...     if result.prediction is not None:
        ...         print(f"Predicted: {result.prediction:.4f}")
        ...     
        ...     # Validate with next point | 用下一个点验证
        ...     if need_validate:
        ...         agent.validate(next_x, next_y)
    """
    
    def __init__(self,
                 window_size: int = 50,
                 error_threshold: float = 0.1,
                 similarity_threshold: float = 0.9,
                 n_negative: int = 20,
                 n_positive: int = 20,
                 center: float = 0.0,
                 enable_special_functions: bool = False,
                 data_dir: Optional[str] = None):
        """
        Initialize scientific discovery agent | 初始化科学发现代理
        
        Args:
            window_size: Fitting window size | 拟合窗口大小
            error_threshold: Error threshold for restart | 重启误差阈值
            similarity_threshold: Function matching threshold | 函数匹配相似度阈值
            n_negative: Number of negative power terms | 负幂项数量
            n_positive: Number of positive power terms | 正幂项数量
            center: Expansion center point | 展开中心点
            enable_special_functions: Whether to enable special functions | 是否启用特殊函数
            data_dir: Directory for function library data | 函数库数据目录
        """
        self.window_size = window_size
        self.error_threshold = error_threshold
        self.similarity_threshold = similarity_threshold
        
        # Initialize core components | 初始化核心组件
        config = VectorSpaceConfig(n_negative=n_negative, n_positive=n_positive, center=center)
        self.space = FunctionSpace(config)
        self.expander = LaurentExpander(n_negative, n_positive, center)
        self.library = FunctionLibrary(self.space, data_dir=data_dir)
        
        # Data buffers | 数据缓冲区
        self._x_buffer: List[float] = []
        self._y_buffer: List[float] = []
        
        # Current state | 当前状态
        self._phase = DiscoveryPhase.INIT
        self._current_entry: Optional[FunctionEntry] = None
        self._current_params: Dict[str, float] = {}
        self._current_vector: Optional[FunctionVector] = None
        
        # History | 历史记录
        self._history: List[DiscoveryResult] = []
        
        # Statistics | 统计信息
        self._restart_count = 0
        self._prediction_count = 0
        self._match_count = 0
        
        self._logger = logging.getLogger("metathin_sci.ScientificDiscoveryAgent")
        self._logger.info(f"Agent initialized: window={window_size}, threshold={error_threshold}")
    
    # ============================================================
    # Properties | 属性
    # ============================================================
    
    @property
    def phase(self) -> DiscoveryPhase:
        """Current discovery phase | 当前发现阶段"""
        return self._phase
    
    @property
    def history(self) -> List[DiscoveryResult]:
        """Full discovery history | 完整发现历史"""
        return self._history.copy()
    
    @property
    def current_function(self) -> Optional[str]:
        """Currently matched function name | 当前匹配的函数名称"""
        return self._current_entry.name if self._current_entry else None
    
    @property
    def current_parameters(self) -> Dict[str, float]:
        """Current fitted parameters | 当前拟合的参数"""
        return self._current_params.copy()
    
    @property
    def buffer_size(self) -> int:
        """Current buffer size | 当前缓冲区大小"""
        return len(self._x_buffer)
    
    # ============================================================
    # Core Methods | 核心方法
    # ============================================================
    
    def _extract_vector(self) -> Optional[FunctionVector]:
        """
        Extract function vector from current buffer | 从当前缓冲区提取函数向量
        
        Returns:
            Optional[FunctionVector]: Extracted vector or None | 提取的向量或 None
        """
        if len(self._x_buffer) < self.window_size:
            return None
        
        x_vals = np.array(self._x_buffer[-self.window_size:])
        y_vals = np.array(self._y_buffer[-self.window_size:])
        
        return self.expander.expand_data(x_vals, y_vals)
    
    def _fit_parameters(self, entry: FunctionEntry,
                        x_vals: np.ndarray, y_vals: np.ndarray) -> Dict[str, float]:
        """
        Fit function parameters | 拟合函数参数
        
        Args:
            entry: Function entry | 函数条目
            x_vals: X coordinates | X 坐标
            y_vals: Y values | Y 值
            
        Returns:
            Dict[str, float]: Fitted parameters | 拟合的参数
        """
        if not entry.parameters:
            return {}
        
        try:
            from scipy.optimize import curve_fit
            from sympy import symbols, lambdify
            
            x_sym = symbols('x')
            expr = sympify(entry.expression)
            
            # Create parameter symbols | 创建参数符号
            params_sym = [symbols(p) for p in entry.parameters]
            
            # Create callable function | 创建可调用函数
            func = lambdify([x_sym] + params_sym, expr, modules='numpy')
            
            def fit_func(x, *params):
                return func(x, *params)
            
            # Initial guesses | 初始猜测
            p0 = [1.0] * len(entry.parameters)
            
            # Fit | 拟合
            popt, _ = curve_fit(fit_func, x_vals, y_vals, p0=p0, maxfev=5000)
            return {p: float(popt[i]) for i, p in enumerate(entry.parameters)}
            
        except Exception as e:
            self._logger.debug(f"Parameter fitting failed: {e}")
            return {}
    
    def _predict_next(self, x_next: float) -> float:
        """
        Predict next value using current function | 使用当前函数预测下一个值
        
        Args:
            x_next: Next X coordinate | 下一个 X 坐标
            
        Returns:
            float: Predicted value | 预测值
        """
        if self._current_entry is None:
            return self._y_buffer[-1] if self._y_buffer else 0.0
        
        try:
            from sympy import symbols, lambdify
            
            x_sym = symbols('x')
            expr = sympify(self._current_entry.expression)
            
            # Substitute parameters | 替换参数
            for name, value in self._current_params.items():
                expr = expr.subs(symbols(name), value)
            
            func = lambdify(x_sym, expr, modules='numpy')
            return float(func(x_next))
            
        except Exception as e:
            self._logger.debug(f"Prediction failed: {e}")
            return self._y_buffer[-1] if self._y_buffer else 0.0
    
    def _should_restart(self, predicted: float, actual: float) -> bool:
        """
        Check if restart is needed | 检查是否需要重启
        
        Args:
            predicted: Predicted value | 预测值
            actual: Actual value | 实际值
            
        Returns:
            bool: Whether to restart | 是否重启
        """
        error = abs(predicted - actual)
        return error > self.error_threshold
    
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
        self._x_buffer.append(x)
        self._y_buffer.append(y)
        
        # Limit buffer size | 限制缓冲区大小
        max_buffer = self.window_size * 2
        if len(self._x_buffer) > max_buffer:
            self._x_buffer = self._x_buffer[-max_buffer:]
            self._y_buffer = self._y_buffer[-max_buffer:]
        
        # Phase 1: Collecting data | 阶段1：收集数据
        if len(self._x_buffer) < self.window_size:
            self._phase = DiscoveryPhase.COLLECTING
            result = DiscoveryResult(
                phase=self._phase,
                timestamp=time.time(),
                x=x, y=y,
                prediction=y,
                matched_function=None,
                similarity=0.0
            )
            self._history.append(result)
            return result
        
        # Phase 2: Extracting vector | 阶段2：提取向量
        self._phase = DiscoveryPhase.EXTRACTING
        vector = self._extract_vector()
        
        if vector is None:
            result = DiscoveryResult(
                phase=self._phase,
                timestamp=time.time(),
                x=x, y=y,
                prediction=y
            )
            self._history.append(result)
            return result
        
        self._current_vector = vector
        
        # Phase 3: Matching function | 阶段3：匹配函数
        self._phase = DiscoveryPhase.MATCHING
        matches = self.library.match(vector, self.similarity_threshold)
        
        if not matches:
            # No match found, use persistence | 未找到匹配，使用持久性预测
            self._phase = DiscoveryPhase.PREDICTING
            prediction = self._y_buffer[-1]
            
            result = DiscoveryResult(
                phase=self._phase,
                timestamp=time.time(),
                x=x, y=y,
                prediction=prediction,
                matched_function=None,
                similarity=0.0
            )
            self._history.append(result)
            self._prediction_count += 1
            return result
        
        best_match, similarity = matches[0]
        self._current_entry = self.library.get(best_match)
        self._match_count += 1
        
        self._logger.debug(f"Matched: {best_match} (sim={similarity:.4f})")
        
        # Phase 4: Fitting parameters | 阶段4：拟合参数
        self._phase = DiscoveryPhase.FITTING
        x_vals = np.array(self._x_buffer[-self.window_size:])
        y_vals = np.array(self._y_buffer[-self.window_size:])
        self._current_params = self._fit_parameters(self._current_entry, x_vals, y_vals)
        
        # Phase 5: Predicting | 阶段5：预测
        self._phase = DiscoveryPhase.PREDICTING
        next_x = x + (x - self._x_buffer[-2]) if len(self._x_buffer) >= 2 else x + 1.0
        prediction = self._predict_next(next_x)
        
        result = DiscoveryResult(
            phase=self._phase,
            timestamp=time.time(),
            x=x, y=y,
            prediction=prediction,
            matched_function=best_match,
            similarity=similarity,
            parameters=self._current_params.copy(),
            metadata={
                'buffer_size': len(self._x_buffer),
                'restart_count': self._restart_count,
                'match_count': self._match_count
            }
        )
        
        self._history.append(result)
        self._prediction_count += 1
        
        return result
    
    def validate(self, x_actual: float, y_actual: float) -> bool:
        """
        Validate prediction and check for restart | 验证预测并检查是否需要重启
        
        Args:
            x_actual: Actual X coordinate | 实际 X 坐标
            y_actual: Actual Y value | 实际 Y 值
            
        Returns:
            bool: Whether restart was triggered | 是否触发了重启
        """
        if not self._history:
            return False
        
        last = self._history[-1]
        if last.prediction is None:
            return False
        
        error = abs(last.prediction - y_actual)
        last.error = error
        
        if self._should_restart(last.prediction, y_actual):
            self._logger.info(f"Error {error:.4f} > threshold {self.error_threshold}, restarting")
            self._phase = DiscoveryPhase.RESTARTING
            self._restart_count += 1
            
            # Reset buffers | 重置缓冲区
            self._x_buffer = []
            self._y_buffer = []
            self._current_entry = None
            self._current_params = {}
            self._current_vector = None
            
            return True
        
        self._phase = DiscoveryPhase.VALIDATING
        return False
    
    def predict(self, x_next: float) -> float:
        """
        Predict next value without updating state | 预测下一个值（不更新状态）
        
        Args:
            x_next: Next X coordinate | 下一个 X 坐标
            
        Returns:
            float: Predicted value | 预测值
        """
        return self._predict_next(x_next)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status | 获取代理状态
        
        Returns:
            Dict: Status information | 状态信息
        """
        last = self._history[-1] if self._history else None
        
        return {
            'phase': self._phase.value,
            'current_function': self.current_function,
            'similarity': last.similarity if last else 0.0,
            'restart_count': self._restart_count,
            'prediction_count': self._prediction_count,
            'match_count': self._match_count,
            'buffer_size': len(self._x_buffer),
            'window_size': self.window_size,
            'error_threshold': self.error_threshold,
            'similarity_threshold': self.similarity_threshold,
            'library_size': len(self.library)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics | 获取详细统计信息
        
        Returns:
            Dict: Statistics | 统计信息
        """
        if not self._history:
            return {'total_points': 0}
        
        errors = [r.error for r in self._history if r.error is not None]
        
        return {
            'total_points': len(self._history),
            'predictions': self._prediction_count,
            'restarts': self._restart_count,
            'matches': self._match_count,
            'avg_error': float(np.mean(errors)) if errors else None,
            'max_error': float(np.max(errors)) if errors else None,
            'library_stats': self.library.get_statistics()
        }
    
    def register_function(self, name: str, expr, **kwargs) -> bool:
        """
        Register a custom function | 注册自定义函数
        
        Args:
            name: Function name | 函数名称
            expr: Function expression | 函数表达式
            **kwargs: Additional arguments for library registration | 额外参数
            
        Returns:
            bool: Success status | 成功状态
        """
        return self.library.register(name, expr, **kwargs)
    
    def get_matched_function(self) -> Optional[FunctionEntry]:
        """Get currently matched function entry | 获取当前匹配的函数条目"""
        return self._current_entry
    
    def get_recent_history(self, n: int = 10) -> List[DiscoveryResult]:
        """Get n most recent results | 获取最近 n 个结果"""
        return self._history[-n:] if self._history else []
    
    def clear_history(self):
        """Clear discovery history | 清空发现历史"""
        self._history = []
        self._logger.info("History cleared")
    
    def reset(self):
        """Reset agent to initial state | 重置代理到初始状态"""
        self._x_buffer = []
        self._y_buffer = []
        self._phase = DiscoveryPhase.INIT
        self._current_entry = None
        self._current_params = {}
        self._current_vector = None
        self._history = []
        self._restart_count = 0
        self._prediction_count = 0
        self._match_count = 0
        self._logger.info("Agent reset")
    
    def save_library(self) -> bool:
        """Save user function library | 保存用户函数库"""
        return self.library._save_user_functions()
    
    def load_library(self) -> bool:
        """Reload function library | 重新加载函数库"""
        return self.library.reload()