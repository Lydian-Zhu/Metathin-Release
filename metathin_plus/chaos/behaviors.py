# metathin_plus/chaos/behaviors.py
"""
Chaos Prediction Behaviors (B) | 混沌预测行为 (B)
=================================================

Each predictor is a MetaBehavior that can be selected by the decision strategy.
每个预测器都是一个元行为，可由决策策略选择。
"""

import numpy as np
from typing import Any, Dict, List, Optional
from collections import deque
import logging

from metathin.core.b_behavior import MetaBehavior
from metathin.core.types import FeatureVector

from .base import SystemState, PredictionResult, T


class BaseChaosBehavior(MetaBehavior):
    """
    Base class for chaos prediction behaviors.
    混沌预测行为的基类。
    
    Maintains prediction history and error tracking.
    维护预测历史和误差跟踪。
    """
    
    def __init__(self, name: str, memory_size: int = 1000):
        """
        Initialize base behavior.
        初始化基类行为。
        
        Args:
            name: Behavior name | 行为名称
            memory_size: History memory size | 历史记忆大小
        """
        super().__init__()
        self._name = name
        self._memory_size = memory_size
        
        # History | 历史记录
        self._predictions: List[PredictionResult] = []
        self._errors: List[float] = []
        self._value_history: deque = deque(maxlen=memory_size)
        self._timestamp_history: deque = deque(maxlen=memory_size)
        
        self._logger = logging.getLogger(f"metathin_plus.chaos.behavior.{name}")
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def last_prediction(self) -> Optional[PredictionResult]:
        """Get last prediction | 获取最后一次预测"""
        return self._predictions[-1] if self._predictions else None
    
    @property
    def recent_error(self) -> float:
        """Get recent average error | 获取近期平均误差"""
        if not self._errors:
            return 1.0
        window = min(10, len(self._errors))
        return float(np.mean(self._errors[-window:]))
    
    def update_actual(self, actual_value: float) -> float:
        """
        Update with actual value and compute error.
        用实际值更新并计算误差。
        
        Args:
            actual_value: Actual observed value | 实际观测值
            
        Returns:
            float: Prediction error | 预测误差
        """
        if not self._predictions:
            return 0.0
        
        last = self._predictions[-1]
        error = abs(last.value - actual_value)
        last.error = error
        self._errors.append(error)
        
        # Limit error history | 限制误差历史
        if len(self._errors) > 1000:
            self._errors = self._errors[-1000:]
        
        return error
    
    def reset(self) -> None:
        """Reset behavior state | 重置行为状态"""
        self._predictions.clear()
        self._errors.clear()
        self._value_history.clear()
        self._timestamp_history.clear()
        self._execution_count = 0
        self._total_execution_time = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get behavior statistics | 获取行为统计"""
        return {
            'name': self._name,
            'execution_count': self._execution_count,
            'total_predictions': len(self._predictions),
            'recent_error': self.recent_error,
            'avg_error': float(np.mean(self._errors)) if self._errors else None,
        }


class PersistentBehavior(BaseChaosBehavior):
    """
    Persistent prediction: next value equals current value.
    持久性预测：下一个值等于当前值。
    
    Baseline method for comparison.
    用于比较的基准方法。
    """
    
    def __init__(self, name: str = "Persistent"):
        super().__init__(name)
    
    def execute(self, features: FeatureVector, **kwargs) -> Any:
        """
        Execute persistent prediction.
        执行持久性预测。
        
        Args:
            features: Feature vector | 特征向量
            **kwargs: May contain 'state' or 'current_value' | 可包含状态或当前值
            
        Returns:
            PredictionResult: Prediction result | 预测结果
        """
        self._execution_count += 1
        import time
        start = time.time()
        
        # Get current value from kwargs or features | 从参数或特征获取当前值
        current_value = kwargs.get('current_value')
        if current_value is None:
            state = kwargs.get('state')
            if isinstance(state, SystemState):
                current_value = state.get_value()
            else:
                current_value = float(features[0]) if len(features) > 0 else 0.0
        
        # Store history | 存储历史
        self._value_history.append(current_value)
        
        # Make prediction | 进行预测
        result = PredictionResult(
            value=current_value,
            confidence=0.5,
            method=self._name
        )
        
        self._predictions.append(result)
        self._last_execution_time = time.time() - start
        self._total_execution_time += self._last_execution_time
        
        return result


class LinearTrendBehavior(BaseChaosBehavior):
    """
    Linear trend prediction using recent points.
    使用近期点的线性趋势预测。
    
    Fits a line to recent values and extrapolates.
    拟合近期值的直线并外推。
    """
    
    def __init__(self, window: int = 5, name: str = "LinearTrend"):
        super().__init__(name)
        self.window = window
    
    def execute(self, features: FeatureVector, **kwargs) -> Any:
        """Execute linear trend prediction | 执行线性趋势预测"""
        self._execution_count += 1
        import time
        start = time.time()
        
        # Get current value | 获取当前值
        current_value = kwargs.get('current_value')
        if current_value is None:
            state = kwargs.get('state')
            if isinstance(state, SystemState):
                current_value = state.get_value()
            else:
                current_value = float(features[0]) if len(features) > 0 else 0.0
        
        self._value_history.append(current_value)
        
        # Need enough history | 需要足够的历史
        if len(self._value_history) < self.window:
            prediction = current_value
            confidence = 0.3
        else:
            values = list(self._value_history)[-self.window:]
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            prediction = coeffs[0] * self.window + coeffs[1]
            confidence = 0.4 + 0.3 * abs(coeffs[0]) / (1 + abs(coeffs[0]))
        
        result = PredictionResult(
            value=float(prediction),
            confidence=float(np.clip(confidence, 0.2, 0.8)),
            method=self._name
        )
        
        self._predictions.append(result)
        self._last_execution_time = time.time() - start
        self._total_execution_time += self._last_execution_time
        
        return result


class PhaseSpaceBehavior(BaseChaosBehavior):
    """
    Phase space reconstruction prediction.
    相空间重构预测。
    
    Uses Takens' embedding theorem to reconstruct attractor.
    使用 Takens 嵌入定理重构吸引子。
    """
    
    def __init__(
        self,
        embed_dim: int = 5,
        delay: int = 3,
        k_neighbors: int = 5,
        min_history: int = 100,
        name: str = "PhaseSpace"
    ):
        super().__init__(name)
        self.embed_dim = embed_dim
        self.delay = delay
        self.k_neighbors = k_neighbors
        self.min_history = min_history
        
        # Storage for phase space | 相空间存储
        self._vectors: List[np.ndarray] = []
        self._targets: List[float] = []
    
    def _build_phase_space(self, values: List[float]) -> None:
        """Build phase space vectors | 构建相空间向量"""
        n = len(values)
        min_required = self.embed_dim * self.delay + self.k_neighbors
        
        if n < min_required:
            self._vectors.clear()
            self._targets.clear()
            return
        
        vectors = []
        targets = []
        
        for i in range(self.embed_dim * self.delay, n - 1):
            vector = []
            for j in range(self.embed_dim):
                idx = i - j * self.delay
                if idx >= 0:
                    vector.append(values[idx])
            if len(vector) == self.embed_dim:
                vectors.append(np.array(vector))
                targets.append(values[i + 1])
        
        self._vectors = vectors
        self._targets = targets
    
    def _find_neighbors(self, query: np.ndarray) -> List[tuple]:
        """Find nearest neighbors | 查找最近邻"""
        if not self._vectors:
            return []
        
        distances = []
        for i, vec in enumerate(self._vectors):
            dist = np.linalg.norm(vec - query)
            distances.append((dist, i))
        
        distances.sort(key=lambda x: x[0])
        return distances[:self.k_neighbors]
    
    def execute(self, features: FeatureVector, **kwargs) -> Any:
        """Execute phase space prediction | 执行相空间预测"""
        self._execution_count += 1
        import time
        start = time.time()
        
        # Get current value | 获取当前值
        current_value = kwargs.get('current_value')
        if current_value is None:
            state = kwargs.get('state')
            if isinstance(state, SystemState):
                current_value = state.get_value()
            else:
                current_value = float(features[0]) if len(features) > 0 else 0.0
        
        self._value_history.append(current_value)
        values = list(self._value_history)
        
        # Need enough history | 需要足够的历史
        if len(values) < self.min_history:
            result = PredictionResult(
                value=current_value,
                confidence=0.3,
                method=self._name
            )
        else:
            # Build phase space | 构建相空间
            self._build_phase_space(values)
            
            # Build current query vector | 构建当前查询向量
            query = []
            for j in range(self.embed_dim):
                idx = -1 - j * self.delay
                if abs(idx) <= len(values):
                    query.append(values[idx])
                else:
                    query.append(0.0)
            query = np.array(query)
            
            # Find neighbors and predict | 查找邻居并预测
            neighbors = self._find_neighbors(query)
            if not neighbors:
                prediction = current_value
                confidence = 0.3
            else:
                weights = [1.0 / (d + 1e-8) for d, _ in neighbors]
                weights = np.array(weights) / (np.sum(weights) + 1e-8)
                pred_values = [self._targets[idx] for _, idx in neighbors]
                prediction = np.sum(np.array(pred_values) * weights)
                confidence = 0.4 + 0.3 * (1.0 / (1.0 + neighbors[0][0]))
            
            result = PredictionResult(
                value=float(prediction),
                confidence=float(np.clip(confidence, 0.2, 0.9)),
                method=self._name
            )
        
        self._predictions.append(result)
        self._last_execution_time = time.time() - start
        self._total_execution_time += self._last_execution_time
        
        return result


class VolterraBehavior(BaseChaosBehavior):
    """
    Volterra series prediction.
    Volterra 级数预测。
    
    Uses polynomial expansion of past values.
    使用过去值的多项式展开。
    """
    
    def __init__(
        self,
        order: int = 2,
        memory: int = 10,
        regularization: float = 0.01,
        name: str = "Volterra"
    ):
        super().__init__(name)
        self.order = min(order, 2)  # Limit to 2nd order for speed | 限制为二阶以保证速度
        self.memory = memory
        self.regularization = regularization
        
        # Linear model coefficients | 线性模型系数
        self._coeffs: Optional[np.ndarray] = None
        self._is_trained = False
        self._feature_dim = self._compute_feature_dim()
    
    def _compute_feature_dim(self) -> int:
        """Compute feature dimension | 计算特征维度"""
        dim = 1  # bias
        dim += self.memory  # linear terms
        
        if self.order >= 2:
            dim += self.memory * (self.memory + 1) // 2  # quadratic terms
        
        return dim
    
    def _build_features(self, history: List[float]) -> np.ndarray:
        """Build feature vector from history | 从历史构建特征向量"""
        features = [1.0]  # bias
        
        # Linear terms | 线性项
        features.extend(history[-self.memory:])
        
        # Quadratic terms | 二次项
        if self.order >= 2:
            n = len(history[-self.memory:])
            for i in range(n):
                for j in range(i, n):
                    features.append(history[-self.memory:][i] * history[-self.memory:][j])
        
        return np.array(features)
    
    def _train(self, values: List[float]) -> None:
        """Train Volterra model | 训练 Volterra 模型"""
        n = len(values)
        min_samples = self._feature_dim * 2
        
        if n < self.memory + min_samples:
            return
        
        X_list = []
        y_list = []
        
        for i in range(self.memory, n - 1):
            history = values[i - self.memory:i]
            features = self._build_features(history)
            X_list.append(features)
            y_list.append(values[i])
        
        if len(X_list) < self._feature_dim:
            return
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Ridge regression | 岭回归
        XTX = X.T @ X
        XTy = X.T @ y
        reg = self.regularization * np.eye(self._feature_dim)
        
        try:
            self._coeffs = np.linalg.solve(XTX + reg, XTy)
            self._is_trained = True
        except np.linalg.LinAlgError:
            self._coeffs = np.linalg.pinv(XTX + reg) @ XTy
            self._is_trained = True
    
    def execute(self, features: FeatureVector, **kwargs) -> Any:
        """Execute Volterra prediction | 执行 Volterra 预测"""
        self._execution_count += 1
        import time
        start = time.time()
        
        # Get current value | 获取当前值
        current_value = kwargs.get('current_value')
        if current_value is None:
            state = kwargs.get('state')
            if isinstance(state, SystemState):
                current_value = state.get_value()
            else:
                current_value = float(features[0]) if len(features) > 0 else 0.0
        
        self._value_history.append(current_value)
        values = list(self._value_history)
        
        # Train if needed | 需要时训练
        if not self._is_trained and len(values) > self.memory + 50:
            self._train(values)
        
        # Predict | 预测
        if self._is_trained and len(values) >= self.memory:
            history = values[-self.memory:]
            feat = self._build_features(history)
            prediction = np.dot(feat, self._coeffs)
            confidence = 0.5 + 0.3 * min(1.0, len(values) / 500)
        else:
            # Fallback to persistence | 回退到持久性预测
            prediction = current_value
            confidence = 0.3
        
        result = PredictionResult(
            value=float(prediction),
            confidence=float(np.clip(confidence, 0.2, 0.9)),
            method=self._name
        )
        
        self._predictions.append(result)
        self._last_execution_time = time.time() - start
        self._total_execution_time += self._last_execution_time
        
        return result


class NeuralBehavior(BaseChaosBehavior):
    """
    Neural network prediction (simplified RNN).
    神经网络预测（简化 RNN）。
    
    Uses a simple recurrent neural network structure.
    使用简单的循环神经网络结构。
    """
    
    def __init__(
        self,
        hidden_size: int = 32,
        memory: int = 10,
        learning_rate: float = 0.01,
        name: str = "Neural"
    ):
        super().__init__(name)
        self.hidden_size = hidden_size
        self.memory = memory
        self.learning_rate = learning_rate
        
        # Simple weights | 简单权重
        self._input_weights: Optional[np.ndarray] = None
        self._hidden_weights: Optional[np.ndarray] = None
        self._output_weights: Optional[np.ndarray] = None
        self._is_trained = False
        self._hidden_state = np.zeros(hidden_size)
    
    def _init_weights(self, input_dim: int) -> None:
        """Initialize weights | 初始化权重"""
        self._input_weights = np.random.randn(self.hidden_size, input_dim) * 0.1
        self._hidden_weights = np.random.randn(self.hidden_size, self.hidden_size) * 0.1
        self._output_weights = np.random.randn(1, self.hidden_size) * 0.1
    
    def _forward(self, x: np.ndarray) -> float:
        """Forward pass | 前向传播"""
        if self._input_weights is None:
            return 0.0
        
        # Update hidden state | 更新隐藏状态
        input_part = self._input_weights @ x
        hidden_part = self._hidden_weights @ self._hidden_state
        self._hidden_state = np.tanh(input_part + hidden_part)
        
        # Output | 输出
        output = float((self._output_weights @ self._hidden_state)[0])
        return output
    
    def _train_step(self, x: np.ndarray, target: float) -> float:
        """Single training step | 单次训练步骤"""
        if self._input_weights is None:
            self._init_weights(len(x))
        
        # Forward | 前向
        prediction = self._forward(x)
        error = target - prediction
        
        # Simple gradient approximation | 简单梯度近似
        learning = self.learning_rate * error
        
        # Update weights (simplified) | 更新权重（简化）
        self._output_weights += learning * self._hidden_state.reshape(1, -1)
        
        return abs(error)
    
    def execute(self, features: FeatureVector, **kwargs) -> Any:
        """Execute neural prediction | 执行神经网络预测"""
        self._execution_count += 1
        import time
        start = time.time()
        
        # Get current value | 获取当前值
        current_value = kwargs.get('current_value')
        if current_value is None:
            state = kwargs.get('state')
            if isinstance(state, SystemState):
                current_value = state.get_value()
            else:
                current_value = float(features[0]) if len(features) > 0 else 0.0
        
        self._value_history.append(current_value)
        values = list(self._value_history)
        
        # Need enough history | 需要足够的历史
        if len(values) < self.memory + 10:
            result = PredictionResult(
                value=current_value,
                confidence=0.3,
                method=self._name
            )
        else:
            # Build input vector | 构建输入向量
            input_vec = np.array(values[-self.memory:])
            
            # Initialize weights if needed | 需要时初始化权重
            if self._input_weights is None:
                self._init_weights(len(input_vec))
            
            # Reset hidden state for each prediction | 每次预测重置隐藏状态
            self._hidden_state = np.zeros(self.hidden_size)
            
            # Predict | 预测
            prediction = self._forward(input_vec)
            
            # Train if we have a target | 如果有目标则训练
            target = kwargs.get('target')
            if target is not None:
                error = self._train_step(input_vec, target)
                self._errors.append(error)
            
            confidence = 0.4 + 0.3 * min(1.0, len(values) / 500)
            
            result = PredictionResult(
                value=float(prediction),
                confidence=float(np.clip(confidence, 0.2, 0.9)),
                method=self._name
            )
        
        self._predictions.append(result)
        self._last_execution_time = time.time() - start
        self._total_execution_time += self._last_execution_time
        
        return result


class SpectralBehavior(BaseChaosBehavior):
    """
    Spectral analysis prediction using FFT.
    使用 FFT 的频谱分析预测。
    
    Detects dominant frequencies and extrapolates.
    检测主频并外推。
    """
    
    def __init__(
        self,
        n_frequencies: int = 3,
        window: int = 100,
        name: str = "Spectral"
    ):
        super().__init__(name)
        self.n_frequencies = n_frequencies
        self.window = window
    
    def _extract_frequencies(self, values: List[float]) -> List[tuple]:
        """Extract dominant frequencies | 提取主频"""
        n = len(values)
        if n < 10:
            return []
        
        # Detrend | 去趋势
        x = np.arange(n)
        coeffs = np.polyfit(x, values, 1)
        trend = coeffs[0] * x + coeffs[1]
        detrended = np.array(values) - trend
        
        # FFT | 快速傅里叶变换
        fft_vals = np.fft.fft(detrended)
        fft_mag = np.abs(fft_vals[:n//2])
        freqs = np.fft.fftfreq(n, d=1.0)[:n//2]
        
        # Find peaks | 找峰值
        peaks = []
        for i in range(1, len(fft_mag) - 1):
            if fft_mag[i] > fft_mag[i-1] and fft_mag[i] > fft_mag[i+1]:
                peaks.append((freqs[i], fft_mag[i]))
        
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks[:self.n_frequencies]
    
    def execute(self, features: FeatureVector, **kwargs) -> Any:
        """Execute spectral prediction | 执行频谱预测"""
        self._execution_count += 1
        import time
        start = time.time()
        
        # Get current value | 获取当前值
        current_value = kwargs.get('current_value')
        if current_value is None:
            state = kwargs.get('state')
            if isinstance(state, SystemState):
                current_value = state.get_value()
            else:
                current_value = float(features[0]) if len(features) > 0 else 0.0
        
        self._value_history.append(current_value)
        values = list(self._value_history)
        
        # Need enough history | 需要足够的历史
        if len(values) < self.window:
            result = PredictionResult(
                value=current_value,
                confidence=0.3,
                method=self._name
            )
        else:
            recent = values[-self.window:]
            freqs = self._extract_frequencies(recent)
            
            if not freqs:
                prediction = current_value
                confidence = 0.3
            else:
                # Simple sinusoidal extrapolation | 简单正弦外推
                n = len(recent)
                prediction = current_value
                for f, amp in freqs:
                    prediction += amp * np.sin(2 * np.pi * f * n)
                
                # Add trend | 加回趋势
                x = np.arange(len(recent))
                coeffs = np.polyfit(x, recent, 1)
                trend_extrap = coeffs[0] * n + coeffs[1]
                prediction += trend_extrap
                
                confidence = 0.4 + 0.2 * min(1.0, len(freqs) / 3)
            
            result = PredictionResult(
                value=float(prediction),
                confidence=float(np.clip(confidence, 0.2, 0.8)),
                method=self._name
            )
        
        self._predictions.append(result)
        self._last_execution_time = time.time() - start
        self._total_execution_time += self._last_execution_time
        
        return result


# Aliases for backward compatibility | 向后兼容别名
PhaseSpacePredictor = PhaseSpaceBehavior
VolterraPredictor = VolterraBehavior
NeuralPredictor = NeuralBehavior
SpectralPredictor = SpectralBehavior
PersistentPredictor = PersistentBehavior
LinearTrendPredictor = LinearTrendBehavior