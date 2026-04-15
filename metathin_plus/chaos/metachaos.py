# metathin_plus/chaos/metachaos.py
"""
MetaChaos - Main Agent for Chaos Prediction
============================================

Main agent that assembles all five components (P, B, S, D, Ψ)
for chaos time series prediction.

MetaChaos - 混沌预测主智能体
组装所有五个 Metathin 组件用于混沌时间序列预测。
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union, Type
import time
import logging

from metathin import Metathin, MetathinBuilder
from metathin.core.types import FeatureVector
from metathin.core.exceptions import NoBehaviorError

from .base import SystemState, PredictionResult, ChaosModel, T
from .pattern_space import ChaosPatternSpace
from .behaviors import (
    BaseChaosBehavior,
    PersistentBehavior,
    LinearTrendBehavior,
    PhaseSpaceBehavior,
    VolterraBehavior,
    NeuralBehavior,
    SpectralBehavior,
)
from .selector import ChaosSelector
from .decision import MinErrorStrategy, WeightedVoteStrategy, AdaptiveStrategy
from .learning import ErrorLearning, RewardLearning


# Behavior registry | 行为注册表
BEHAVIOR_REGISTRY = {
    'persistent': PersistentBehavior,
    'linear_trend': LinearTrendBehavior,
    'phase_space': PhaseSpaceBehavior,
    'volterra': VolterraBehavior,
    'neural': NeuralBehavior,
    'spectral': SpectralBehavior,
}


class MetaChaos:
    """
    MetaChaos - Main chaos prediction agent.
    MetaChaos - 混沌预测主智能体。
    
    Assembles all five Metathin components for chaos prediction.
    组装所有五个 Metathin 组件用于混沌预测。
    
    Components | 组件:
        - P (PatternSpace): ChaosPatternSpace | 混沌模式空间
        - B (Behaviors): List of prediction behaviors | 预测行为列表
        - S (Selector): ChaosSelector | 混沌选择器
        - D (Decision): Decision strategy | 决策策略
        - Ψ (Learning): Learning mechanism | 学习机制
    
    Example | 示例:
        >>> agent = MetaChaos()
        >>> 
        >>> # Online prediction | 在线预测
        >>> for value in data_stream:
        ...     prediction = agent.predict(value)
        ...     agent.update(value)  # Update with actual value
        ...     print(f"Predicted: {prediction.value:.4f}")
    """
    
    def __init__(
        self,
        name: str = "MetaChaos",
        window_size: int = 10,
        include_velocity: bool = True,
        include_acceleration: bool = True,
        include_stats: bool = True,
        behaviors: Optional[List[str]] = None,
        decision_strategy: str = "min_error",
        decision_params: Optional[Dict[str, Any]] = None,
        selector_error_window: int = 10,
        enable_learning: bool = True,
        learning_type: str = "error",
        learning_rate: float = 0.1,
        error_threshold: float = 0.5,
        memory_size: int = 1000,
        verbose: bool = False,
    ):
        """
        Initialize MetaChaos agent.
        初始化 MetaChaos 智能体。
        
        Args:
            name: Agent name | 智能体名称
            window_size: Feature extraction window | 特征提取窗口
            include_velocity: Include velocity in features | 特征中包含速度
            include_acceleration: Include acceleration in features | 特征中包含加速度
            include_stats: Include statistical features | 包含统计特征
            behaviors: List of behavior names to use (None = all) | 使用的行为列表
            decision_strategy: Strategy type: 'min_error', 'weighted_vote', 'adaptive'
                               决策策略类型
            decision_params: Additional parameters for decision strategy | 决策策略参数
            selector_error_window: Error window for selector | 选择器误差窗口
            enable_learning: Enable learning mechanism | 启用学习机制
            learning_type: Learning type: 'error', 'reward' | 学习类型
            learning_rate: Learning rate | 学习率
            error_threshold: Error threshold for learning | 学习误差阈值
            memory_size: History memory size | 历史记忆大小
            verbose: Enable verbose logging | 启用详细日志
        """
        self.name = name
        self.verbose = verbose
        self.enable_learning = enable_learning
        
        # Setup logging | 设置日志
        self._logger = logging.getLogger(f"metathin_plus.chaos.{name}")
        if verbose:
            self._logger.setLevel(logging.DEBUG)
        
        # ============================================================
        # 1. Pattern Space (P) | 感知层
        # ============================================================
        self.pattern_space = ChaosPatternSpace(
            window_size=window_size,
            include_velocity=include_velocity,
            include_acceleration=include_acceleration,
            include_stats=include_stats,
            name=f"{name}_pattern"
        )
        
        # ============================================================
        # 2. Behaviors (B) | 行动层
        # ============================================================
        self.behaviors: List[BaseChaosBehavior] = []
        self._behavior_dict: Dict[str, BaseChaosBehavior] = {}
        self._init_behaviors(behaviors, memory_size)
        
        # ============================================================
        # 3. Selector (S) | 评估层
        # ============================================================
        self.selector = ChaosSelector(
            error_window=selector_error_window,
            use_confidence=True,
            default_fitness=0.5
        )
        
        # ============================================================
        # 4. Decision Strategy (D) | 决策层
        # ============================================================
        self.decision_strategy = self._create_decision_strategy(
            decision_strategy, decision_params
        )
        
        # ============================================================
        # 5. Learning Mechanism (Ψ) | 学习层
        # ============================================================
        self.learning_mechanism = self._create_learning_mechanism(
            learning_type, learning_rate, error_threshold
        ) if enable_learning else None
        
        # ============================================================
        # State Management | 状态管理
        # ============================================================
        self._history: List[float] = []
        self._timestamps: List[float] = []
        self._predictions: List[PredictionResult] = []
        self._selected_behavior_history: List[str] = []
        
        # Statistics | 统计信息
        self._total_predictions = 0
        self._restart_count = 0
        self._start_time = time.time()
        
        self._logger.info(f"MetaChaos '{name}' initialized")
        self._logger.info(f"  Behaviors: {[b.name for b in self.behaviors]}")
        self._logger.info(f"  Decision: {decision_strategy}")
        self._logger.info(f"  Learning: {learning_type if enable_learning else 'disabled'}")
    
    def _init_behaviors(self, behavior_names: Optional[List[str]], memory_size: int):
        """
        Initialize prediction behaviors.
        初始化预测行为。
        
        Args:
            behavior_names: List of behavior names, None for all | 行为名称列表
            memory_size: Memory size for each behavior | 每个行为的内存大小
        """
        if behavior_names is None:
            # Use all behaviors | 使用所有行为
            behavior_names = list(BEHAVIOR_REGISTRY.keys())
        
        for name in behavior_names:
            if name not in BEHAVIOR_REGISTRY:
                self._logger.warning(f"Unknown behavior: {name}, skipping")
                continue
            
            behavior_class = BEHAVIOR_REGISTRY[name]
            
            # Create behavior with appropriate parameters | 使用适当参数创建行为
            if name == 'phase_space':
                behavior = behavior_class(
                    embed_dim=5, delay=3, k_neighbors=5,
                    min_history=100, name=f"{self.name}_{name}"
                )
            elif name == 'volterra':
                behavior = behavior_class(
                    order=2, memory=10, regularization=0.01,
                    name=f"{self.name}_{name}"
                )
            elif name == 'neural':
                behavior = behavior_class(
                    hidden_size=32, memory=10, learning_rate=0.01,
                    name=f"{self.name}_{name}"
                )
            elif name == 'spectral':
                behavior = behavior_class(
                    n_frequencies=3, window=100,
                    name=f"{self.name}_{name}"
                )
            elif name == 'linear_trend':
                behavior = behavior_class(window=5, name=f"{self.name}_{name}")
            else:  # persistent
                behavior = behavior_class(name=f"{self.name}_{name}")
            
            self.behaviors.append(behavior)
            self._behavior_dict[behavior.name] = behavior
    
    def _create_decision_strategy(self, strategy: str, params: Optional[Dict]) -> Any:
        """
        Create decision strategy.
        创建决策策略。
        
        Args:
            strategy: Strategy name | 策略名称
            params: Additional parameters | 额外参数
            
        Returns:
            DecisionStrategy: Decision strategy instance | 决策策略实例
        """
        params = params or {}
        
        if strategy == 'min_error':
            return MinErrorStrategy(error_window=params.get('error_window', 10))
        
        elif strategy == 'weighted_vote':
            return WeightedVoteStrategy(temperature=params.get('temperature', 1.0))
        
        elif strategy == 'adaptive':
            return AdaptiveStrategy(
                epsilon=params.get('epsilon', 0.1),
                decay=params.get('decay', 0.99),
                min_epsilon=params.get('min_epsilon', 0.01),
                weighted_threshold=params.get('weighted_threshold', 0.7)
            )
        
        else:
            self._logger.warning(f"Unknown strategy: {strategy}, using min_error")
            return MinErrorStrategy()
    
    def _create_learning_mechanism(self, learning_type: str, lr: float, threshold: float) -> Any:
        """
        Create learning mechanism.
        创建学习机制。
        
        Args:
            learning_type: Learning type | 学习类型
            lr: Learning rate | 学习率
            threshold: Error threshold | 误差阈值
            
        Returns:
            LearningMechanism: Learning mechanism instance | 学习机制实例
        """
        if learning_type == 'error':
            return ErrorLearning(learning_rate=lr, error_threshold=threshold)
        elif learning_type == 'reward':
            return RewardLearning(learning_rate=lr, baseline=threshold)
        else:
            self._logger.warning(f"Unknown learning type: {learning_type}, using error")
            return ErrorLearning(learning_rate=lr, error_threshold=threshold)
    
    def _extract_features(self, state: SystemState) -> FeatureVector:
        """
        Extract features using pattern space.
        使用模式空间提取特征。
        
        Args:
            state: System state | 系统状态
            
        Returns:
            FeatureVector: Feature vector | 特征向量
        """
        return self.pattern_space.extract(state)
    
    def _select_behavior(self, features: FeatureVector) -> BaseChaosBehavior:
        """
        Select best behavior using decision strategy.
        使用决策策略选择最佳行为。
        
        Args:
            features: Feature vector | 特征向量
            
        Returns:
            BaseChaosBehavior: Selected behavior | 选中的行为
        """
        # Compute fitness for each behavior | 计算每个行为的适应度
        fitness_scores = []
        for behavior in self.behaviors:
            fitness = self.selector.compute_fitness(behavior, features)
            fitness_scores.append(fitness)
        
        # Select using decision strategy | 使用决策策略选择
        selected = self.decision_strategy.select(
            self.behaviors, fitness_scores, features
        )
        
        return selected
    
    def _apply_learning(self, behavior: BaseChaosBehavior, prediction: PredictionResult, actual: float):
        """
        Apply learning based on prediction error.
        基于预测误差应用学习。
        
        Args:
            behavior: The behavior that made the prediction | 做出预测的行为
            prediction: Prediction result | 预测结果
            actual: Actual value | 实际值
        """
        if self.learning_mechanism is None:
            return
        
        # Compute adjustment | 计算调整量
        context = {
            'behavior_name': behavior.name,
            'features': None,  # Can be added if needed | 需要时可添加
            'parameters': self.selector.get_parameters(),
        }
        
        adjustment = self.learning_mechanism.compute_adjustment(
            prediction.value, actual, context
        )
        
        # Apply adjustment to selector | 将调整应用于选择器
        if adjustment:
            self.selector.update_parameters(adjustment)
    
    def predict(self, value: float, timestamp: Optional[float] = None) -> PredictionResult:
        """
        Make a prediction for the next value.
        预测下一个值。
        
        Args:
            value: Current observed value | 当前观测值
            timestamp: Optional timestamp | 可选时间戳
            
        Returns:
            PredictionResult: Prediction result | 预测结果
        """
        # Create state | 创建状态
        state = SystemState(
            data=value,
            timestamp=timestamp or time.time(),
            metadata={'start_time': self._start_time}
        )
        
        # Update history | 更新历史
        self._history.append(value)
        if timestamp is not None:
            self._timestamps.append(timestamp)
        
        # Extract features | 提取特征
        features = self._extract_features(state)
        
        # Select behavior | 选择行为
        selected_behavior = self._select_behavior(features)
        self._selected_behavior_history.append(selected_behavior.name)
        
        # Make prediction | 进行预测
        # Pass current value to behavior for better baseline | 传递当前值给行为以获得更好的基线
        prediction_value = selected_behavior.execute(
            features,
            current_value=value,
            state=state
        )
        
        # Handle different return types | 处理不同的返回类型
        if isinstance(prediction_value, PredictionResult):
            result = prediction_value
        else:
            # If behavior returns raw value | 如果行为返回原始值
            result = PredictionResult(
                value=float(prediction_value),
                confidence=0.5,
                method=selected_behavior.name
            )
        
        self._predictions.append(result)
        self._total_predictions += 1
        
        # Record for selector | 记录给选择器
        self.selector.record_fitness(selected_behavior.name, result.confidence)
        
        if self.verbose:
            self._logger.debug(f"Prediction: {result.value:.4f} (method={result.method})")
        
        return result
    
    def update(self, actual_value: float, behavior_name: Optional[str] = None) -> float:
        """
        Update agent with actual observed value.
        用实际观测值更新智能体。
        
        Args:
            actual_value: Actual observed value | 实际观测值
            behavior_name: Specific behavior to update (None = last used)
                           要更新的特定行为（None = 最后使用的）
            
        Returns:
            float: Prediction error | 预测误差
        """
        # Get last prediction | 获取最后一次预测
        if not self._predictions:
            return 0.0
        
        last_prediction = self._predictions[-1]
        
        # Determine which behavior made the prediction | 确定哪个行为做出了预测
        if behavior_name is None:
            behavior_name = last_prediction.method
        
        behavior = self._behavior_dict.get(behavior_name)
        if behavior is None:
            self._logger.warning(f"Behavior {behavior_name} not found")
            return 0.0
        
        # Update behavior with actual value | 用实际值更新行为
        error = behavior.update_actual(actual_value)
        
        # Update selector | 更新选择器
        self.selector.record_prediction(behavior_name, last_prediction, actual_value)
        
        # Apply learning | 应用学习
        if self.enable_learning:
            self._apply_learning(behavior, last_prediction, actual_value)
        
        # Update last prediction with error | 用误差更新最后一次预测
        last_prediction.error = error
        self._predictions[-1] = last_prediction
        
        # Check for restart condition | 检查重启条件
        if error > 1.0:  # Large error threshold | 大误差阈值
            self._restart_count += 1
            self._logger.info(f"Large error {error:.4f}, restarting...")
            self.reset()
        
        return error
    
    def reset(self) -> None:
        """Reset agent state | 重置智能体状态"""
        self.pattern_space.reset()
        
        for behavior in self.behaviors:
            behavior.reset()
        
        self.selector.reset()
        self._history.clear()
        self._timestamps.clear()
        self._predictions.clear()
        self._selected_behavior_history.clear()
        self._total_predictions = 0
        self._start_time = time.time()
        
        self._logger.info("MetaChaos reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics.
        获取智能体统计信息。
        
        Returns:
            Dict: Statistics dictionary | 统计字典
        """
        # Calculate recent error | 计算近期误差
        recent_errors = []
        for pred in self._predictions[-100:]:
            if pred.error is not None:
                recent_errors.append(pred.error)
        
        # Get behavior stats | 获取行为统计
        behavior_stats = {}
        for behavior in self.behaviors:
            behavior_stats[behavior.name] = behavior.get_stats()
        
        return {
            'name': self.name,
            'total_predictions': self._total_predictions,
            'restart_count': self._restart_count,
            'uptime': time.time() - self._start_time,
            'recent_error': float(np.mean(recent_errors)) if recent_errors else None,
            'last_error': self._predictions[-1].error if self._predictions else None,
            'selected_behavior': self._selected_behavior_history[-1] if self._selected_behavior_history else None,
            'behavior_usage': self._get_behavior_usage(),
            'behavior_stats': behavior_stats,
            'selector_stats': {
                'n_behaviors': len(self.behaviors),
                'error_window': self.selector.error_window,
            },
            'learning_enabled': self.enable_learning,
        }
    
    def _get_behavior_usage(self) -> Dict[str, int]:
        """Get behavior usage counts | 获取行为使用次数"""
        usage = {}
        for name in self._selected_behavior_history:
            usage[name] = usage.get(name, 0) + 1
        return usage
    
    def get_behavior(self, name: str) -> Optional[BaseChaosBehavior]:
        """Get behavior by name | 根据名称获取行为"""
        return self._behavior_dict.get(name)
    
    def add_behavior(self, behavior: BaseChaosBehavior) -> None:
        """
        Add a custom behavior.
        添加自定义行为。
        
        Args:
            behavior: Behavior instance | 行为实例
        """
        self.behaviors.append(behavior)
        self._behavior_dict[behavior.name] = behavior
        self._logger.info(f"Added behavior: {behavior.name}")
    
    def remove_behavior(self, name: str) -> bool:
        """
        Remove a behavior.
        移除行为。
        
        Args:
            name: Behavior name | 行为名称
            
        Returns:
            bool: Success status | 成功状态
        """
        if name in self._behavior_dict:
            behavior = self._behavior_dict[name]
            self.behaviors.remove(behavior)
            del self._behavior_dict[name]
            self._logger.info(f"Removed behavior: {name}")
            return True
        return False
    
    def set_decision_strategy(self, strategy: str, **params) -> None:
        """
        Change decision strategy at runtime.
        在运行时更改决策策略。
        
        Args:
            strategy: Strategy name | 策略名称
            **params: Strategy parameters | 策略参数
        """
        self.decision_strategy = self._create_decision_strategy(strategy, params)
        self._logger.info(f"Decision strategy changed to: {strategy}")
    
    def set_learning_rate(self, learning_rate: float) -> None:
        """Set learning rate | 设置学习率"""
        if self.learning_mechanism:
            if hasattr(self.learning_mechanism, 'learning_rate'):
                self.learning_mechanism.learning_rate = learning_rate
            self._logger.debug(f"Learning rate set to {learning_rate}")
    
    def __call__(self, value: float, **kwargs) -> PredictionResult:
        """Make the agent callable | 使智能体可调用"""
        return self.predict(value, **kwargs)
    
    def __repr__(self) -> str:
        return (f"MetaChaos(name='{self.name}', "
                f"predictions={self._total_predictions}, "
                f"behaviors={len(self.behaviors)})")


# ============================================================
# Convenience Functions | 便捷函数
# ============================================================

def create_default_chaos_agent(
    name: str = "DefaultChaos",
    include_neural: bool = True,
    verbose: bool = False
) -> MetaChaos:
    """
    Create a chaos agent with default configuration.
    创建具有默认配置的混沌智能体。
    
    Args:
        name: Agent name | 智能体名称
        include_neural: Include neural network behavior | 包含神经网络行为
        verbose: Enable verbose logging | 启用详细日志
        
    Returns:
        MetaChaos: Configured agent | 配置好的智能体
    """
    behaviors = ['persistent', 'linear_trend', 'phase_space', 'volterra', 'spectral']
    if include_neural:
        behaviors.append('neural')
    
    return MetaChaos(
        name=name,
        window_size=10,
        include_velocity=True,
        include_acceleration=True,
        include_stats=True,
        behaviors=behaviors,
        decision_strategy='adaptive',
        decision_params={'epsilon': 0.1, 'decay': 0.99},
        selector_error_window=10,
        enable_learning=True,
        learning_type='error',
        learning_rate=0.1,
        error_threshold=0.5,
        verbose=verbose
    )


def create_minimal_chaos_agent(name: str = "MinimalChaos") -> MetaChaos:
    """
    Create a minimal chaos agent (fast, fewer behaviors).
    创建最小混沌智能体（快速，更少行为）。
    
    Args:
        name: Agent name | 智能体名称
        
    Returns:
        MetaChaos: Minimal agent | 最小智能体
    """
    return MetaChaos(
        name=name,
        window_size=5,
        include_velocity=True,
        include_acceleration=False,
        include_stats=False,
        behaviors=['persistent', 'linear_trend'],
        decision_strategy='min_error',
        enable_learning=False,
        verbose=False
    )


def create_full_chaos_agent(name: str = "FullChaos", verbose: bool = False) -> MetaChaos:
    """
    Create a full-featured chaos agent (all behaviors, adaptive).
    创建全功能混沌智能体（所有行为，自适应）。
    
    Args:
        name: Agent name | 智能体名称
        verbose: Enable verbose logging | 启用详细日志
        
    Returns:
        MetaChaos: Full-featured agent | 全功能智能体
    """
    return MetaChaos(
        name=name,
        window_size=15,
        include_velocity=True,
        include_acceleration=True,
        include_stats=True,
        behaviors=['persistent', 'linear_trend', 'phase_space', 'volterra', 'neural', 'spectral'],
        decision_strategy='adaptive',
        decision_params={'epsilon': 0.1, 'decay': 0.995, 'min_epsilon': 0.01},
        selector_error_window=20,
        enable_learning=True,
        learning_type='reward',
        learning_rate=0.05,
        error_threshold=0.3,
        memory_size=2000,
        verbose=verbose
    )