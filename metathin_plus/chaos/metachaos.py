# metathin_plus/chaos/metachaos.py - 完整修复版

"""
MetaChaos - Main Agent for Chaos Prediction with Training Phase
================================================================
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import time
import logging
from dataclasses import dataclass, field
from collections import defaultdict, Counter

from metathin.core.types import FeatureVector

from .base import SystemState, PredictionResult
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
from .decision import MinErrorStrategy, AdaptiveStrategy


@dataclass
class TrainingSample:
    """训练样本"""
    features: np.ndarray
    behavior_name: str
    prediction: float
    actual: float
    error: float
    timestamp: float


class MetaChaos:
    """
    MetaChaos - Main chaos prediction agent with training phase.
    """
    
    BEHAVIOR_REGISTRY = {
        'persistent': PersistentBehavior,
        'linear_trend': LinearTrendBehavior,
        'phase_space': PhaseSpaceBehavior,
        'volterra': VolterraBehavior,
        'neural': NeuralBehavior,
        'spectral': SpectralBehavior,
    }
    
    def __init__(
        self,
        name: str = "MetaChaos",
        window_size: int = 15,
        include_velocity: bool = True,
        include_acceleration: bool = True,
        include_stats: bool = True,
        behaviors: Optional[List[str]] = None,
        feature_dim: Optional[int] = None,
        learning_rate: float = 0.01,
        temperature: float = 1.0,
        verbose: bool = False,
    ):
        self.name = name
        self.verbose = verbose
        self.learning_rate = learning_rate
        
        self._logger = logging.getLogger(f"metathin_plus.chaos.{name}")
        if verbose:
            self._logger.setLevel(logging.DEBUG)
        
        # Pattern Space (P)
        self.pattern_space = ChaosPatternSpace(
            window_size=window_size,
            include_velocity=include_velocity,
            include_acceleration=include_acceleration,
            include_stats=include_stats,
            name=f"{name}_pattern"
        )
        
        # Determine feature dimension
        if feature_dim is None:
            sample_state = SystemState(data=0.0, timestamp=0)
            sample_features = self.pattern_space.extract(sample_state)
            self.feature_dim = len(sample_features)
        else:
            self.feature_dim = feature_dim
        
        # Behaviors (B)
        self.behaviors: List[BaseChaosBehavior] = []
        self._behavior_dict: Dict[str, BaseChaosBehavior] = {}
        self._init_behaviors(behaviors)
        
        # Selector (S)
        self.selector = ChaosSelector(
            n_features=self.feature_dim,
            n_behaviors=len(self.behaviors),
            learning_rate=learning_rate,
            temperature=temperature,
            use_history=True,
            history_weight=0.3
        )
        
        # Decision Strategy (D)
        self.decision_strategy = AdaptiveStrategy(epsilon=0.1, decay=0.99, min_epsilon=0.01)
        
        # Training state
        self._is_trained = False
        self._training_samples: List[TrainingSample] = []
        
        # Runtime state
        self._history: List[float] = []
        self._timestamps: List[float] = []
        self._predictions: List[PredictionResult] = []
        self._selected_history: List[str] = []
        self._error_history: List[float] = []
        
        # Statistics
        self._total_predictions = 0
        self._train_steps = 0
        self._exploration_count = 0
        self._exploitation_count = 0
        self._start_time = time.time()
        
        self._logger.info(f"MetaChaos '{name}' initialized with {len(self.behaviors)} behaviors")
        self._logger.info(f"  Feature dimension: {self.feature_dim}")
    
    def _init_behaviors(self, behavior_names: Optional[List[str]]):
        """初始化行为"""
        if behavior_names is None:
            behavior_names = list(self.BEHAVIOR_REGISTRY.keys())
        
        for name in behavior_names:
            if name not in self.BEHAVIOR_REGISTRY:
                self._logger.warning(f"Unknown behavior: {name}, skipping")
                continue
            
            behavior_class = self.BEHAVIOR_REGISTRY[name]
            
            if name == 'phase_space':
                behavior = behavior_class(
                    embed_dim=5, delay=3, k_neighbors=5,
                    min_history=50, name=f"{self.name}_{name}"
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
            else:
                behavior = behavior_class(name=f"{self.name}_{name}")
            
            self.behaviors.append(behavior)
            self._behavior_dict[behavior.name] = behavior
    
    # ============================================================
    # 训练阶段
    # ============================================================
    
    def train(
        self,
        data: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        train_steps: Optional[int] = None,
        exploration_rate: float = 0.3,
        exploration_decay: float = 0.99,
        min_exploration: float = 0.05,
        batch_size: int = 50,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """训练智能体"""
        if timestamps is None:
            timestamps = np.arange(len(data))
        
        if train_steps is None:
            train_steps = len(data)
        
        train_steps = min(train_steps, len(data))
        train_data = data[:train_steps]
        train_times = timestamps[:train_steps]
        
        self._logger.info(f"\n{'='*60}")
        self._logger.info(f"TRAINING PHASE | 训练阶段")
        self._logger.info(f"  Steps: {train_steps}")
        self._logger.info(f"  Initial exploration rate: {exploration_rate}")
        self._logger.info(f"{'='*60}")
        
        current_epsilon = exploration_rate
        self._training_samples.clear()
        
        behavior_errors = {b.name: [] for b in self.behaviors}
        behavior_counts = {b.name: 0 for b in self.behaviors}
        
        for step in range(train_steps):
            value = train_data[step]
            t = train_times[step]
            
            self._history.append(value)
            self._timestamps.append(t)
            
            if step == 0:
                continue
            
            state = SystemState(data=value, timestamp=t, metadata={'step': step, 'phase': 'train'})
            features = self.pattern_space.extract(state)
            
            # 探索 vs 利用
            if np.random.random() < current_epsilon:
                behavior = np.random.choice(self.behaviors)
                self._exploration_count += 1
            else:
                fitness_scores = [self.selector.compute_fitness(b, features) for b in self.behaviors]
                best_idx = np.argmax(fitness_scores)
                behavior = self.behaviors[best_idx]
                self._exploitation_count += 1
            
            behavior_counts[behavior.name] += 1
            
            # 用上一个值预测当前值
            prev_value = train_data[step - 1]
            pred_result = behavior.execute(features, current_value=prev_value, state=state)
            
            if isinstance(pred_result, PredictionResult):
                pred_value = pred_result.value
            else:
                pred_value = pred_result
            
            error = abs(pred_value - value)
            behavior_errors[behavior.name].append(error)
            
            self._training_samples.append(TrainingSample(
                features=features.copy(),
                behavior_name=behavior.name,
                prediction=pred_value,
                actual=value,
                error=error,
                timestamp=t
            ))
            
            # 批量更新权重
            if step % batch_size == 0 and step > 0:
                self._update_selector_weights(behavior_errors)
            
            current_epsilon = max(min_exploration, current_epsilon * exploration_decay)
            
            if verbose and (step + 1) % (max(1, train_steps // 10)) == 0:
                recent_errors = [e for errs in behavior_errors.values() for e in errs[-100:]]
                avg_error = np.mean(recent_errors) if recent_errors else 1.0
                self._logger.info(f"  Step {step+1}/{train_steps}: ε={current_epsilon:.3f}, avg_error={avg_error:.4f}")
        
        # 最终更新权重
        self._update_selector_weights(behavior_errors, final=True)
        
        # 分析结果
        training_stats = self._analyze_training_results(behavior_counts, behavior_errors)
        
        self._is_trained = True
        self._train_steps = train_steps
        
        self._print_training_summary(training_stats)
        
        return training_stats
    
    def _update_selector_weights(self, behavior_errors: Dict[str, List[float]], final: bool = False):
        """更新选择器权重"""
        avg_errors = {}
        for name, errors in behavior_errors.items():
            if errors:
                avg_errors[name] = np.mean(errors[-100:])
            else:
                avg_errors[name] = 0.5
        
        best_behavior = min(avg_errors, key=avg_errors.get)
        
        if final:
            samples = self._training_samples
        else:
            samples = self._training_samples[-100:]
        
        for sample in samples:
            behavior_idx = self.selector._behavior_indices.get(sample.behavior_name)
            if behavior_idx is None:
                continue
            
            target_fitness = 1.0 if sample.behavior_name == best_behavior else 0.0
            
            z = np.dot(self.selector.weights[behavior_idx], sample.features) + self.selector.bias[behavior_idx]
            current_fitness = 1.0 / (1.0 + np.exp(-z / self.selector.temperature))
            
            error = target_fitness - current_fitness
            grad = current_fitness * (1 - current_fitness) * error / self.selector.temperature
            
            for j in range(self.feature_dim):
                self.selector.weights[behavior_idx, j] += self.learning_rate * grad * sample.features[j]
            self.selector.bias[behavior_idx] += self.learning_rate * grad
    
    def _analyze_training_results(self, behavior_counts: Dict[str, int], 
                                   behavior_errors: Dict[str, List[float]]) -> Dict[str, Any]:
        """分析训练结果"""
        behavior_stats = {}
        for behavior in self.behaviors:
            name = behavior.name
            count = behavior_counts.get(name, 0)
            errors = behavior_errors.get(name, [])
            
            behavior_stats[name] = {
                'usage_count': count,
                'usage_percentage': count / self._train_steps * 100 if self._train_steps > 0 else 0,
                'mean_error': np.mean(errors) if errors else 1.0,
                'std_error': np.std(errors) if errors else 0,
                'min_error': np.min(errors) if errors else 1.0,
                'max_error': np.max(errors) if errors else 1.0,
            }
        
        best_behavior = min(behavior_stats.items(), key=lambda x: x[1]['mean_error'])
        
        return {
            'total_steps': self._train_steps,
            'exploration_count': self._exploration_count,
            'exploitation_count': self._exploitation_count,
            'exploration_rate': self._exploration_count / self._train_steps if self._train_steps > 0 else 0,
            'behavior_stats': behavior_stats,
            'best_behavior': best_behavior[0],
            'best_behavior_error': best_behavior[1]['mean_error'],
        }
    
    def _print_training_summary(self, stats: Dict[str, Any]):
        """打印训练总结"""
        self._logger.info(f"\n{'='*60}")
        self._logger.info(f"TRAINING SUMMARY | 训练总结")
        self._logger.info(f"{'='*60}")
        
        self._logger.info(f"\nOverall Statistics:")
        self._logger.info(f"  Total steps: {stats['total_steps']}")
        self._logger.info(f"  Exploration rate: {stats['exploration_rate']:.1%}")
        self._logger.info(f"  Best behavior: {stats['best_behavior']} (error={stats['best_behavior_error']:.6f})")
        
        self._logger.info(f"\nBehavior Performance:")
        self._logger.info(f"  {'Behavior':<40} {'Usage':>8} {'Mean Error':>12}")
        self._logger.info(f"  {'-'*60}")
        
        behavior_stats = stats['behavior_stats']
        for name, bstats in sorted(behavior_stats.items(), key=lambda x: x[1]['mean_error']):
            short_name = name.replace(f"{self.name}_", "")[:35]
            self._logger.info(f"  {short_name:<40} {bstats['usage_count']:>8} {bstats['mean_error']:>12.6f}")
        
        self._logger.info(f"\n✅ Training complete!")
    
    # ============================================================
    # 预测阶段
    # ============================================================
    
    def predict(self, value: float, timestamp: Optional[float] = None) -> PredictionResult:
        """预测下一个值"""
        if timestamp is None:
            timestamp = time.time()
        
        self._history.append(value)
        self._timestamps.append(timestamp)
        
        state = SystemState(data=value, timestamp=timestamp, 
                           metadata={'step': len(self._history), 'phase': 'prediction'})
        features = self.pattern_space.extract(state)
        
        fitness_scores = [self.selector.compute_fitness(b, features) for b in self.behaviors]
        best_idx = np.argmax(fitness_scores)
        selected_behavior = self.behaviors[best_idx]
        self._selected_history.append(selected_behavior.name)
        
        pred_result = selected_behavior.execute(features, current_value=value, state=state)
        
        if isinstance(pred_result, PredictionResult):
            result = pred_result
            if result.method == "unknown":
                result.method = selected_behavior.name
        else:
            result = PredictionResult(
                value=float(pred_result),
                confidence=fitness_scores[best_idx],
                method=selected_behavior.name
            )
        
        self._predictions.append(result)
        self._total_predictions += 1
        
        return result
    
    def update(self, actual_value: float) -> float:
        """用实际值更新智能体"""
        if not self._predictions:
            return 0.0
        
        last_prediction = self._predictions[-1]
        error = abs(last_prediction.value - actual_value)
        self._error_history.append(error)
        
        behavior_name = last_prediction.method
        self.selector.record_prediction_error(behavior_name, error)
        
        behavior = self._behavior_dict.get(behavior_name)
        if behavior:
            behavior.update_actual(actual_value)
        
        return error
    
    def reset(self):
        """重置运行时状态（保留训练好的权重）"""
        self._history.clear()
        self._timestamps.clear()
        self._predictions.clear()
        self._selected_history.clear()
        self._error_history.clear()
        self._total_predictions = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        runtime = time.time() - self._start_time
        behavior_usage = Counter(self._selected_history)
        recent_errors = self._error_history[-100:] if self._error_history else []
        
        return {
            'name': self.name,
            'is_trained': self._is_trained,
            'train_steps': self._train_steps,
            'total_predictions': self._total_predictions,
            'runtime': runtime,
            'exploration_count': self._exploration_count,
            'exploitation_count': self._exploitation_count,
            'behavior_usage': dict(behavior_usage),
            'recent_mean_error': np.mean(recent_errors) if recent_errors else None,
            'feature_dim': self.feature_dim,
            'n_behaviors': len(self.behaviors),
        }