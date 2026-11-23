"""
Learning Mechanism Components | 学习机制组件
=============================================
"""

import random
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import deque, defaultdict
import logging
from dataclasses import dataclass, field

from ..core.psi_learning import LearningMechanism
from ..core.types import ParameterDict, FeatureVector
from ..core.exceptions import LearningError


# ============================================================
# Helper Functions
# ============================================================

def safe_float_convert(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _numeric(value: Any) -> Optional[float]:
    """Return a finite float, or None for non-numeric labels/objects."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(out) or np.isinf(out):
        return None
    return out


def safe_array_convert(data: Any, default_dim: int = 1) -> np.ndarray:
    if data is None:
        return np.zeros(default_dim, dtype=np.float64)
    if isinstance(data, np.ndarray):
        return data.astype(np.float64).flatten()
    elif isinstance(data, (list, tuple)):
        try:
            return np.array([safe_float_convert(x) for x in data], dtype=np.float64)
        except:
            return np.zeros(len(data), dtype=np.float64)
    elif isinstance(data, (int, float)):
        return np.array([float(data)], dtype=np.float64)
    else:
        return np.zeros(default_dim, dtype=np.float64)


# ============================================================
# Experience Data Class
# ============================================================

@dataclass
class Experience:
    features: FeatureVector
    behavior: str
    expected: Any = None
    actual: Any = None
    reward: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.features is None:
            self.features = np.array([0.0], dtype=np.float64)
        elif not isinstance(self.features, np.ndarray):
            self.features = np.array(self.features, dtype=np.float64)
        if self.features.ndim > 1:
            self.features = self.features.flatten()
    
    def similarity(self, other: 'Experience') -> float:
        v1 = self.features
        v2 = other.features
        min_len = min(len(v1), len(v2))
        v1 = v1[:min_len]
        v2 = v2[:min_len]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cos_sim = np.dot(v1, v2) / (norm1 * norm2)
        time_decay = np.exp(-abs(self.timestamp - other.timestamp) / 3600)
        behavior_sim = 1.0 if self.behavior == other.behavior else 0.0
        return 0.5 * (cos_sim + 1) / 2 + 0.3 * behavior_sim + 0.2 * time_decay


# ============================================================
# 1. Gradient Learning
# ============================================================

class GradientLearning(LearningMechanism):
    def __init__(self, 
                 learning_rate: float = 0.01, 
                 momentum: float = 0.0,
                 decay: float = 1.0,
                 clip_norm: float = 1.0,
                 loss_function: str = 'mse'):
        super().__init__()
        self.base_lr = learning_rate
        self.current_lr = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.clip_norm = clip_norm
        self.loss_function = loss_function
        self.velocity: Dict[str, float] = {}
        self.loss_history: List[float] = []
        self.gradient_history: Dict[str, List[float]] = defaultdict(list)
        self._logger = logging.getLogger("metathin.learning.GradientLearning")
        self._step = 0
    
    def compute_adjustment(self, expected, actual, context):
        try:
            exp_val = _numeric(expected)
            act_val = _numeric(actual)
            if exp_val is None or act_val is None:
                self._logger.warning(
                    "GradientLearning requires numeric feedback; got expected=%r actual=%r",
                    expected, actual,
                )
                return None

            self._step += 1
            params = context.get('parameters', {})
            features = safe_array_convert(context.get('features'), default_dim=1)
            if not params:
                return None

            error = exp_val - act_val
            loss = error ** 2
            self.loss_history.append(loss)
            if len(self.loss_history) > 1000:
                self.loss_history = self.loss_history[-1000:]

            lr = context.get('learning_rate', self.current_lr)
            adjustments = {}
            for key in params:
                if key.startswith('w_'):
                    parts = key.split('_')
                    try:
                        feat_idx = int(parts[2])
                    except (IndexError, ValueError):
                        feat_idx = 0
                    feat = features[feat_idx] if feat_idx < len(features) else 0.0
                    delta = -2 * lr * error * feat
                elif key.startswith('b_'):
                    delta = -2 * lr * error
                else:
                    continue

                if abs(delta) > self.clip_norm:
                    delta = self.clip_norm if delta > 0 else -self.clip_norm
                if self.momentum > 0:
                    if key in self.velocity:
                        delta = self.momentum * self.velocity[key] + (1 - self.momentum) * delta
                    self.velocity[key] = delta
                if not np.isnan(delta) and not np.isinf(delta):
                    adjustments[key] = delta

            if self.decay < 1.0:
                self.current_lr = self.base_lr * (self.decay ** self._step)
            return adjustments if adjustments else None
        except Exception as e:
            self._logger.error(f"Gradient computation failed: {e}")
            return None

    def get_loss_stats(self):
        if not self.loss_history:
            return {}
        recent = self.loss_history[-100:] if len(self.loss_history) > 100 else self.loss_history
        return {
            'current_loss': self.loss_history[-1],
            'avg_loss': float(np.mean(recent)),
            'min_loss': float(np.min(self.loss_history)),
            'max_loss': float(np.max(self.loss_history)),
        }
    
    def reset(self):
        self.velocity.clear()
        self.gradient_history.clear()
        self.loss_history.clear()
        self.current_lr = self.base_lr
        self._step = 0


# ============================================================
# 2. Reward Learning
# ============================================================

class RewardLearning(LearningMechanism):
    def __init__(self, learning_rate: float = 0.1, baseline: float = 0.5, use_advantage: bool = True):
        super().__init__()
        self.learning_rate = learning_rate
        self.baseline = baseline
        self.use_advantage = use_advantage
        self.reward_history: List[float] = []
        self.cumulative_reward = 0.0
        self.reward_count = 0
        self._logger = logging.getLogger("metathin.learning.RewardLearning")
    
    def compute_adjustment(self, expected, actual, context):
        try:
            reward = _numeric(context.get('reward'))
            if reward is None:
                self._logger.warning("RewardLearning requires a numeric reward in context['reward']")
                return None

            # Compute advantage against the *previous* average, then update stats.
            if self.use_advantage:
                advantage = reward - self.cumulative_reward if self.reward_count else reward - self.baseline
            else:
                advantage = reward - self.baseline

            self.reward_history.append(reward)
            if len(self.reward_history) > 1000:
                self.reward_history = self.reward_history[-1000:]
            self.cumulative_reward = (self.cumulative_reward * self.reward_count + reward)
            self.reward_count += 1
            self.cumulative_reward /= self.reward_count

            params = context.get('parameters', {})
            if not params:
                return None

            features = context.get('features')
            features = safe_array_convert(features, default_dim=1)
            
            adjustments = {}
            effective_reward = advantage
            for key in params:
                if key.startswith('w_'):
                    parts = key.split('_')
                    try:
                        feat_idx = int(parts[2])
                    except (IndexError, ValueError):
                        feat_idx = 0
                    feat = features[feat_idx] if feat_idx < len(features) else 0.0
                    delta = self.learning_rate * effective_reward * feat
                elif key.startswith('b_'):
                    delta = self.learning_rate * effective_reward
                else:
                    continue
                if not np.isnan(delta) and not np.isinf(delta):
                    adjustments[key] = delta

            return adjustments if adjustments else None
            
        except Exception as e:
            self._logger.error(f"Reward learning failed: {e}")
            return None
    
    def get_average_reward(self, window: Optional[int] = None) -> float:
        if not self.reward_history:
            return 0.0
        if window:
            recent = self.reward_history[-min(window, len(self.reward_history)):]
            return float(np.mean(recent))
        return float(np.mean(self.reward_history))
    
    def update_baseline(self, new_baseline: float) -> None:
        self.baseline = new_baseline


# ============================================================
# 3. Memory Learning
# ============================================================

class MemoryLearning(LearningMechanism):
    def __init__(self, memory_size: int = 1000, similarity_threshold: float = 0.7,
                 k_neighbors: int = 5, learning_rate: float = 0.01):
        super().__init__()
        self.memory_size = memory_size
        self.similarity_threshold = similarity_threshold
        self.k_neighbors = k_neighbors
        self.learning_rate = learning_rate
        self.memory: deque = deque(maxlen=memory_size)
        self.recall_count = 0
        self.hit_count = 0
        self.miss_count = 0
        self._logger = logging.getLogger("metathin.learning.MemoryLearning")
    
    def remember(self, experience: Experience) -> None:
        if experience is not None:
            self.memory.append(experience)

    def record_experience(self, features, behavior, expected=None, actual=None,
                          reward: float = 0.0, metadata=None) -> None:
        """Pipeline hook: persist the current thought as an Experience."""
        self.remember(Experience(
            features=np.asarray(features, dtype=np.float64).flatten(),
            behavior=str(behavior),
            expected=expected,
            actual=actual,
            reward=float(reward or 0.0),
            metadata=metadata or {},
        ))
    
    def compute_adjustment(self, expected, actual, context):
        try:
            features = context.get('features')
            behavior_name = context.get('behavior_name')
            
            if features is None or behavior_name is None or len(self.memory) == 0:
                return None
            
            features = safe_array_convert(features).flatten()
            
            similarities = []
            for exp in self.memory:
                if exp is None or exp.behavior != behavior_name:
                    continue
                sim = self._compute_similarity(features, exp.features)
                if sim >= self.similarity_threshold:
                    similarities.append((sim, exp))
            
            similarities.sort(key=lambda x: x[0], reverse=True)
            similar = similarities[:self.k_neighbors]
            
            self.recall_count += 1
            
            if not similar:
                self.miss_count += 1
                return None
            
            self.hit_count += 1
            
            total_weight = 0.0
            weighted_error = 0.0
            
            for sim, exp in similar:
                exp_val = safe_float_convert(exp.expected)
                act_val = safe_float_convert(exp.actual)
                error = exp_val - act_val
                weighted_error += sim * error
                total_weight += sim
            
            if total_weight == 0:
                return None
            
            avg_error = weighted_error / total_weight
            
            params = context.get('parameters', {})
            if not params:
                return None
            
            adjustments = {}
            for key in params:
                if key.startswith('w_'):
                    parts = key.split('_')
                    try:
                        feat_idx = int(parts[2])
                    except (IndexError, ValueError):
                        feat_idx = 0
                    feat = features[feat_idx] if feat_idx < len(features) else 0.0
                    delta = self.learning_rate * avg_error * feat
                    if not np.isnan(delta) and not np.isinf(delta):
                        adjustments[key] = delta
            
            return adjustments if adjustments else None
            
        except Exception as e:
            self._logger.error(f"Memory learning failed: {e}")
            return None
    
    def _compute_similarity(self, f1: np.ndarray, f2: np.ndarray) -> float:
        min_len = min(len(f1), len(f2))
        v1 = f1[:min_len]
        v2 = f2[:min_len]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return (np.dot(v1, v2) / (norm1 * norm2) + 1) / 2
    
    def prune(self, max_age: float) -> int:
        """Prune old memories."""
        now = time.time()
        old_count = len(self.memory)
        new_memory = deque(maxlen=self.memory.maxlen)
        for exp in self.memory:
            if exp is not None and now - exp.timestamp <= max_age:
                new_memory.append(exp)
        removed = old_count - len(new_memory)
        self.memory = new_memory
        return removed
    
    def get_stats(self):
        total = self.hit_count + self.miss_count
        behavior_counts = {}
        for exp in self.memory:
            if exp is not None:
                behavior_counts[exp.behavior] = behavior_counts.get(exp.behavior, 0) + 1
        return {
            'memory_size': len(self.memory),
            'capacity': self.memory.maxlen,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': self.hit_count / max(1, total),
            'behavior_distribution': behavior_counts,
        }
    
    def clear(self):
        self.memory.clear()
        self.recall_count = 0
        self.hit_count = 0
        self.miss_count = 0


# ============================================================
# 4. Hebbian Learning
# ============================================================

class HebbianLearning(LearningMechanism):
    def __init__(self, learning_rate: float = 0.01, use_anti: bool = False,
                 normalize: bool = True, max_weight: float = 10.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.use_anti = use_anti
        self.normalize = normalize
        self.max_weight = max_weight
        self._logger = logging.getLogger("metathin.learning.HebbianLearning")
    
    def _activation(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
    
    def compute_adjustment(self, expected, actual, context):
        try:
            features = context.get('features')
            params = context.get('parameters', {})
            
            if not params:
                return None
            
            features = safe_array_convert(features, default_dim=1)
            act_val = _numeric(actual)
            if act_val is None:
                self._logger.warning("HebbianLearning requires numeric actual output")
                return None
            post = self._activation(act_val)

            adjustments = {}
            for key in params:
                if key.startswith('w_'):
                    parts = key.split('_')
                    try:
                        feat_idx = int(parts[2])
                    except (IndexError, ValueError):
                        feat_idx = 0
                    feat = features[feat_idx] if feat_idx < len(features) else 0.0
                    pre = self._activation(feat)
                    delta = self.learning_rate * pre * post
                    if self.use_anti:
                        delta = -delta
                elif key.startswith('b_'):
                    delta = self.learning_rate * post
                    if self.use_anti:
                        delta = -delta
                else:
                    continue
                if not np.isnan(delta) and not np.isinf(delta):
                    adjustments[key] = delta
            
            return adjustments if adjustments else None
            
        except Exception as e:
            self._logger.error(f"Hebbian learning failed: {e}")
            return None
    
    def post_update(self, params: ParameterDict) -> ParameterDict:
        if not self.normalize:
            return params
        weights = [v for k, v in params.items() if k.startswith('w_')]
        if weights:
            norm = np.sqrt(sum(w**2 for w in weights))
            if norm > 0:
                scale = min(1.0, self.max_weight / norm)
                for k in params:
                    if k.startswith('w_'):
                        params[k] *= scale
        return params


# ============================================================
# 5. Ensemble Learning
# ============================================================

class EnsembleLearning(LearningMechanism):
    def __init__(self, learners: List[LearningMechanism],
                 weights: Optional[List[float]] = None,
                 aggregation: str = 'weighted_average'):
        super().__init__()
        if not learners:
            raise ValueError("Learners list cannot be empty")
        self.learners = learners
        if weights is None:
            self.weights = [1.0 / len(learners)] * len(learners)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]
        self.aggregation = aggregation
        self._logger = logging.getLogger("metathin.learning.EnsembleLearning")
    
    def compute_adjustment(self, expected, actual, context):
        try:
            all_adjustments = []
            valid_weights = []
            
            for learner, weight in zip(self.learners, self.weights):
                try:
                    adj = learner.compute_adjustment(expected, actual, context)
                    if adj:
                        all_adjustments.append(adj)
                        valid_weights.append(weight)
                except Exception as e:
                    self._logger.debug(f"Learner {type(learner).__name__} failed: {e}")
                    continue
            
            if not all_adjustments:
                return None
            
            valid_weights = np.array(valid_weights)
            valid_weights = valid_weights / (valid_weights.sum() + 1e-8)
            
            all_keys = set()
            for adj in all_adjustments:
                all_keys.update(adj.keys())
            
            final_adjustment = {}
            for key in all_keys:
                values = []
                for adj, weight in zip(all_adjustments, valid_weights):
                    if key in adj:
                        delta = adj[key]
                        if not np.isnan(delta) and not np.isinf(delta):
                            values.append(delta * weight)
                
                if values:
                    if self.aggregation == 'weighted_average':
                        final_adjustment[key] = sum(values)
                    elif self.aggregation == 'max':
                        final_adjustment[key] = max(values)
                    elif self.aggregation == 'min':
                        final_adjustment[key] = min(values)
                    else:
                        final_adjustment[key] = sum(values) / len(values)
            
            return final_adjustment if final_adjustment else None
            
        except Exception as e:
            self._logger.error(f"Ensemble learning failed: {e}")
            return None
    
    def add_learner(self, learner: LearningMechanism, weight: Optional[float] = None) -> None:
        """Add a learner to the ensemble."""
        self.learners.append(learner)
        if weight is None:
            new_weight = 1.0 / len(self.learners)
            self.weights = [new_weight] * len(self.learners)
        else:
            self.weights.append(weight)
            total = sum(self.weights)
            if total > 0:
                self.weights = [w / total for w in self.weights]


# ============================================================
# 6. Behavior Feedback Learning (negative feedback)
# ============================================================

class BehaviorFeedbackLearning(LearningMechanism):
    """Behavior-level negative feedback.

    与 GradientLearning 的“输出回归”不同，这里只惩罚/奖励本次被执行的
    行为：选错时压低该行为的适应度权重；选对时默认不更新（pure negative
    feedback），可通过 ``positive_feedback`` 加一点正向强化。selector
    权重决定“选谁”，因此反馈方向与决策语义一致。

    对分类任务：
        feedback = positive_feedback (actual == expected) / -1 (actual != expected)
    对无标签任务：
        feedback = context['reward']（裁剪到 [-1, 1]）

    只更新被选中行为的权重行，不做 bias 更新（bias 容易把不同行为一起抬高）。

    训练阶段可按 ``exploration_rate`` 做 ε-greedy 概率探索；若 top-1/top-2
    适应度差已超过 ``exploration_margin``，则视为足够自信、不注入噪声。
    推理（未传 expected）不探索，保持确定性。
    """

    def __init__(self, learning_rate: float = 0.01, normalize_features: bool = True,
                 max_step: float = 0.05, bias_scale: float = 0.0,
                 positive_feedback: float = 0.0, numeric_tolerance: float = 1e-6,
                 exploration_rate: float = 0.2, exploration_decay: float = 0.99,
                 min_exploration: float = 0.01, exploration_margin: float = 0.1):
        super().__init__()
        if not 0.0 <= exploration_rate <= 1.0:
            raise ValueError(f"exploration_rate must be in [0,1], got {exploration_rate}")
        if not 0.0 < exploration_decay <= 1.0:
            raise ValueError(f"exploration_decay must be in (0,1], got {exploration_decay}")
        if not 0.0 <= min_exploration <= 1.0:
            raise ValueError(f"min_exploration must be in [0,1], got {min_exploration}")
        if exploration_margin < 0.0:
            raise ValueError(f"exploration_margin must be >= 0, got {exploration_margin}")
        self.learning_rate = learning_rate
        self.normalize_features = normalize_features
        self.max_step = max_step
        self.bias_scale = bias_scale
        self.positive_feedback = positive_feedback
        self.numeric_tolerance = numeric_tolerance
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.exploration_margin = exploration_margin
        self._exploration_step = 0
        self._exploration_count = 0
        self.feedback_history: List[float] = []
        self._scales: List[float] = []
        self._logger = logging.getLogger("metathin.learning.BehaviorFeedbackLearning")

    def should_explore(self, fitness_scores: Optional[List[float]] = None) -> bool:
        """Sample an epsilon-greedy exploration decision, then decay epsilon.

        If ``exploration_margin`` is positive and the top-1/top-2 fitness gap
        already exceeds it, the decision is confident enough and no noise is
        injected. This keeps confident predictions stable while exploring
        ambiguous regions.
        """
        if self.exploration_rate <= 0.0:
            return False

        explore = True
        if fitness_scores is not None and len(fitness_scores) >= 2 and self.exploration_margin > 0.0:
            top = sorted((float(v) for v in fitness_scores), reverse=True)
            margin = top[0] - top[1]
            if margin >= self.exploration_margin:
                explore = False

        if explore:
            explore = random.random() < self.exploration_rate
            if explore:
                self._exploration_count += 1

        self._exploration_step += 1
        self.exploration_rate = max(
            self.min_exploration, self.exploration_rate * self.exploration_decay
        )
        return explore

    def get_exploration_rate(self) -> float:
        """Current exploration probability | 当前探索概率"""
        return self.exploration_rate

    def compute_adjustment(self, expected, actual, context):
        try:
            behavior_index = context.get('behavior_index')
            if behavior_index is None:
                self._logger.warning(
                    "BehaviorFeedbackLearning requires context['behavior_index']"
                )
                return None

            if expected is not None:
                exp_num = _numeric(expected)
                act_num = _numeric(actual)
                if exp_num is not None and act_num is not None:
                    # Numeric target: graded negative feedback by relative error.
                    rel_error = abs(exp_num - act_num) / max(abs(exp_num), 1.0)
                    correct = rel_error <= self.numeric_tolerance
                    feedback = self.positive_feedback if correct else -min(1.0, rel_error)
                else:
                    correct = expected == actual
                    feedback = self.positive_feedback if correct else -1.0
            else:
                reward = _numeric(context.get('reward'))
                if reward is None:
                    return None
                feedback = max(-1.0, min(1.0, reward))

            features = safe_array_convert(context.get('features'), default_dim=1)
            if self.normalize_features:
                if len(self._scales) < len(features):
                    self._scales.extend([1.0] * (len(features) - len(self._scales)))
                for j, value in enumerate(features):
                    self._scales[j] = max(self._scales[j], abs(float(value)), 1.0)
                features = features / np.asarray(self._scales[:len(features)], dtype=np.float64)

            self.feedback_history.append(feedback)
            if len(self.feedback_history) > 1000:
                self.feedback_history = self.feedback_history[-1000:]

            idx = int(behavior_index)
            adjustments = {}
            for j, feat in enumerate(features):
                delta = self.learning_rate * feedback * float(feat)
                if abs(delta) > self.max_step:
                    delta = self.max_step if delta > 0 else -self.max_step
                adjustments[f'w_{idx}_{j}'] = delta

            if self.bias_scale:
                adjustments[f'b_{idx}'] = self.learning_rate * feedback * self.bias_scale

            return adjustments if adjustments else None
        except Exception as e:
            self._logger.error(f"Behavior feedback failed: {e}")
            return None

    def get_stats(self):
        if not self.feedback_history:
            return super().get_stats()
        recent = self.feedback_history[-100:]
        return {
            'name': self.__class__.__name__,
            'type': 'learning_mechanism',
            'updates': len(self.feedback_history),
            'positive_rate': sum(1 for f in recent if f > 0) / len(recent),
            'avg_feedback': float(np.mean(recent)),
            'exploration_rate': self.exploration_rate,
            'exploration_count': self._exploration_count,
        }


# ============================================================
# Export
# ============================================================

__all__ = [
    'GradientLearning',
    'RewardLearning',
    'MemoryLearning',
    'HebbianLearning',
    'EnsembleLearning',
    'BehaviorFeedbackLearning',
    'Experience',
]