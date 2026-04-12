"""
Decision Strategy Components | 决策策略组件
===========================================

Provides various behavior selection strategies that make decisions based on fitness scores.
Decision strategies embody the agent's "personality" and exploration-exploitation trade-off.

提供各种基于适应度分数做出决策的行为选择策略。
决策策略体现了代理的"个性"和探索-利用权衡。

Strategy Types | 策略类型:
    - MaxFitnessStrategy: Always selects highest fitness | 总是选择最高适应度
    - ProbabilisticStrategy: Selects proportionally to fitness | 按适应度比例选择
    - EpsilonGreedyStrategy: Explores with probability ε | 以 ε 概率探索
    - RoundRobinStrategy: Cycles through behaviors | 循环遍历行为
    - RandomStrategy: Completely random selection | 完全随机选择
    - BoltzmannStrategy: Softmax with temperature | 带温度的 Softmax
    - HybridStrategy: Combines multiple strategies | 组合多种策略

Design Philosophy | 设计理念:
    - Diversity: Supports deterministic, stochastic, and hybrid strategies
      多样性：支持确定性、随机性和混合策略
    - Tunable: Exploration degree can be controlled via parameters
      可调节：可通过参数控制探索程度
    - Observable: Provides confidence and probability information
      可观测：提供置信度和概率信息
    - Robust: Handles edge cases and numerical issues gracefully
      健壮：优雅处理边缘情况和数值问题
"""

import time
import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import logging
from collections import deque

# ============================================================
# Fixed Imports for Refactored Core | 重构后核心的修复导入
# ============================================================
from ..core.d_decision import DecisionStrategy
from ..core.b_behavior import MetaBehavior
from ..core.types import FeatureVector
from ..core.exceptions import DecisionError, NoBehaviorError


# ============================================================
# Helper Functions | 辅助函数
# ============================================================

def softmax(scores: List[float], temperature: float = 1.0) -> np.ndarray:
    """
    Safe softmax function.
    
    安全的 softmax 函数。
    
    Converts scores to probability distribution, commonly used in probabilistic selection strategies.
    
    将分数转换为概率分布，常用于概率选择策略。
    
    Mathematical principle | 数学原理:
        p_i = exp(s_i / T) / Σ_j exp(s_j / T)
    
    Args | 参数:
        scores: List of raw scores | 原始分数列表
        temperature: Temperature parameter controlling randomness
                    控制随机性的温度参数
            - Higher temperature → more uniform distribution (more exploration)
              更高温度 → 更均匀分布（更多探索）
            - Lower temperature → more peaked distribution (more exploitation)
              更低温度 → 更集中分布（更多利用）
    
    Returns | 返回:
        np.ndarray: Probability distribution summing to 1 | 总和为 1 的概率分布
    """
    scores = np.array(scores, dtype=np.float64)
    
    # Subtract maximum for numerical stability to prevent exp overflow
    # 减去最大值以保持数值稳定，防止 exp 溢出
    scores = scores - np.max(scores)
    
    # Apply temperature | 应用温度
    if temperature != 1.0:
        scores = scores / temperature
    
    # Compute exp, clip to prevent overflow | 计算 exp，裁剪以防止溢出
    exp_scores = np.exp(np.clip(scores, -500, 500))
    
    # Normalize | 归一化
    probs = exp_scores / (np.sum(exp_scores) + 1e-8)
    
    # Ensure probabilities sum to 1 | 确保概率总和为 1
    probs = probs / np.sum(probs)
    
    return probs


def normalize_fitness(fitness_scores: List[float], epsilon: float = 1e-8) -> np.ndarray:
    """
    Normalize fitness scores to a probability distribution.
    
    将适应度分数归一化为概率分布。
    
    Ensures non-negative values that sum to 1.
    
    确保非负值且总和为 1。
    
    Args | 参数:
        fitness_scores: List of fitness scores | 适应度分数列表
        epsilon: Small value to avoid division by zero | 防止除零的小值
    
    Returns | 返回:
        np.ndarray: Normalized probability distribution | 归一化后的概率分布
    """
    scores = np.array(fitness_scores, dtype=np.float64)
    
    # Handle all zeros case | 处理全零情况
    if np.all(scores < epsilon):
        return np.ones_like(scores) / len(scores)
    
    # Ensure non-negative | 确保非负
    scores = np.maximum(scores, epsilon)
    
    return scores / (np.sum(scores) + epsilon)


# ============================================================
# 1. Maximum Fitness Strategy | 最大适应度策略
# ============================================================

class MaxFitnessStrategy(DecisionStrategy):
    """
    Maximum Fitness Strategy: Always selects the behavior with highest fitness.
    
    最大适应度策略：总是选择适应度最高的行为。
    
    This is the most straightforward deterministic strategy, suitable for exploitation-heavy scenarios.
    When multiple behaviors have the same maximum fitness, tie-breaking is applied.
    
    这是最直接的确定性策略，适用于重利用场景。
    当多个行为具有相同的最大适应度时，应用平局打破机制。
    
    Characteristics | 特性:
        - Deterministic: Same input always yields same output | 确定性：相同输入总是产生相同输出
        - Greedy: Always picks what's currently considered best | 贪婪：总是选择当前认为最好的
        - No exploration: Never tries suboptimal behaviors | 无探索：从不尝试次优行为
    
    Attributes | 属性:
        tie_breaker: Tie-breaking method | 平局打破方法
            - 'first': Select the first occurrence | 选择第一个出现的
            - 'last': Select the last occurrence | 选择最后一个出现的
            - 'random': Select randomly among ties | 在平局中随机选择
    """
    
    def __init__(self, tie_breaker: str = 'random'):
        """
        Initialize maximum fitness strategy.
        
        初始化最大适应度策略。
        
        Args | 参数:
            tie_breaker: Tie-breaking method ('first', 'last', 'random')
                        平局打破方法
        
        Raises | 抛出:
            ValueError: If tie_breaker is invalid | 如果 tie_breaker 无效
        """
        if tie_breaker not in ['first', 'last', 'random']:
            raise ValueError(f"tie_breaker must be 'first', 'last', or 'random'")
        
        self.tie_breaker = tie_breaker
        self._logger = logging.getLogger("metathin.decision.MaxFitnessStrategy")
        self._selection_history: List[Dict] = []
    
    def select(self, 
               behaviors: List[MetaBehavior], 
               fitness_scores: List[float],
               features: FeatureVector) -> MetaBehavior:
        """
        Select the behavior with maximum fitness.
        
        选择适应度最大的行为。
        
        Args | 参数:
            behaviors: List of candidate behaviors | 候选行为列表
            fitness_scores: Corresponding fitness scores | 对应的适应度分数
            features: Current feature vector | 当前特征向量
            
        Returns | 返回:
            MetaBehavior: Selected behavior | 选中的行为
            
        Raises | 抛出:
            NoBehaviorError: If behaviors list is empty | 如果行为列表为空
            DecisionError: If input validation fails | 如果输入验证失败
        """
        # Input validation | 输入验证
        if not behaviors:
            raise NoBehaviorError("No behaviors available")
        
        if len(behaviors) != len(fitness_scores):
            raise DecisionError(
                f"Number of behaviors ({len(behaviors)}) and fitness scores ({len(fitness_scores)}) mismatch"
            )
        import random
        # Handle all equal scores | 处理所有分数相等的情况
        if len(set(fitness_scores)) == 1:
            self._logger.debug("All fitness scores equal, selecting randomly")
            import random; return random.choice(behaviors)
        
        # Find maximum fitness | 找到最大适应度
        max_fitness = max(fitness_scores)
        max_indices = [i for i, f in enumerate(fitness_scores) if abs(f - max_fitness) < 1e-6]
        
        # Handle ties | 处理平局
        if len(max_indices) > 1:
            self._logger.debug(f"Detected tie: {len(max_indices)} behaviors with same fitness {max_fitness:.3f}")
            
            if self.tie_breaker == 'first':
                idx = max_indices[0]
            elif self.tie_breaker == 'last':
                idx = max_indices[-1]
            else:  # random
                import random
                idx = random.choice(max_indices)
        else:
            idx = max_indices[0]
        
        selected = behaviors[idx]
        
        # Record decision | 记录决策
        self._selection_history.append({
            'timestamp': time.time(),
            'selected': selected.name,
            'fitness': max_fitness,
            'n_candidates': len(behaviors),
            'tie_resolved': len(max_indices) > 1
        })
        
        return selected
    
    def get_confidence(self, fitness_scores: List[float]) -> float:
        """
        Calculate decision confidence.
        
        计算决策置信度。
        
        Uses the difference between highest and second-highest scores as confidence.
        
        使用最高分与第二高分的差值作为置信度。
        
        Args | 参数:
            fitness_scores: List of fitness scores | 适应度分数列表
            
        Returns | 返回:
            float: Confidence in [0,1] | [0,1] 范围内的置信度
        """
        if len(fitness_scores) < 2:
            return 1.0
        
        sorted_scores = sorted(fitness_scores, reverse=True)
        diff = sorted_scores[0] - sorted_scores[1]
        
        return min(1.0, max(0.0, diff))


# ============================================================
# 2. Probabilistic Strategy | 概率策略
# ============================================================

class ProbabilisticStrategy(DecisionStrategy):
    """
    Probabilistic Strategy: Selects behaviors proportionally to their fitness.
    
    概率策略：按适应度比例选择行为。
    
    Behaviors with higher fitness have higher selection probability, but exploration is still possible.
    Temperature parameter controls randomness.
    
    适应度越高的行为被选中的概率越高，但仍然可能进行探索。
    温度参数控制随机性。
    
    Characteristics | 特性:
        - Stochastic: Same input may yield different outputs | 随机性：相同输入可能产生不同输出
        - Exploratory: Suboptimal behaviors have chance to be selected | 探索性：次优行为有机会被选中
        - Tunable: Temperature controls exploration degree | 可调节：温度控制探索程度
    
    Attributes | 属性:
        temperature: Temperature parameter | 温度参数
            - temperature = 1.0: Direct fitness proportion | 直接适应度比例
            - temperature > 1.0: More random (more exploration) | 更随机（更多探索）
            - temperature < 1.0: More deterministic (more exploitation) | 更确定（更多利用）
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize probabilistic strategy.
        
        初始化概率策略。
        
        Args | 参数:
            temperature: Temperature parameter, must be > 0 | 温度参数，必须 > 0
            
        Raises | 抛出:
            ValueError: If temperature <= 0 | 如果 temperature <= 0
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        
        self.temperature = temperature
        self._logger = logging.getLogger("metathin.decision.ProbabilisticStrategy")
        self._selection_history: List[Dict] = []
    
    def select(self, 
               behaviors: List[MetaBehavior], 
               fitness_scores: List[float],
               features: FeatureVector) -> MetaBehavior:
        """
        Select behavior probabilistically.
        
        概率性地选择行为。
        
        Args | 参数:
            behaviors: List of candidate behaviors | 候选行为列表
            fitness_scores: Corresponding fitness scores | 对应的适应度分数
            features: Current feature vector | 当前特征向量
            
        Returns | 返回:
            MetaBehavior: Selected behavior | 选中的行为
        """
        if not behaviors:
            raise NoBehaviorError("No behaviors available")
        
        # Calculate selection probabilities | 计算选择概率
        probs = self.get_probabilities(fitness_scores)
        
        self._logger.debug(f"Selection probabilities: {dict(zip([b.name for b in behaviors], probs))}")
        
        # Random selection | 随机选择
        try:
            idx = np.random.choice(len(behaviors), p=probs)
            selected = behaviors[idx]
        except ValueError as e:
            # Probability distribution may have issues, fallback to uniform random
            # 概率分布可能有问题，回退到均匀随机
            self._logger.warning(f"Probabilistic selection failed ({e}), falling back to uniform random")
            selected = self._random.choice(behaviors)
            idx = behaviors.index(selected)
        
        # Record decision | 记录决策
        self._selection_history.append({
            'timestamp': time.time(),
            'selected': selected.name,
            'probability': float(probs[idx]),
            'temperature': self.temperature
        })
        
        return selected
    
    def get_probabilities(self, fitness_scores: List[float]) -> List[float]:
        """
        Get selection probability for each behavior.
        
        获取每个行为的选择概率。
        
        Args | 参数:
            fitness_scores: List of fitness scores | 适应度分数列表
            
        Returns | 返回:
            List[float]: Probabilities summing to 1 | 总和为 1 的概率列表
        """
        if not fitness_scores:
            return []
        
        # Use softmax to compute probabilities | 使用 softmax 计算概率
        probs = softmax(fitness_scores, self.temperature)
        
        return probs.tolist()
    
    def get_confidence(self, fitness_scores: List[float]) -> float:
        """
        Calculate decision confidence.
        
        计算决策置信度。
        
        Uses the ratio between highest and second-highest probabilities.
        
        使用最高概率与第二高概率的比率。
        
        Args | 参数:
            fitness_scores: List of fitness scores | 适应度分数列表
            
        Returns | 返回:
            float: Confidence in [0,1] | [0,1] 范围内的置信度
        """
        probs = self.get_probabilities(fitness_scores)
        
        if len(probs) < 2:
            return 1.0
        
        sorted_probs = sorted(probs, reverse=True)
        
        if sorted_probs[1] > 0:
            confidence = 1.0 - (sorted_probs[1] / sorted_probs[0])
        else:
            confidence = 1.0
        
        return min(1.0, max(0.0, confidence))
    
    def set_temperature(self, temperature: float) -> None:
        """
        Dynamically adjust temperature.
        
        动态调整温度。
        
        Args | 参数:
            temperature: New temperature value | 新的温度值
        """
        if temperature <= 0:
            raise ValueError(f"Temperature must be > 0")
        self.temperature = temperature


# ============================================================
# 3. Epsilon-Greedy Strategy | ε-贪婪策略
# ============================================================

class EpsilonGreedyStrategy(DecisionStrategy):
    """
    Epsilon-Greedy Strategy: Explores randomly with probability ε, exploits optimally with probability 1-ε.
    
    ε-贪婪策略：以 ε 概率随机探索，以 1-ε 概率最优利用。
    
    Classic reinforcement learning exploration strategy balancing exploration and exploitation.
    Supports decay mechanism to gradually reduce exploration.
    
    经典的强化学习探索策略，平衡探索和利用。
    支持衰减机制以逐渐减少探索。
    
    Characteristics | 特性:
        - Simple and intuitive: Only two modes | 简单直观：只有两种模式
        - Tunable: ε controls exploration degree | 可调节：ε 控制探索程度
        - Decay: Automatically reduces exploration over time | 衰减：随时间自动减少探索
    
    Attributes | 属性:
        epsilon: Current exploration probability | 当前探索概率
        decay: Decay factor (multiplied after each selection) | 衰减因子（每次选择后相乘）
        min_epsilon: Minimum exploration probability | 最小探索概率
    """
    
    def __init__(self, 
                 epsilon: float = 0.1, 
                 decay: float = 1.0, 
                 min_epsilon: float = 0.01):
        """
        Initialize epsilon-greedy strategy.
        
        初始化 ε-贪婪策略。
        
        Args | 参数:
            epsilon: Initial exploration probability, must be in [0,1] | 初始探索概率，必须在 [0,1] 范围内
            decay: Decay factor, multiplied after each selection | 衰减因子，每次选择后相乘
            min_epsilon: Minimum exploration probability | 最小探索概率
            
        Raises | 抛出:
            ValueError: If parameters are out of valid range | 如果参数超出有效范围
        """
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon must be in [0,1]")
        
        if decay <= 0 or decay > 1:
            raise ValueError(f"decay must be in (0,1]")
        
        if not 0 <= min_epsilon <= 1:
            raise ValueError(f"min_epsilon must be in [0,1]")
        
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.step = 0
        self._logger = logging.getLogger("metathin.decision.EpsilonGreedyStrategy")
        self._selection_history: List[Dict] = []
    
    def select(self, 
               behaviors: List[MetaBehavior], 
               fitness_scores: List[float],
               features: FeatureVector) -> MetaBehavior:
        """
        Epsilon-greedy selection.
        
        ε-贪婪选择。
        
        Args | 参数:
            behaviors: List of candidate behaviors | 候选行为列表
            fitness_scores: Corresponding fitness scores | 对应的适应度分数
            features: Current feature vector | 当前特征向量
            
        Returns | 返回:
            MetaBehavior: Selected behavior | 选中的行为
        """
        if not behaviors:
            raise NoBehaviorError("No behaviors available")
        
        self.step += 1
        current_epsilon = self.epsilon
        
        # Decide between exploration and exploitation | 在探索和利用之间做决定
        if random.random() < current_epsilon:
            # Exploration: random selection | 探索：随机选择
            idx = random.randrange(len(behaviors))
            selected = behaviors[idx]
            mode = 'explore'
            self._logger.debug(f"Exploration mode: randomly selected {selected.name}")
        else:
            # Exploitation: select highest fitness | 利用：选择最高适应度
            idx = np.argmax(fitness_scores)
            selected = behaviors[idx]
            mode = 'exploit'
            self._logger.debug(f"Exploitation mode: selected optimal {selected.name} (fitness={fitness_scores[idx]:.3f})")
        
        # Decay epsilon | 衰减 epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        
        # Record decision | 记录决策
        self._selection_history.append({
            'step': self.step,
            'selected': selected.name,
            'mode': mode,
            'epsilon': current_epsilon,
            'fitness': fitness_scores[idx] if mode == 'exploit' else None
        })
        
        return selected
    
    def get_confidence(self, fitness_scores: List[float]) -> float:
        """
        Calculate decision confidence.
        
        计算决策置信度。
        
        Confidence = 1 - ε (current exploration rate)
        
        置信度 = 1 - ε（当前探索率）
        
        Args | 参数:
            fitness_scores: List of fitness scores | 适应度分数列表
            
        Returns | 返回:
            float: Confidence in [0,1] | [0,1] 范围内的置信度
        """
        return 1.0 - self.epsilon
    
    def get_exploration_rate(self) -> float:
        """Get current exploration rate | 获取当前探索率"""
        return self.epsilon
    
    def reset_epsilon(self, epsilon: float) -> None:
        """
        Reset exploration rate.
        
        重置探索率。
        
        Args | 参数:
            epsilon: New exploration rate | 新的探索率
        """
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon must be in [0,1]")
        self.epsilon = epsilon
        self.step = 0


# ============================================================
# 4. Round-Robin Strategy | 轮询策略
# ============================================================

class RoundRobinStrategy(DecisionStrategy):
    """
    Round-Robin Strategy: Cycles through behaviors in order.
    
    轮询策略：按顺序循环遍历行为。
    
    Ignores fitness scores, fairly selecting each behavior in sequence.
    Suitable for testing, load balancing, and uniform evaluation.
    
    忽略适应度分数，公平地按顺序选择每个行为。
    适用于测试、负载均衡和均匀评估。
    
    Characteristics | 特性:
        - Fair: Each behavior gets equal selection count | 公平：每个行为获得相等的选择次数
        - Simple: No parameters, easy to understand | 简单：无参数，易于理解
        - Predictable: Selection order is fixed | 可预测：选择顺序固定
    
    Attributes | 属性:
        current: Current index in the cycle | 当前在循环中的索引
        cycle: Number of completed cycles | 完成的循环次数
    """
    
    def __init__(self):
        """Initialize round-robin strategy | 初始化轮询策略"""
        self.current = 0
        self.cycle = 0
        self._logger = logging.getLogger("metathin.decision.RoundRobinStrategy")
        self._selection_history: List[str] = []
    
    def select(self, 
               behaviors: List[MetaBehavior], 
               fitness_scores: List[float],
               features: FeatureVector) -> MetaBehavior:
        """
        Round-robin selection.
        
        轮询选择。
        
        Args | 参数:
            behaviors: List of candidate behaviors | 候选行为列表
            fitness_scores: Corresponding fitness scores (ignored) | 对应的适应度分数（忽略）
            features: Current feature vector | 当前特征向量
            
        Returns | 返回:
            MetaBehavior: Selected behavior | 选中的行为
        """
        if not behaviors:
            raise NoBehaviorError("No behaviors available")
        
        # Select behavior at current index | 选择当前索引处的行为
        selected = behaviors[self.current]
        
        # Record selection | 记录选择
        self._selection_history.append(selected.name)
        
        self._logger.debug(f"Round-robin selection: {selected.name} (index={self.current})")
        
        # Update index | 更新索引
        self.current = (self.current + 1) % len(behaviors)
        if self.current == 0:
            self.cycle += 1
            self._logger.debug(f"Completed cycle {self.cycle}")
        
        return selected
    
    def get_cycle(self) -> int:
        """Get number of completed cycles | 获取完成的循环次数"""
        return self.cycle
    
    def reset(self) -> None:
        """Reset round-robin state | 重置轮询状态"""
        self.current = 0
        self.cycle = 0
        self._selection_history.clear()


# ============================================================
# 5. Random Strategy | 随机策略
# ============================================================

class RandomStrategy(DecisionStrategy):
    """
    Random Strategy: Completely random behavior selection.
    
    随机策略：完全随机选择行为。
    
    Ignores all fitness information, selects uniformly at random.
    Suitable for baseline comparison and pure exploration scenarios.
    
    忽略所有适应度信息，均匀随机选择。
    适用于基线比较和纯探索场景。
    
    Characteristics | 特性:
        - Purely random: All behaviors equally probable | 纯随机：所有行为等概率
        - Memoryless: Each selection independent | 无记忆：每次选择独立
        - Reproducible: Supports random seed | 可重现：支持随机种子
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random strategy.
        
        初始化随机策略。
        
        Args | 参数:
            seed: Random seed for reproducible experiments | 用于可重现实验的随机种子
        """
        import random
        import numpy as np
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self._random = random.Random(seed)
        else:
            self._random = random.Random()
        
        self.seed = seed
        self._logger = logging.getLogger("metathin.decision.RandomStrategy")
        self._selection_history: List[str] = []
    
    def select(self, 
               behaviors: List[MetaBehavior], 
               fitness_scores: List[float],
               features: FeatureVector) -> MetaBehavior:
        """
        Random selection.
        
        随机选择。
        
        Args | 参数:
            behaviors: List of candidate behaviors | 候选行为列表
            fitness_scores: Corresponding fitness scores (ignored) | 对应的适应度分数（忽略）
            features: Current feature vector | 当前特征向量
            
        Returns | 返回:
            MetaBehavior: Selected behavior | 选中的行为
        """
        if not behaviors:
            raise NoBehaviorError("No behaviors available")
        
        selected = self._random.choice(behaviors)
        
        # Record selection | 记录选择
        self._selection_history.append(selected.name)
        
        self._logger.debug(f"Random selection: {selected.name}")
        
        return selected
    
    def get_confidence(self, fitness_scores: List[float]) -> float:
        """
        Random strategy confidence is always 0.
        
        随机策略的置信度始终为 0。
        
        Args | 参数:
            fitness_scores: List of fitness scores | 适应度分数列表
            
        Returns | 返回:
            float: Always 0.0 | 始终为 0.0
        """
        return 0.0
class BoltzmannStrategy(DecisionStrategy):
    """
    Boltzmann Exploration: Probabilistic selection based on exponential weighting.
    
    玻尔兹曼探索：基于指数加权的概率选择。
    
    Uses Boltzmann distribution (softmax) for probabilistic selection,
    a generalization of the probabilistic strategy that handles extreme values better.
    
    使用玻尔兹曼分布（softmax）进行概率选择，
    是概率策略的推广，能更好地处理极端值。
    
    Characteristics | 特性:
        - Temperature control: Temperature adjusts exploration degree | 温度控制：温度调整探索程度
        - Probability smoothing: Avoids overly peaked distributions | 概率平滑：避免过于集中的分布
        - Theoretical foundation: Rooted in statistical mechanics | 理论基础：根植于统计力学
    
    Attributes | 属性:
        temperature: Temperature parameter controlling randomness | 控制随机性的温度参数
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize Boltzmann strategy.
        
        初始化玻尔兹曼策略。
        
        Args | 参数:
            temperature: Temperature parameter, must be > 0 | 温度参数，必须 > 0
            
        Raises | 抛出:
            ValueError: If temperature <= 0 | 如果 temperature <= 0
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        
        self.temperature = temperature
        self._logger = logging.getLogger("metathin.decision.BoltzmannStrategy")
        self._selection_history: List[Dict] = []
    
    def select(self, 
               behaviors: List[MetaBehavior], 
               fitness_scores: List[float],
               features: FeatureVector) -> MetaBehavior:
        """
        Boltzmann selection.
        
        玻尔兹曼选择。
        
        Args | 参数:
            behaviors: List of candidate behaviors | 候选行为列表
            fitness_scores: Corresponding fitness scores | 对应的适应度分数
            features: Current feature vector | 当前特征向量
            
        Returns | 返回:
            MetaBehavior: Selected behavior | 选中的行为
        """
        if not behaviors:
            raise NoBehaviorError("No behaviors available")
        
        # Calculate Boltzmann probabilities | 计算玻尔兹曼概率
        probs = self.get_probabilities(fitness_scores)
        
        self._logger.debug(f"Boltzmann probabilities: {dict(zip([b.name for b in behaviors], probs))}")
        
        # Random selection | 随机选择
        try:
            idx = np.random.choice(len(behaviors), p=probs)
            selected = behaviors[idx]
        except ValueError as e:
            self._logger.warning(f"Probabilistic selection failed ({e}), falling back to uniform random")
            selected = self._random.choice(behaviors)
            idx = behaviors.index(selected)
        
        # Record decision | 记录决策
        self._selection_history.append({
            'timestamp': time.time(),
            'selected': selected.name,
            'probability': float(probs[idx]),
            'temperature': self.temperature,
            'max_fitness': max(fitness_scores)
        })
        
        return selected
    
    def get_probabilities(self, fitness_scores: List[float]) -> List[float]:
        """
        Get selection probability for each behavior (Boltzmann distribution).
        
        获取每个行为的选择概率（玻尔兹曼分布）。
        
        Args | 参数:
            fitness_scores: List of fitness scores | 适应度分数列表
            
        Returns | 返回:
            List[float]: Probabilities summing to 1 | 总和为 1 的概率列表
        """
        if not fitness_scores:
            return []
        
        # Use softmax (Boltzmann distribution) | 使用 softmax（玻尔兹曼分布）
        probs = softmax(fitness_scores, self.temperature)
        
        return probs.tolist()
    
    def get_confidence(self, fitness_scores: List[float]) -> float:
        """
        Calculate decision confidence.
        
        计算决策置信度。
        
        Uses the difference between highest and second-highest probabilities.
        
        使用最高概率与第二高概率的差值。
        
        Args | 参数:
            fitness_scores: List of fitness scores | 适应度分数列表
            
        Returns | 返回:
            float: Confidence in [0,1] | [0,1] 范围内的置信度
        """
        probs = self.get_probabilities(fitness_scores)
        
        if len(probs) < 2:
            return 1.0
        
        sorted_probs = sorted(probs, reverse=True)
        confidence = sorted_probs[0] - sorted_probs[1]
        
        return min(1.0, max(0.0, confidence))


# ============================================================
# 7. Hybrid Strategy | 混合策略
# ============================================================

class HybridStrategy(DecisionStrategy):
    """
    Hybrid Strategy: Combines multiple strategies, switching based on conditions.
    
    混合策略：组合多种策略，根据条件切换。
    
    Dynamically switches between different strategies to adapt to various scenarios.
    
    在不同策略之间动态切换，以适应各种场景。
    
    Characteristics | 特性:
        - Flexible: Can choose different strategies based on conditions | 灵活：可根据条件选择不同策略
        - Extensible: Can combine any number of strategies | 可扩展：可组合任意数量的策略
        - Adaptive: Strategy selector can use context information | 自适应：策略选择器可使用上下文信息
    
    Attributes | 属性:
        strategies: List of component strategies | 组件策略列表
        selector: Function that selects which strategy to use | 选择使用哪个策略的函数
        step: Number of selections performed | 已执行的选择次数
    """
    
    def __init__(self,
                 strategies: List[DecisionStrategy],
                 selector: Callable[[int, Dict], int]):
        """
        Initialize hybrid strategy.
        
        初始化混合策略。
        
        Args | 参数:
            strategies: List of component strategies | 组件策略列表
            selector: Selection function, receives (step, context) and returns strategy index
                     选择函数，接收 (step, context) 并返回策略索引
            
        Raises | 抛出:
            ValueError: If strategies list is empty | 如果策略列表为空
        """
        if not strategies:
            raise ValueError("Strategies list cannot be empty")
        
        self.strategies = strategies
        self.selector = selector
        self.step = 0
        self._logger = logging.getLogger("metathin.decision.HybridStrategy")
        self._selection_history: List[Dict] = []
    
    def select(self, 
               behaviors: List[MetaBehavior], 
               fitness_scores: List[float],
               features: FeatureVector) -> MetaBehavior:
        """
        Select using currently chosen strategy.
        
        使用当前选择的策略进行选择。
        
        Args | 参数:
            behaviors: List of candidate behaviors | 候选行为列表
            fitness_scores: Corresponding fitness scores | 对应的适应度分数
            features: Current feature vector | 当前特征向量
            
        Returns | 返回:
            MetaBehavior: Selected behavior | 选中的行为
        """
        self.step += 1
        
        # Choose current strategy | 选择当前策略
        context = {
            'behaviors': behaviors,
            'fitness_scores': fitness_scores,
            'features': features,
            'step': self.step
        }
        
        try:
            strategy_idx = self.selector(self.step, context)
            if strategy_idx < 0 or strategy_idx >= len(self.strategies):
                self._logger.warning(f"Invalid strategy index {strategy_idx}, using default 0")
                strategy_idx = 0
        except Exception as e:
            self._logger.error(f"Strategy selection failed: {e}, using default 0")
            strategy_idx = 0
        
        strategy = self.strategies[strategy_idx]
        
        self._logger.debug(f"Using strategy {strategy_idx}: {type(strategy).__name__}")
        
        # Execute selection | 执行选择
        selected = strategy.select(behaviors, fitness_scores, features)
        
        # Record decision | 记录决策
        self._selection_history.append({
            'step': self.step,
            'strategy_idx': strategy_idx,
            'strategy_name': type(strategy).__name__,
            'selected': selected.name
        })
        
        return selected
    
    def get_confidence(self, fitness_scores: List[float]) -> float:
        """
        Get confidence from current strategy.
        
        从当前策略获取置信度。
        
        Simplified implementation using first strategy's confidence.
        
        使用第一个策略的置信度的简化实现。
        
        Args | 参数:
            fitness_scores: List of fitness scores | 适应度分数列表
            
        Returns | 返回:
            float: Confidence in [0,1] | [0,1] 范围内的置信度
        """
        if self.strategies:
            return self.strategies[0].get_confidence(fitness_scores)
        return 0.5


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    'MaxFitnessStrategy',
    'ProbabilisticStrategy',
    'EpsilonGreedyStrategy',
    'RoundRobinStrategy',
    'RandomStrategy',
    'BoltzmannStrategy',
    'HybridStrategy',
]