"""
Thinking Pipeline - Core Cognitive Cycle | 思考流水线 - 核心认知循环
====================================================================

Implements the complete cognitive cycle: P → B → S → D → Ψ.
This is the heart of the Metathin framework - a pure function that orchestrates
the five components to process input and produce output.

实现完整的认知循环：P → B → S → D → Ψ。
这是 Metathin 框架的核心——一个编排五个组件来处理输入并产生输出的纯函数。

Cognitive Cycle | 认知循环:
    1. Perceive (P): Extract features from raw input | 从原始输入提取特征
    2. Hypothesize (S): Compute fitness for each behavior | 计算每个行为的适应度
    3. Decide (D): Select optimal behavior | 选择最优行为
    4. Execute (B): Run the selected behavior | 运行选中的行为
    5. Learn (Ψ): Update parameters based on feedback | 基于反馈更新参数

Design Philosophy | 设计理念:
    - Pure function: No side effects, deterministic given inputs
      纯函数：无副作用，给定输入时确定性
    - Stateless: All state is passed via context
      无状态：所有状态通过上下文传递
    - Observable: Returns detailed execution results
      可观测：返回详细的执行结果
"""

from __future__ import annotations

import math
import numbers
import random
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

from ..core.types import FeatureVector
from ..core.p_pattern import PatternSpace
from ..core.b_behavior import MetaBehavior
from ..core.s_selector import Selector
from ..core.d_decision import DecisionStrategy
from ..core.psi_learning import LearningMechanism
from ..core.exceptions import (
    MetathinError,
    PatternExtractionError,
    BehaviorExecutionError,
    FitnessComputationError,
    DecisionError,
    NoBehaviorError,
    LearningError,
    ParameterUpdateError,
)

from .context import ThinkingContext, PerceiveResult, HypothesizeResult, DecideResult, ExecuteResult, LearnResult


# ============================================================
# Pipeline Result | 流水线结果
# ============================================================

@dataclass
class PipelineResult:
    """
    Result of running the thinking pipeline.

    运行思考流水线的结果。

    Attributes | 属性:
        success: Whether the pipeline completed successfully | 流水线是否成功完成
        result: Execution result (if successful) | 执行结果（如果成功）
        selected_behavior: Name of the selected behavior | 选中的行为名称
        fitness_scores: Fitness scores for all candidates | 所有候选行为的适应度分数
        stage_times: Time spent in each stage | 每个阶段花费的时间
        error: Error that occurred (if any) | 发生的错误（如果有）
        context: Final context after pipeline execution | 流水线执行后的最终上下文
    """
    success: bool
    result: Any
    selected_behavior: Optional[str]
    fitness_scores: Dict[str, float]
    stage_times: Dict[str, float]
    error: Optional[Exception]
    context: ThinkingContext


# ============================================================
# Thinking Pipeline (Pure Function) | 思考流水线（纯函数）
# ============================================================

class ThinkingPipeline:
    """
    Pure function pipeline for the cognitive cycle.

    认知循环的纯函数流水线。

    This class encapsulates the thinking logic but maintains no state.
    All state is passed via the ThinkingContext.

    这个类封装了思考逻辑，但不维护任何状态。
    所有状态都通过 ThinkingContext 传递。

    Usage | 使用:
        >>> pipeline = ThinkingPipeline()
        >>> result = pipeline.run(
        ...     raw_input="hello",
        ...     pattern_space=my_pattern,
        ...     behaviors=[greet, echo],
        ...     selector=my_selector,
        ...     decision_strategy=my_strategy,
        ...     learning_mechanism=my_learner,
        ... )
        >>> print(result.result)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline with configuration.

        使用配置初始化流水线。

        Args | 参数:
            config: Pipeline configuration (min_fitness_threshold, enable_learning, etc.)
                    流水线配置（min_fitness_threshold、enable_learning 等）
        """
        self.config = config or {}
        self.min_fitness_threshold = self.config.get('min_fitness_threshold', 0.0)
        self.enable_learning = self.config.get('enable_learning', True)

    def run(
        self,
        raw_input: Any,
        pattern_space: PatternSpace,
        behaviors: List[MetaBehavior],
        selector: Selector,
        decision_strategy: DecisionStrategy,
        learning_mechanism: Optional[LearningMechanism] = None,
        expected: Any = None,
        context_params: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Execute the complete thinking pipeline.

        执行完整的思考流水线。

        This is the main entry point - orchestrates all five stages.

        这是主入口——编排所有五个阶段。

        Args | 参数:
            raw_input: Raw input data | 原始输入数据
            pattern_space: Pattern space for feature extraction (P) | 特征提取的模式空间 (P)
            behaviors: List of available behaviors (B) | 可用行为列表 (B)
            selector: Selector for fitness computation (S) | 计算适应度的选择器 (S)
            decision_strategy: Decision strategy for behavior selection (D) | 选择行为的决策策略 (D)
            learning_mechanism: Learning mechanism for parameter updates (Ψ, optional)
                                参数更新的学习机制 (Ψ，可选)
            expected: Expected result (for learning) | 期望结果（用于学习）
            context_params: Additional context parameters | 额外的上下文参数

        Returns | 返回:
            PipelineResult: Result containing output, timing, and status
                            包含输出、计时和状态的结果
        """
        # Create initial context | 创建初始上下文
        context = ThinkingContext(
            raw_input=raw_input,
            expected=expected,
            context_params=context_params or {},
        )

        # Validate contracts before touching any component | 调用任何组件前先校验契约
        try:
            self._validate_contracts(
                pattern_space, behaviors, selector, decision_strategy, learning_mechanism
            )
        except Exception as exc:  # noqa: BLE001 - convert to a pipeline failure
            error = exc if isinstance(exc, MetathinError) else MetathinError(
                f"component contract validation failed: {exc}"
            )
            return PipelineResult(
                success=False,
                result=None,
                selected_behavior=None,
                fitness_scores={},
                stage_times=context.stage_times,
                error=error,
                context=context.with_error(error, 'contract'),
            )

        # ============================================================
        # Stage 1: Perceive (P) | 阶段1：感知
        # ============================================================
        try:
            perceive_start = time.time()
            features = pattern_space.extract(raw_input)
            if not pattern_space.validate_features(features):
                raise PatternExtractionError(
                    "PatternSpace.extract() must return a 1-D float64 numpy array "
                    "with finite values"
                )
            perceive_duration = time.time() - perceive_start
            context = context.with_stage_time('perceive', perceive_duration)

            # Update context with features | 用特征更新上下文
            context = ThinkingContext(
                raw_input=context.raw_input,
                expected=context.expected,
                context_params=context.context_params,
                features=features,
                fitness_scores=context.fitness_scores,
                candidate_behaviors=context.candidate_behaviors,
                selected_behavior=context.selected_behavior,
                decision_confidence=context.decision_confidence,
                result=context.result,
                learning_occurred=context.learning_occurred,
                parameter_changes=context.parameter_changes,
                start_time=context.start_time,
                stage_times=context.stage_times,
                metadata=context.metadata,
                error=context.error,
            )
        except Exception as e:
            return PipelineResult(
                success=False,
                result=None,
                selected_behavior=None,
                fitness_scores={},
                stage_times=context.stage_times,
                error=PatternExtractionError(f"Feature extraction failed: {e}"),
                context=context.with_error(e, 'perceive'),
            )

        # ============================================================
        # Stage 2: Hypothesize (S) | 阶段2：假设
        # ============================================================
        try:
            hypothesize_start = time.time()
            candidates, fitness_scores = self._hypothesize(
                behaviors, selector, features
            )
            hypothesize_duration = time.time() - hypothesize_start
            context = context.with_stage_time('hypothesize', hypothesize_duration)

            # Update context with fitness scores | 用适应度分数更新上下文
            context = ThinkingContext(
                raw_input=context.raw_input,
                expected=context.expected,
                context_params=context.context_params,
                features=context.features,
                fitness_scores=fitness_scores,
                candidate_behaviors=[b.name for b in candidates],
                selected_behavior=context.selected_behavior,
                decision_confidence=context.decision_confidence,
                result=context.result,
                learning_occurred=context.learning_occurred,
                parameter_changes=context.parameter_changes,
                start_time=context.start_time,
                stage_times=context.stage_times,
                metadata=context.metadata,
                error=context.error,
            )
        except Exception as e:
            return PipelineResult(
                success=False,
                result=None,
                selected_behavior=None,
                fitness_scores={},
                stage_times=context.stage_times,
                error=e,
                context=context.with_error(e, 'hypothesize'),
            )

        # Check if any behaviors are available | 检查是否有可用行为
        if not candidates:
            return PipelineResult(
                success=False,
                result=None,
                selected_behavior=None,
                fitness_scores=fitness_scores,
                stage_times=context.stage_times,
                error=NoBehaviorError("No behaviors available for current state"),
                context=context,
            )

        # ============================================================
        # Stage 3: Decide (D) | 阶段3：决策
        # ============================================================
        try:
            decide_start = time.time()
            explore = False
            exploration_rate = 0.0
            if (
                expected is not None
                and self.enable_learning
                and learning_mechanism is not None
                and hasattr(learning_mechanism, 'should_explore')
            ):
                explore = learning_mechanism.should_explore(
                    [fitness_scores[b.name] for b in candidates]
                )
                exploration_rate = getattr(
                    learning_mechanism, 'get_exploration_rate', lambda: 0.0
                )()

            if explore:
                # Probabilistic exploration: bypass the greedy strategy.
                selected = random.choice(candidates)
                confidence = 1.0 - exploration_rate
            else:
                selected = decision_strategy.select(
                    candidates,
                    [fitness_scores[b.name] for b in candidates],
                    features
                )
                confidence = decision_strategy.get_confidence(
                    [fitness_scores[b.name] for b in candidates]
                )

            if not isinstance(selected, MetaBehavior) or not any(
                selected is candidate for candidate in candidates
            ):
                raise DecisionError(
                    "DecisionStrategy.select() must return one of the candidate behaviors"
                )
            decide_duration = time.time() - decide_start
            context = context.with_stage_time('decide', decide_duration)

            # Update context with selected behavior | 用选中的行为更新上下文
            context = ThinkingContext(
                raw_input=context.raw_input,
                expected=context.expected,
                context_params=context.context_params,
                features=context.features,
                fitness_scores=context.fitness_scores,
                candidate_behaviors=context.candidate_behaviors,
                selected_behavior=selected.name,
                decision_confidence=confidence,
                result=context.result,
                learning_occurred=context.learning_occurred,
                parameter_changes=context.parameter_changes,
                start_time=context.start_time,
                stage_times=context.stage_times,
                metadata=context.metadata,
                error=context.error,
            )
        except Exception as e:
            return PipelineResult(
                success=False,
                result=None,
                selected_behavior=None,
                fitness_scores=fitness_scores,
                stage_times=context.stage_times,
                error=DecisionError(f"Decision failed: {e}"),
                context=context.with_error(e, 'decide'),
            )

        # ============================================================
        # Stage 4: Execute (B) | 阶段4：执行
        # ============================================================
        try:
            execute_start = time.time()
            result = selected.execute(features, **context_params or {})
            execute_duration = time.time() - execute_start
            context = context.with_stage_time('execute', execute_duration)

            # Update context with execution result | 用执行结果更新上下文
            context = ThinkingContext(
                raw_input=context.raw_input,
                expected=context.expected,
                context_params=context.context_params,
                features=context.features,
                fitness_scores=context.fitness_scores,
                candidate_behaviors=context.candidate_behaviors,
                selected_behavior=context.selected_behavior,
                decision_confidence=context.decision_confidence,
                result=result,
                learning_occurred=context.learning_occurred,
                parameter_changes=context.parameter_changes,
                start_time=context.start_time,
                stage_times=context.stage_times,
                metadata=context.metadata,
                error=context.error,
            )
        except Exception as e:
            return PipelineResult(
                success=False,
                result=None,
                selected_behavior=selected.name,
                fitness_scores=fitness_scores,
                stage_times=context.stage_times,
                error=BehaviorExecutionError(f"Behavior execution failed: {e}"),
                context=context.with_error(e, 'execute'),
            )

        # ============================================================
        # Stage 5: Learn (Ψ) | 阶段5：学习
        # ============================================================
        if learning_mechanism and self.enable_learning and expected is not None:
            try:
                learn_start = time.time()
                learning_occurred, param_changes = self._learn(
                    learning_mechanism,
                    selector,
                    expected,
                    result,
                    features,
                    selected.name,
                    reward=context_params.get('reward') if context_params else None,
                )
                learn_duration = time.time() - learn_start
                context = context.with_stage_time('learn', learn_duration)

                # Update context with learning results | 用学习结果更新上下文
                context = ThinkingContext(
                    raw_input=context.raw_input,
                    expected=context.expected,
                    context_params=context.context_params,
                    features=context.features,
                    fitness_scores=context.fitness_scores,
                    candidate_behaviors=context.candidate_behaviors,
                    selected_behavior=context.selected_behavior,
                    decision_confidence=context.decision_confidence,
                    result=context.result,
                    learning_occurred=learning_occurred,
                    parameter_changes=param_changes,
                    start_time=context.start_time,
                    stage_times=context.stage_times,
                    metadata=context.metadata,
                    error=context.error,
                )
            except Exception as e:
                # Learning failure doesn't fail the entire pipeline | 学习失败不会导致整个流水线失败
                # Just record the error in context | 只在上下文中记录错误
                context = context.with_error(e, 'learn')

        # ============================================================
        # Return successful result | 返回成功结果
        # ============================================================
        return PipelineResult(
            success=True,
            result=result,
            selected_behavior=selected.name,
            fitness_scores=fitness_scores,
            stage_times=context.stage_times,
            error=None,
            context=context,
        )

    def _validate_contracts(
        self,
        pattern_space: PatternSpace,
        behaviors: List[MetaBehavior],
        selector: Selector,
        decision_strategy: DecisionStrategy,
        learning_mechanism: Optional[LearningMechanism],
    ) -> None:
        """Validate the component contracts before running the pipeline."""
        if not isinstance(pattern_space, PatternSpace):
            raise MetathinError(
                f"pattern_space must implement PatternSpace, got {type(pattern_space)}"
            )
        if not isinstance(selector, Selector):
            raise MetathinError(
                f"selector must implement Selector, got {type(selector)}"
            )
        if not isinstance(decision_strategy, DecisionStrategy):
            raise MetathinError(
                "decision_strategy must implement DecisionStrategy, "
                f"got {type(decision_strategy)}"
            )
        if learning_mechanism is not None and not isinstance(
            learning_mechanism, LearningMechanism
        ):
            raise MetathinError(
                "learning_mechanism must implement LearningMechanism, "
                f"got {type(learning_mechanism)}"
            )
        if not isinstance(behaviors, (list, tuple)):
            raise MetathinError(
                f"behaviors must be a list or tuple, got {type(behaviors)}"
            )

        names = []
        for behavior in behaviors:
            if not isinstance(behavior, MetaBehavior):
                raise MetathinError(
                    f"every behavior must implement MetaBehavior, got {type(behavior)}"
                )
            if not isinstance(behavior.name, str) or not behavior.name:
                raise MetathinError("every behavior must have a non-empty string name")
            names.append(behavior.name)
        if len(names) != len(set(names)):
            raise MetathinError("behavior names must be unique")

    def _validate_fitness(self, behavior_name: str, fitness: Any) -> float:
        """Validate and normalize a fitness score."""
        if isinstance(fitness, bool) or not isinstance(fitness, numbers.Real):
            raise FitnessComputationError(
                f"Selector returned a non-numeric fitness for '{behavior_name}': "
                f"{fitness!r}"
            )
        value = float(fitness)
        if not math.isfinite(value):
            raise FitnessComputationError(
                f"Selector returned a non-finite fitness for '{behavior_name}': {value!r}"
            )
        if not 0.0 <= value <= 1.0:
            raise FitnessComputationError(
                f"Fitness for '{behavior_name}' must be in [0, 1], got {value!r}"
            )
        return value

    def _validate_parameters(self, selector: Selector) -> Dict[str, float]:
        """Read and validate selector parameters."""
        parameters = selector.get_parameters()
        return self._validate_parameter_mapping(parameters, "selector parameters")

    def _validate_parameter_mapping(
        self, mapping: Any, label: str
    ) -> Dict[str, float]:
        """Validate a parameter mapping returned by a component."""
        if not isinstance(mapping, dict):
            raise ParameterUpdateError(f"{label} must be a dict, got {type(mapping)}")
        normalized: Dict[str, float] = {}
        for key, value in mapping.items():
            if not isinstance(key, str) or not key:
                raise ParameterUpdateError(
                    f"{label} keys must be non-empty strings, got {key!r}"
                )
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ParameterUpdateError(
                    f"{label}['{key}'] must be numeric, got {value!r}"
                )
            number = float(value)
            if not math.isfinite(number):
                raise ParameterUpdateError(
                    f"{label}['{key}'] must be finite, got {number!r}"
                )
            normalized[key] = number
        return normalized

    def _validate_adjustment(
        self, selector: Selector, adjustment: Any
    ) -> Optional[Dict[str, float]]:
        """Validate an adjustment dictionary before applying it."""
        if adjustment is None:
            return None
        if not isinstance(adjustment, dict):
            raise LearningError(
                f"compute_adjustment() must return a dict or None, got {type(adjustment)}"
            )
        normalized = self._validate_parameter_mapping(adjustment, "adjustment")
        known = set(selector.get_parameters().keys())
        unknown = sorted(set(normalized) - known)
        if unknown:
            raise LearningError(
                f"compute_adjustment() returned unknown selector keys: {unknown}"
            )
        return normalized


    def _hypothesize(
        self,
        behaviors: List[MetaBehavior],
        selector: Selector,
        features: FeatureVector,
    ) -> tuple[List[MetaBehavior], Dict[str, float]]:
        """
        Hypothesize stage: compute fitness for all behaviors.

        假设阶段：计算所有行为的适应度。

        Args | 参数:
            behaviors: List of available behaviors | 可用行为列表
            selector: Selector for fitness computation | 计算适应度的选择器
            features: Feature vector | 特征向量

        Returns | 返回:
            tuple: (candidate_behaviors, fitness_scores)
                   (候选行为列表, 适应度分数字典)
        """
        candidates = []
        fitness_scores = {}

        for behavior in behaviors:
            # Check if behavior can execute | 检查行为是否可执行
            if not behavior.can_execute(features):
                continue

            # Compute fitness | 计算适应度
            fitness = selector.compute_fitness(behavior, features)
            fitness = self._validate_fitness(behavior.name, fitness)

            # Record fitness for analysis | 记录适应度用于分析
            selector.record_fitness(behavior.name, fitness)
            fitness_scores[behavior.name] = fitness

            # Check against threshold | 检查阈值
            if fitness >= self.min_fitness_threshold:
                candidates.append(behavior)

        return candidates, fitness_scores

    def _learn(
        self,
        learning_mechanism: LearningMechanism,
        selector: Selector,
        expected: Any,
        actual: Any,
        features: FeatureVector,
        behavior_name: str,
        reward: Optional[float] = None,
    ) -> tuple[bool, Dict[str, float]]:
        """
        Learn stage: update selector parameters based on feedback.

        学习阶段：基于反馈更新选择器参数。

        Args | 参数:
            learning_mechanism: Learning mechanism | 学习机制
            selector: Selector to update | 要更新的选择器
            expected: Expected result | 期望结果
            actual: Actual result | 实际结果
            features: Feature vector | 特征向量
            behavior_name: Name of executed behavior | 执行的行为名称

        Returns | 返回:
            tuple: (learning_occurred, parameter_changes)
                   (是否发生学习, 参数变化字典)
        """
        # Check if learning should occur | 检查是否应该学习
        if not learning_mechanism.should_learn(expected, actual):
            return False, {}

        # Build context for learning | 构建学习上下文
        behavior_index = None
        expected_behavior_index = None
        mapping = getattr(selector, '_behavior_indices', None)
        if isinstance(mapping, dict):
            behavior_index = mapping.get(behavior_name)
            if expected is not None:
                expected_behavior_index = mapping.get(expected)
                if expected_behavior_index is None:
                    expected_behavior_index = mapping.get(str(expected))

        context = {
            'features': features,
            'raw_features': features,
            'behavior_name': behavior_name,
            'behavior_index': behavior_index,
            'expected_behavior_index': expected_behavior_index,
            'parameter_prefix': getattr(selector, 'parameter_prefix', 'w_'),
            'parameters': self._validate_parameters(selector),
            'expected': expected,
            'actual': actual,
            'reward': reward,
        }

        get_learning_features = getattr(selector, 'get_learning_features', None)
        if callable(get_learning_features):
            learning_features = get_learning_features(features)
            if not (
                isinstance(learning_features, np.ndarray)
                and learning_features.ndim == 1
                and learning_features.dtype == np.float64
                and learning_features.size > 0
                and np.all(np.isfinite(learning_features))
            ):
                raise LearningError(
                    "Selector.get_learning_features() must return a non-empty 1-D "
                    "float64 numpy array with finite values"
                )
            context['features'] = learning_features

        # Compute adjustment | 计算调整量
        adjustment = self._validate_adjustment(
            selector,
            learning_mechanism.compute_adjustment(expected, actual, context),
        )

        # Apply adjustment if any | 如果有调整量则应用
        if adjustment:
            selector.update_parameters(adjustment)

        # Optional post-update (e.g. weight normalization)
        if hasattr(learning_mechanism, 'post_update'):
            mid = self._validate_parameters(selector)
            post_result = learning_mechanism.post_update(mid.copy())
            after = (
                {}
                if post_result is None
                else self._validate_parameter_mapping(
                    post_result, "post_update result"
                )
            )
            if after:
                delta = {
                    k: after[k] - mid.get(k, 0.0)
                    for k in after
                    if abs(after[k] - mid.get(k, 0.0)) > 1e-12
                }
                if delta:
                    selector.update_parameters(delta)

        # Persist experience for memory/replay learners
        if hasattr(learning_mechanism, 'record_experience'):
            learning_mechanism.record_experience(
                features, behavior_name, expected, actual, reward or 0.0
            )

        if adjustment:
            return True, adjustment
        return False, {}


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    'ThinkingPipeline',   # Main pipeline class | 主流水线类
    'PipelineResult',     # Pipeline execution result | 流水线执行结果
]
