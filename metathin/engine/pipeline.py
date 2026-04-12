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

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..core.types import FeatureVector
from ..core.p_pattern import PatternSpace
from ..core.b_behavior import MetaBehavior
from ..core.s_selector import Selector
from ..core.d_decision import DecisionStrategy
from ..core.psi_learning import LearningMechanism
from ..core.exceptions import (
    PatternExtractionError,
    BehaviorExecutionError,
    FitnessComputationError,
    DecisionError,
    NoBehaviorError,
    LearningError,
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
        
        # ============================================================
        # Stage 1: Perceive (P) | 阶段1：感知
        # ============================================================
        try:
            perceive_start = time.time()
            features = pattern_space.extract(raw_input)
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
            selected = decision_strategy.select(
                candidates, 
                [fitness_scores[b.name] for b in candidates],
                features
            )
            confidence = decision_strategy.get_confidence(
                [fitness_scores[b.name] for b in candidates]
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
        context = {
            'features': features,
            'behavior_name': behavior_name,
            'parameters': selector.get_parameters(),
            'expected': expected,
            'actual': actual,
        }
        
        # Compute adjustment | 计算调整量
        adjustment = learning_mechanism.compute_adjustment(expected, actual, context)
        
        # Apply adjustment if any | 如果有调整量则应用
        if adjustment:
            selector.update_parameters(adjustment)
            return True, adjustment
        
        return False, {}


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    'ThinkingPipeline',   # Main pipeline class | 主流水线类
    'PipelineResult',     # Pipeline execution result | 流水线执行结果
]