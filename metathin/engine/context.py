"""
Thinking Context - State Container | 思考上下文 - 状态容器
============================================================

Defines the context object that flows through the thinking pipeline.
This container holds all state information needed during a single think cycle.

定义在思考流水线中传递的上下文对象。
这个容器保存单次思考周期所需的所有状态信息。

Design Philosophy | 设计理念:
    - Immutable: Context is passed forward, never mutated in place
      不可变：上下文向前传递，从不原地修改
    - Explicit: All state is clearly typed and documented
      显式：所有状态都有清晰的类型和文档
    - Extensible: Can carry additional metadata
      可扩展：可以携带额外的元数据
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..core.types import FeatureVector


# ============================================================
# Thinking Context | 思考上下文
# ============================================================

@dataclass
class ThinkingContext:
    """
    Context object for a single think cycle.
    
    单次思考周期的上下文对象。
    
    This immutable container holds all state information that flows through
    the thinking pipeline. It is created at the start of think() and
    passed through each stage (Perceive → Hypothesize → Decide → Execute → Learn).
    
    这个不可变容器保存了思考流水线中传递的所有状态信息。
    它在 think() 开始时创建，并贯穿每个阶段（感知 → 假设 → 决策 → 执行 → 学习）。
    
    Attributes | 属性:
        raw_input: Original input data | 原始输入数据
        expected: Expected result (for learning) | 期望结果（用于学习）
        features: Extracted feature vector | 提取的特征向量
        selected_behavior: Name of selected behavior | 选中的行为名称
        result: Execution result | 执行结果
        fitness_scores: Fitness scores for all behaviors | 所有行为的适应度分数
        candidate_behaviors: List of candidate behavior names | 候选行为名称列表
        start_time: Start timestamp of thinking cycle | 思考周期开始时间戳
        stage_times: Time spent in each stage | 每个阶段花费的时间
        metadata: Additional user-defined metadata | 额外的用户定义元数据
    """
    
    # ===== Input | 输入 =====
    raw_input: Any
    """Original input data | 原始输入数据"""
    
    expected: Any = None
    """Expected result (for learning) | 期望结果（用于学习）"""
    
    # ===== Context parameters | 上下文参数 =====
    context_params: Dict[str, Any] = field(default_factory=dict)
    """Additional context parameters passed to think() | 传递给 think() 的额外上下文参数"""
    
    # ===== Perceive stage (P) | 感知阶段 =====
    features: Optional[FeatureVector] = None
    """Extracted feature vector | 提取的特征向量"""
    
    # ===== Hypothesize stage (S) | 假设阶段 =====
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    """Fitness scores for all behaviors | 所有行为的适应度分数"""
    
    candidate_behaviors: List[str] = field(default_factory=list)
    """List of candidate behavior names | 候选行为名称列表"""
    
    # ===== Decide stage (D) | 决策阶段 =====
    selected_behavior: Optional[str] = None
    """Name of selected behavior | 选中的行为名称"""
    
    decision_confidence: float = 0.0
    """Confidence of the decision (0-1) | 决策置信度 (0-1)"""
    
    # ===== Execute stage (B) | 执行阶段 =====
    result: Any = None
    """Execution result | 执行结果"""
    
    # ===== Learn stage (Ψ) | 学习阶段 =====
    learning_occurred: bool = False
    """Whether learning was performed | 是否执行了学习"""
    
    parameter_changes: Dict[str, float] = field(default_factory=dict)
    """Parameter changes applied during learning | 学习过程中应用的参数变化"""
    
    # ===== Timing | 计时 =====
    start_time: datetime = field(default_factory=datetime.now)
    """Start timestamp of thinking cycle | 思考周期开始时间戳"""
    
    stage_times: Dict[str, float] = field(default_factory=dict)
    """Time spent in each stage (seconds) | 每个阶段花费的时间（秒）"""
    
    # ===== Metadata | 元数据 =====
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional user-defined metadata | 额外的用户定义元数据"""
    
    # ===== Error handling | 错误处理 =====
    error: Optional[Exception] = None
    """Error that occurred during thinking | 思考过程中发生的错误"""
    
    def total_time(self) -> float:
        """
        Calculate total thinking time.
        
        计算总思考时间。
        
        Returns | 返回:
            float: Total time in seconds | 总时间（秒）
        """
        # 如果有 stage_times，优先使用它们
        if self.stage_times:
            return sum(self.stage_times.values())
        
        # 如果没有 stage_times 但有 start_time，计算时间差
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        
        return 0.0
    
    def to_thought_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for Thought record.
        
        转换为 Thought 记录的字典格式。
        
        Returns | 返回:
            Dict: Dictionary representation | 字典表示
        """
        return {
            'raw_input': self.raw_input,
            'expected': self.expected,
            'features': self.features.tolist() if self.features is not None else None,
            'selected_behavior': self.selected_behavior,
            'fitness_scores': self.fitness_scores.copy(),
            'candidate_behaviors': self.candidate_behaviors.copy(),
            'result': self.result,
            'success': self.error is None,
            'error_message': str(self.error) if self.error else None,
            'decision_time': self.stage_times.get('decide', 0.0),
            'execution_time': self.stage_times.get('execute', 0.0),
            'total_time': self.total_time(),
            'metadata': self.metadata.copy(),
        }
    
    def with_stage_time(self, stage: str, duration: float) -> 'ThinkingContext':
        """
        Create a new context with updated stage time.
        
        创建带有更新阶段时间的新上下文。
        
        This returns a new context (immutable pattern).
        
        返回新的上下文（不可变模式）。
        
        Args | 参数:
            stage: Stage name ('perceive', 'hypothesize', 'decide', 'execute', 'learn')
                  阶段名称
            duration: Time spent in this stage (seconds) | 该阶段花费的时间（秒）
            
        Returns | 返回:
            ThinkingContext: New context with updated stage times
                             带有更新阶段时间的新上下文
        """
        new_times = self.stage_times.copy()
        new_times[stage] = duration
        return ThinkingContext(
            raw_input=self.raw_input,
            expected=self.expected,
            context_params=self.context_params.copy(),
            features=self.features,
            fitness_scores=self.fitness_scores.copy(),
            candidate_behaviors=self.candidate_behaviors.copy(),
            selected_behavior=self.selected_behavior,
            decision_confidence=self.decision_confidence,
            result=self.result,
            learning_occurred=self.learning_occurred,
            parameter_changes=self.parameter_changes.copy(),
            start_time=self.start_time,
            stage_times=new_times,
            metadata=self.metadata.copy(),
            error=self.error,
        )
    
    def with_error(self, error: Exception, stage: str) -> 'ThinkingContext':
        """
        Create a new context with error information.
        
        创建带有错误信息的新上下文。
        
        Args | 参数:
            error: The exception that occurred | 发生的异常
            stage: Stage where error occurred | 发生错误的阶段
            
        Returns | 返回:
            ThinkingContext: New context with error set | 设置了错误的新上下文
        """
        new_times = self.stage_times.copy()
        return ThinkingContext(
            raw_input=self.raw_input,
            expected=self.expected,
            context_params=self.context_params.copy(),
            features=self.features,
            fitness_scores=self.fitness_scores.copy(),
            candidate_behaviors=self.candidate_behaviors.copy(),
            selected_behavior=self.selected_behavior,
            decision_confidence=self.decision_confidence,
            result=self.result,
            learning_occurred=self.learning_occurred,
            parameter_changes=self.parameter_changes.copy(),
            start_time=self.start_time,
            stage_times=new_times,
            metadata=self.metadata.copy(),
            error=error,
        )


# ============================================================
# Stage Result Types | 阶段结果类型
# ============================================================

@dataclass
class PerceiveResult:
    """Result of the perceive stage | 感知阶段结果"""
    features: FeatureVector
    duration: float


@dataclass
class HypothesizeResult:
    """Result of the hypothesize stage | 假设阶段结果"""
    candidates: List[str]  # Behavior names | 行为名称
    fitness_scores: Dict[str, float]
    duration: float


@dataclass
class DecideResult:
    """Result of the decide stage | 决策阶段结果"""
    selected_behavior: str
    confidence: float
    duration: float


@dataclass
class ExecuteResult:
    """Result of the execute stage | 执行阶段结果"""
    result: Any
    duration: float


@dataclass
class LearnResult:
    """Result of the learn stage | 学习阶段结果"""
    learning_occurred: bool
    parameter_changes: Dict[str, float]
    duration: float


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    'ThinkingContext',    # Main context container | 主上下文容器
    'PerceiveResult',     # Perceive stage result | 感知阶段结果
    'HypothesizeResult',  # Hypothesize stage result | 假设阶段结果
    'DecideResult',       # Decide stage result | 决策阶段结果
    'ExecuteResult',      # Execute stage result | 执行阶段结果
    'LearnResult',        # Learn stage result | 学习阶段结果
]