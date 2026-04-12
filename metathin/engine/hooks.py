"""
Thinking Hooks - Extension Points | 思考钩子 - 扩展点
=======================================================

Defines hook interfaces for extending the thinking pipeline.
Hooks allow users to inject custom behavior at various points in the cognitive cycle.

定义用于扩展思考流水线的钩子接口。
钩子允许用户在认知循环的各个点注入自定义行为。

Design Philosophy | 设计理念:
    - Non-intrusive: Hooks are optional, pipeline works without them
      非侵入式：钩子是可选的，流水线在没有钩子时也能工作
    - Composable: Multiple hooks can be chained
      可组合：多个钩子可以串联
    - Observable: Hooks can log, monitor, or modify the pipeline
      可观测：钩子可以记录、监控或修改流水线
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .context import ThinkingContext
from ..core.types import FeatureVector
from ..core.b_behavior import MetaBehavior


# ============================================================
# Hook Base Classes | 钩子基类
# ============================================================

class BeforePerceiveHook(ABC):
    """
    Hook called before feature extraction.
    
    在特征提取之前调用的钩子。
    
    Can be used for input validation, preprocessing, or logging.
    
    可用于输入验证、预处理或日志记录。
    """
    
    @abstractmethod
    def on_before_perceive(self, context: ThinkingContext) -> ThinkingContext:
        """
        Called before the perceive stage.
        
        在感知阶段之前调用。
        
        Args | 参数:
            context: Current thinking context | 当前思考上下文
            
        Returns | 返回:
            ThinkingContext: Modified context (or same if unchanged)
                             修改后的上下文（或未更改的原上下文）
        """
        pass


class AfterPerceiveHook(ABC):
    """
    Hook called after feature extraction.
    
    在特征提取之后调用的钩子。
    
    Can be used for feature validation, transformation, or logging.
    
    可用于特征验证、转换或日志记录。
    """
    
    @abstractmethod
    def on_after_perceive(self, context: ThinkingContext, features: FeatureVector) -> ThinkingContext:
        """
        Called after the perceive stage.
        
        在感知阶段之后调用。
        
        Args | 参数:
            context: Current thinking context | 当前思考上下文
            features: Extracted feature vector | 提取的特征向量
            
        Returns | 返回:
            ThinkingContext: Modified context | 修改后的上下文
        """
        pass


class BeforeHypothesizeHook(ABC):
    """
    Hook called before fitness computation.
    
    在适应度计算之前调用的钩子。
    
    Can be used to filter behaviors or modify features.
    
    可用于过滤行为或修改特征。
    """
    
    @abstractmethod
    def on_before_hypothesize(
        self, 
        context: ThinkingContext, 
        behaviors: List[MetaBehavior]
    ) -> tuple[ThinkingContext, List[MetaBehavior]]:
        """
        Called before the hypothesize stage.
        
        在假设阶段之前调用。
        
        Args | 参数:
            context: Current thinking context | 当前思考上下文
            behaviors: List of available behaviors | 可用行为列表
            
        Returns | 返回:
            tuple: (Modified context, filtered/modified behaviors)
                   (修改后的上下文, 过滤/修改后的行为列表)
        """
        pass


class AfterHypothesizeHook(ABC):
    """
    Hook called after fitness computation.
    
    在适应度计算之后调用的钩子。
    
    Can be used to adjust fitness scores or log results.
    
    可用于调整适应度分数或记录结果。
    """
    
    @abstractmethod
    def on_after_hypothesize(
        self,
        context: ThinkingContext,
        fitness_scores: Dict[str, float]
    ) -> ThinkingContext:
        """
        Called after the hypothesize stage.
        
        在假设阶段之后调用。
        
        Args | 参数:
            context: Current thinking context | 当前思考上下文
            fitness_scores: Computed fitness scores | 计算出的适应度分数
            
        Returns | 返回:
            ThinkingContext: Modified context | 修改后的上下文
        """
        pass


class BeforeDecideHook(ABC):
    """
    Hook called before behavior selection.
    
    在行为选择之前调用的钩子。
    
    Can be used to modify fitness scores or add constraints.
    
    可用于修改适应度分数或添加约束。
    """
    
    @abstractmethod
    def on_before_decide(
        self,
        context: ThinkingContext,
        candidates: List[MetaBehavior],
        fitness_scores: List[float]
    ) -> tuple[ThinkingContext, List[MetaBehavior], List[float]]:
        """
        Called before the decide stage.
        
        在决策阶段之前调用。
        
        Args | 参数:
            context: Current thinking context | 当前思考上下文
            candidates: List of candidate behaviors | 候选行为列表
            fitness_scores: Corresponding fitness scores | 对应的适应度分数
            
        Returns | 返回:
            tuple: (Modified context, modified candidates, modified fitness scores)
                   (修改后的上下文, 修改后的候选列表, 修改后的适应度分数)
        """
        pass


class AfterDecideHook(ABC):
    """
    Hook called after behavior selection.
    
    在行为选择之后调用的钩子。
    
    Can be used to validate selection or log decisions.
    
    可用于验证选择或记录决策。
    """
    
    @abstractmethod
    def on_after_decide(
        self,
        context: ThinkingContext,
        selected: MetaBehavior
    ) -> ThinkingContext:
        """
        Called after the decide stage.
        
        在决策阶段之后调用。
        
        Args | 参数:
            context: Current thinking context | 当前思考上下文
            selected: Selected behavior | 选中的行为
            
        Returns | 返回:
            ThinkingContext: Modified context | 修改后的上下文
        """
        pass


class BeforeExecuteHook(ABC):
    """
    Hook called before behavior execution.
    
    在行为执行之前调用的钩子。
    
    Can be used for resource preparation or parameter injection.
    
    可用于资源准备或参数注入。
    """
    
    @abstractmethod
    def on_before_execute(
        self,
        context: ThinkingContext,
        behavior: MetaBehavior
    ) -> ThinkingContext:
        """
        Called before the execute stage.
        
        在执行阶段之前调用。
        
        Args | 参数:
            context: Current thinking context | 当前思考上下文
            behavior: Behavior to execute | 要执行的行为
            
        Returns | 返回:
            ThinkingContext: Modified context | 修改后的上下文
        """
        pass


class AfterExecuteHook(ABC):
    """
    Hook called after behavior execution.
    
    在行为执行之后调用的钩子。
    
    Can be used for result validation, caching, or logging.
    
    可用于结果验证、缓存或日志记录。
    """
    
    @abstractmethod
    def on_after_execute(
        self,
        context: ThinkingContext,
        result: Any
    ) -> ThinkingContext:
        """
        Called after the execute stage.
        
        在执行阶段之后调用。
        
        Args | 参数:
            context: Current thinking context | 当前思考上下文
            result: Execution result | 执行结果
            
        Returns | 返回:
            ThinkingContext: Modified context | 修改后的上下文
        """
        pass


class BeforeLearnHook(ABC):
    """
    Hook called before learning.
    
    在学习之前调用的钩子。
    
    Can be used to modify expected/actual values or skip learning.
    
    可用于修改期望/实际值或跳过学习。
    """
    
    @abstractmethod
    def on_before_learn(
        self,
        context: ThinkingContext,
        expected: Any,
        actual: Any
    ) -> tuple[ThinkingContext, Any, Any]:
        """
        Called before the learn stage.
        
        在学习阶段之前调用。
        
        Args | 参数:
            context: Current thinking context | 当前思考上下文
            expected: Expected result | 期望结果
            actual: Actual result | 实际结果
            
        Returns | 返回:
            tuple: (Modified context, modified expected, modified actual)
                   (修改后的上下文, 修改后的期望值, 修改后的实际值)
        """
        pass


class AfterLearnHook(ABC):
    """
    Hook called after learning.
    
    在学习之后调用的钩子。
    
    Can be used to validate parameter updates or log learning events.
    
    可用于验证参数更新或记录学习事件。
    """
    
    @abstractmethod
    def on_after_learn(
        self,
        context: ThinkingContext,
        parameter_changes: Dict[str, float]
    ) -> ThinkingContext:
        """
        Called after the learn stage.
        
        在学习阶段之后调用。
        
        Args | 参数:
            context: Current thinking context | 当前思考上下文
            parameter_changes: Applied parameter changes | 应用的参数变化
            
        Returns | 返回:
            ThinkingContext: Modified context | 修改后的上下文
        """
        pass


class OnErrorHook(ABC):
    """
    Hook called when an error occurs.
    
    发生错误时调用的钩子。
    
    Can be used for error recovery, logging, or fallback behavior.
    
    可用于错误恢复、日志记录或回退行为。
    """
    
    @abstractmethod
    def on_error(
        self,
        context: ThinkingContext,
        error: Exception,
        stage: str
    ) -> ThinkingContext:
        """
        Called when an error occurs in any stage.
        
        在任何阶段发生错误时调用。
        
        Args | 参数:
            context: Current thinking context | 当前思考上下文
            error: The error that occurred | 发生的错误
            stage: Stage where error occurred | 发生错误的阶段
            
        Returns | 返回:
            ThinkingContext: Modified context (possibly with fallback values)
                             修改后的上下文（可能带有回退值）
        """
        pass


# ============================================================
# Hook Manager | 钩子管理器
# ============================================================

@dataclass
class HookManager:
    """
    Manages and orchestrates multiple hooks.
    
    管理和编排多个钩子。
    
    This class allows registering multiple hooks of the same type and
    executes them in order.
    
    这个类允许注册多个同类型的钩子，并按顺序执行它们。
    """
    
    # Before stage hooks | 阶段前钩子
    before_perceive_hooks: List[BeforePerceiveHook] = field(default_factory=list)
    before_hypothesize_hooks: List[BeforeHypothesizeHook] = field(default_factory=list)
    before_decide_hooks: List[BeforeDecideHook] = field(default_factory=list)
    before_execute_hooks: List[BeforeExecuteHook] = field(default_factory=list)
    before_learn_hooks: List[BeforeLearnHook] = field(default_factory=list)
    
    # After stage hooks | 阶段后钩子
    after_perceive_hooks: List[AfterPerceiveHook] = field(default_factory=list)
    after_hypothesize_hooks: List[AfterHypothesizeHook] = field(default_factory=list)
    after_decide_hooks: List[AfterDecideHook] = field(default_factory=list)
    after_execute_hooks: List[AfterExecuteHook] = field(default_factory=list)
    after_learn_hooks: List[AfterLearnHook] = field(default_factory=list)
    
    # Error hooks | 错误钩子
    error_hooks: List[OnErrorHook] = field(default_factory=list)
    
    def register_before_perceive(self, hook: BeforePerceiveHook) -> 'HookManager':
        """Register a before-perceive hook."""
        self.before_perceive_hooks.append(hook)
        return self
    
    def register_after_perceive(self, hook: AfterPerceiveHook) -> 'HookManager':
        """Register an after-perceive hook."""
        self.after_perceive_hooks.append(hook)
        return self
    
    def register_before_hypothesize(self, hook: BeforeHypothesizeHook) -> 'HookManager':
        """Register a before-hypothesize hook."""
        self.before_hypothesize_hooks.append(hook)
        return self
    
    def register_after_hypothesize(self, hook: AfterHypothesizeHook) -> 'HookManager':
        """Register an after-hypothesize hook."""
        self.after_hypothesize_hooks.append(hook)
        return self
    
    def register_before_decide(self, hook: BeforeDecideHook) -> 'HookManager':
        """Register a before-decide hook."""
        self.before_decide_hooks.append(hook)
        return self
    
    def register_after_decide(self, hook: AfterDecideHook) -> 'HookManager':
        """Register an after-decide hook."""
        self.after_decide_hooks.append(hook)
        return self
    
    def register_before_execute(self, hook: BeforeExecuteHook) -> 'HookManager':
        """Register a before-execute hook."""
        self.before_execute_hooks.append(hook)
        return self
    
    def register_after_execute(self, hook: AfterExecuteHook) -> 'HookManager':
        """Register an after-execute hook."""
        self.after_execute_hooks.append(hook)
        return self
    
    def register_before_learn(self, hook: BeforeLearnHook) -> 'HookManager':
        """Register a before-learn hook."""
        self.before_learn_hooks.append(hook)
        return self
    
    def register_after_learn(self, hook: AfterLearnHook) -> 'HookManager':
        """Register an after-learn hook."""
        self.after_learn_hooks.append(hook)
        return self
    
    def register_on_error(self, hook: OnErrorHook) -> 'HookManager':
        """Register an error hook."""
        self.error_hooks.append(hook)
        return self


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    # Hook interfaces | 钩子接口
    'BeforePerceiveHook',
    'AfterPerceiveHook',
    'BeforeHypothesizeHook',
    'AfterHypothesizeHook',
    'BeforeDecideHook',
    'AfterDecideHook',
    'BeforeExecuteHook',
    'AfterExecuteHook',
    'BeforeLearnHook',
    'AfterLearnHook',
    'OnErrorHook',
    # Hook manager | 钩子管理器
    'HookManager',
]