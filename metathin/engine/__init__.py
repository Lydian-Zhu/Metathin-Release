"""
Metathin Engine - Core Cognitive Cycle Implementation
=======================================================

Metathin 引擎 - 核心认知循环实现

This package contains the thinking pipeline that orchestrates the five
components (P, B, S, D, Ψ) to process input and produce output.

本包包含编排五个组件 (P, B, S, D, Ψ) 来处理输入并产生输出的思考流水线。

Modules | 模块:
    - context.py: Thinking context state container | 思考上下文状态容器
    - pipeline.py: Pure function thinking pipeline | 纯函数思考流水线
    - hooks.py: Extension points for custom behavior | 自定义行为的扩展点
"""

from .context import (
    ThinkingContext,
    PerceiveResult,
    HypothesizeResult,
    DecideResult,
    ExecuteResult,
    LearnResult,
)

from .pipeline import (
    ThinkingPipeline,
    PipelineResult,
)

from .hooks import (
    BeforePerceiveHook,
    AfterPerceiveHook,
    BeforeHypothesizeHook,
    AfterHypothesizeHook,
    BeforeDecideHook,
    AfterDecideHook,
    BeforeExecuteHook,
    AfterExecuteHook,
    BeforeLearnHook,
    AfterLearnHook,
    OnErrorHook,
    HookManager,
)


__all__ = [
    # Context | 上下文
    'ThinkingContext',
    'PerceiveResult',
    'HypothesizeResult',
    'DecideResult',
    'ExecuteResult',
    'LearnResult',
    
    # Pipeline | 流水线
    'ThinkingPipeline',
    'PipelineResult',
    
    # Hooks | 钩子
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
    'HookManager',
]