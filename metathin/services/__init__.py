"""
Metathin Services - Optional Service Layer
===========================================

Metathin 服务层 - 可选服务层

This package contains optional services that can be injected into the agent.
All services are designed to be independent and replaceable.

本包包含可注入代理的可选服务。
所有服务都设计为独立且可替换的。

Services | 服务:
    - MemoryManager: Two-tier caching memory service | 二级缓存记忆服务
    - HistoryTracker: Thought history tracking service | 思考历史追踪服务
    - MetricsCollector: Performance metrics collection | 性能指标收集服务

Design Philosophy | 设计理念:
    - Optional: Services can be enabled/disabled as needed
      可选：服务可根据需要启用/禁用
    - Independent: Services don't depend on each other
      独立：服务之间不相互依赖
    - Injectable: Services are injected, not hardcoded
      可注入：服务是注入的，不是硬编码的
"""

from .memory_manager import MemoryManager
from .history_tracker import HistoryTracker as _BaseHistoryTracker, ThoughtRecord
from .metrics_collector import MetricsCollector, ThoughtMetrics, AggregatedMetrics


class HistoryTracker(_BaseHistoryTracker):
    """HistoryTracker compatibility wrapper.

    空的 HistoryTracker 在布尔判断中会被当成 False，导致 Metathin.think()
    跳过历史记录；同时 agent 调用的是历史上并不存在的
    ``record_from_context``。这里补齐这两个兼容点，后续可把调用点改为
    ``is not None`` 后移除。
    """

    def __bool__(self) -> bool:
        return True

    def record_from_context(self, context, thought_id=None):
        return self.record(context, thought_id)


__all__ = [
    # Memory Service | 记忆服务
    'MemoryManager',
    
    # History Service | 历史服务
    'HistoryTracker',
    'ThoughtRecord',
    
    # Metrics Service | 指标服务
    'MetricsCollector',
    'ThoughtMetrics',
    'AggregatedMetrics',
]