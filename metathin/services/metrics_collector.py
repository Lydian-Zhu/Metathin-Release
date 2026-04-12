"""
Metrics Collector Service - Performance Monitoring | 指标收集器服务 - 性能监控
===============================================================================

Provides metrics collection and aggregation for monitoring agent performance.
Tracks key performance indicators (KPIs) over time.

提供用于监控代理性能的指标收集和聚合。
随时间追踪关键性能指标 (KPI)。

Metrics Tracked | 追踪的指标:
    - Thought counts (total, successful, failed) | 思考计数（总数、成功、失败）
    - Timing metrics (decision, execution, total) | 时间指标（决策、执行、总时间）
    - Behavior usage statistics | 行为使用统计
    - Learning events | 学习事件
    - Error rates | 错误率

Design Philosophy | 设计理念:
    - Non-blocking: Metrics collection doesn't impact performance
      非阻塞：指标收集不影响性能
    - Aggregatable: Can compute running averages and trends
      可聚合：可以计算运行平均值和趋势
    - Exportable: Can output to various formats (JSON, CSV)
      可导出：可以输出到各种格式（JSON、CSV）
"""

import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
from pathlib import Path
import json
import csv
import logging

from metathin.engine.context import ThinkingContext


# ============================================================
# Metrics Data Classes | 指标数据类
# ============================================================

@dataclass
class ThoughtMetrics:
    """Metrics for a single thought cycle | 单次思考周期的指标"""
    timestamp: float
    success: bool
    decision_time: float
    execution_time: float
    total_time: float
    selected_behavior: Optional[str]
    learning_occurred: bool
    error_type: Optional[str] = None


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over a time window | 时间窗口内的聚合指标"""
    # Counts | 计数
    total_thoughts: int = 0
    successful_thoughts: int = 0
    failed_thoughts: int = 0
    
    # Rates | 比率
    success_rate: float = 0.0
    failure_rate: float = 0.0
    
    # Timing (seconds) | 时间（秒）
    avg_decision_time: float = 0.0
    avg_execution_time: float = 0.0
    avg_total_time: float = 0.0
    min_total_time: float = 0.0
    max_total_time: float = 0.0
    
    # Behavior stats | 行为统计
    behavior_usage: Dict[str, int] = field(default_factory=dict)
    most_used_behavior: Optional[str] = None
    
    # Learning stats | 学习统计
    learning_events: int = 0
    learning_rate: float = 0.0
    
    # Error stats | 错误统计
    error_types: Dict[str, int] = field(default_factory=dict)
    most_common_error: Optional[str] = None
    
    def update_from_thought(self, metrics: ThoughtMetrics) -> None:
        """Update aggregated metrics from a single thought."""
        self.total_thoughts += 1
        
        if metrics.success:
            self.successful_thoughts += 1
        else:
            self.failed_thoughts += 1
            if metrics.error_type:
                self.error_types[metrics.error_type] = \
                    self.error_types.get(metrics.error_type, 0) + 1
        
        # Update behavior usage | 更新行为使用
        if metrics.selected_behavior:
            self.behavior_usage[metrics.selected_behavior] = \
                self.behavior_usage.get(metrics.selected_behavior, 0) + 1
        
        # Update learning | 更新学习
        if metrics.learning_occurred:
            self.learning_events += 1
        
        # Update running averages | 更新运行平均值
        n = self.total_thoughts
        self.avg_decision_time = (self.avg_decision_time * (n - 1) + metrics.decision_time) / n
        self.avg_execution_time = (self.avg_execution_time * (n - 1) + metrics.execution_time) / n
        self.avg_total_time = (self.avg_total_time * (n - 1) + metrics.total_time) / n
        
        # Update min/max | 更新最小/最大值
        if n == 1:
            self.min_total_time = metrics.total_time
            self.max_total_time = metrics.total_time
        else:
            self.min_total_time = min(self.min_total_time, metrics.total_time)
            self.max_total_time = max(self.max_total_time, metrics.total_time)
        
        # Update rates | 更新比率
        self.success_rate = self.successful_thoughts / max(1, self.total_thoughts)
        self.failure_rate = self.failed_thoughts / max(1, self.total_thoughts)
        self.learning_rate = self.learning_events / max(1, self.total_thoughts)
        
        # Update most used behavior | 更新最常用行为
        if self.behavior_usage:
            self.most_used_behavior = max(self.behavior_usage, key=self.behavior_usage.get)
        
        # Update most common error | 更新最常见错误
        if self.error_types:
            self.most_common_error = max(self.error_types, key=self.error_types.get)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_thoughts': self.total_thoughts,
            'successful_thoughts': self.successful_thoughts,
            'failed_thoughts': self.failed_thoughts,
            'success_rate': self.success_rate,
            'failure_rate': self.failure_rate,
            'avg_decision_time': self.avg_decision_time,
            'avg_execution_time': self.avg_execution_time,
            'avg_total_time': self.avg_total_time,
            'min_total_time': self.min_total_time,
            'max_total_time': self.max_total_time,
            'behavior_usage': self.behavior_usage,
            'most_used_behavior': self.most_used_behavior,
            'learning_events': self.learning_events,
            'learning_rate': self.learning_rate,
            'error_types': self.error_types,
            'most_common_error': self.most_common_error,
        }


# ============================================================
# Metrics Collector Service | 指标收集器服务
# ============================================================

class MetricsCollector:
    """
    Metrics Collector Service - Collects and aggregates performance metrics.
    
    指标收集器服务 - 收集和聚合性能指标。
    
    This service tracks key performance indicators over time and provides
    both real-time and aggregated statistics.
    
    这个服务随时间追踪关键性能指标，并提供实时和聚合统计。
    
    Example | 示例:
        >>> collector = MetricsCollector(window_size=100)
        >>> 
        >>> # Record a thought | 记录一次思考
        >>> collector.record(thought_metrics)
        >>> 
        >>> # Get current metrics | 获取当前指标
        >>> metrics = collector.get_metrics()
        >>> print(f"Success rate: {metrics.success_rate:.2%}")
        >>> 
        >>> # Get time series | 获取时间序列
        >>> series = collector.get_time_series()
        >>> 
        >>> # Export to file | 导出到文件
        >>> collector.export_json("metrics.json")
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        enable_time_series: bool = True,
        max_time_series_length: int = 10000,
    ):
        """
        Initialize metrics collector.
        
        初始化指标收集器。
        
        Args | 参数:
            window_size: Number of thoughts to keep in rolling window
                         滚动窗口中保留的思考数量
            enable_time_series: Whether to store time series data | 是否存储时间序列数据
            max_time_series_length: Maximum length of time series | 时间序列的最大长度
        """
        self.window_size = window_size
        self.enable_time_series = enable_time_series
        self.max_time_series_length = max_time_series_length
        
        # Rolling window of recent thoughts | 最近思考的滚动窗口
        self._recent: deque = deque(maxlen=window_size)
        
        # Full history (if enabled) | 完整历史（如果启用）
        self._time_series: List[ThoughtMetrics] = []
        
        # Aggregated metrics | 聚合指标
        self._aggregated = AggregatedMetrics()
        
        # Start time | 开始时间
        self._start_time: float = time.time()
        
        # Logger | 日志记录器
        self._logger = logging.getLogger("metathin.services.MetricsCollector")
        
        self._logger.info(
            f"MetricsCollector initialized: window_size={window_size}, "
            f"time_series={enable_time_series}"
        )
    
    def record(
        self,
        success: bool,
        decision_time: float,
        execution_time: float,
        selected_behavior: Optional[str] = None,
        learning_occurred: bool = False,
        error_type: Optional[str] = None,
    ) -> None:
        """
        Record metrics from a single thought cycle.
        
        记录单次思考周期的指标。
        
        Args | 参数:
            success: Whether the thought was successful | 思考是否成功
            decision_time: Time spent in decision stage (seconds) | 决策阶段耗时（秒）
            execution_time: Time spent in execution stage (seconds) | 执行阶段耗时（秒）
            selected_behavior: Name of selected behavior | 选中的行为名称
            learning_occurred: Whether learning was performed | 是否执行了学习
            error_type: Type of error (if any) | 错误类型（如果有）
        """
        metrics = ThoughtMetrics(
            timestamp=time.time(),
            success=success,
            decision_time=decision_time,
            execution_time=execution_time,
            total_time=decision_time + execution_time,
            selected_behavior=selected_behavior,
            learning_occurred=learning_occurred,
            error_type=error_type,
        )
        
        # Add to rolling window | 添加到滚动窗口
        self._recent.append(metrics)
        
        # Add to time series (if enabled) | 添加到时间序列（如果启用）
        if self.enable_time_series:
            self._time_series.append(metrics)
            if len(self._time_series) > self.max_time_series_length:
                self._time_series = self._time_series[-self.max_time_series_length:]
        
        # Update aggregated metrics | 更新聚合指标
        self._aggregated.update_from_thought(metrics)
        
        self._logger.debug(
            f"Recorded: success={success}, decision={decision_time:.3f}s, "
            f"execution={execution_time:.3f}s, behavior={selected_behavior}"
        )
    
    def record_from_context(self, context: 'ThinkingContext') -> None:
        """
        Record metrics from a ThinkingContext.
        
        从 ThinkingContext 记录指标。
        
        Args | 参数:
            context: Thinking context after pipeline execution | 流水线执行后的思考上下文
        """
        error_type = None
        if context.error:
            error_type = context.error.__class__.__name__
        
        self.record(
            success=context.error is None,
            decision_time=context.stage_times.get('decide', 0.0),
            execution_time=context.stage_times.get('execute', 0.0),
            selected_behavior=context.selected_behavior,
            learning_occurred=context.learning_occurred,
            error_type=error_type,
        )
    
    def get_metrics(self, window: Optional[int] = None) -> AggregatedMetrics:
        """
        Get aggregated metrics for the specified window.
        
        获取指定窗口的聚合指标。
        
        Args | 参数:
            window: Number of recent thoughts to consider (None = all)
                    要考虑的最近思考数量（None = 全部）
                    
        Returns | 返回:
            AggregatedMetrics: Aggregated metrics | 聚合指标
        """
        if window is None or window >= len(self._recent):
            return self._aggregated
        
        # Compute metrics for specific window | 计算特定窗口的指标
        window_metrics = list(self._recent)[-window:]
        aggregated = AggregatedMetrics()
        for m in window_metrics:
            aggregated.update_from_thought(m)
        return aggregated
    
    def get_recent_metrics(self, window: int = 100) -> AggregatedMetrics:
        """
        Get metrics for the most recent N thoughts.
        
        获取最近 N 次思考的指标。
        
        Args | 参数:
            window: Number of recent thoughts to consider | 要考虑的最近思考数量
            
        Returns | 返回:
            AggregatedMetrics: Aggregated metrics for recent window
                                最近窗口的聚合指标
        """
        return self.get_metrics(window)
    
    def get_time_series(self) -> List[Dict[str, Any]]:
        """
        Get time series data for analysis.
        
        获取用于分析的时间序列数据。
        
        Returns | 返回:
            List[Dict]: List of metrics over time | 随时间变化的指标列表
        """
        return [
            {
                'timestamp': m.timestamp,
                'success': m.success,
                'decision_time': m.decision_time,
                'execution_time': m.execution_time,
                'total_time': m.total_time,
                'selected_behavior': m.selected_behavior,
                'learning_occurred': m.learning_occurred,
                'error_type': m.error_type,
            }
            for m in self._time_series
        ]
    
    def get_success_rate_trend(self, bucket_size: int = 100) -> List[float]:
        """
        Get success rate trend over time.
        
        获取随时间变化的成功率趋势。
        
        Args | 参数:
            bucket_size: Number of thoughts per bucket | 每个桶的思考数量
            
        Returns | 返回:
            List[float]: Success rate for each bucket | 每个桶的成功率
        """
        if not self._time_series:
            return []
        
        trends = []
        for i in range(0, len(self._time_series), bucket_size):
            bucket = self._time_series[i:i + bucket_size]
            success_count = sum(1 for m in bucket if m.success)
            trends.append(success_count / len(bucket))
        
        return trends
    
    def get_behavior_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics per behavior.
        
        获取每个行为的性能指标。
        
        Returns | 返回:
            Dict: Behavior name -> {success_rate, avg_time, count}
                  行为名称 -> {成功率, 平均时间, 次数}
        """
        behavior_stats = {}
        
        for m in self._time_series:
            if not m.selected_behavior:
                continue
            
            if m.selected_behavior not in behavior_stats:
                behavior_stats[m.selected_behavior] = {
                    'count': 0,
                    'successes': 0,
                    'total_time': 0.0,
                }
            
            stats = behavior_stats[m.selected_behavior]
            stats['count'] += 1
            if m.success:
                stats['successes'] += 1
            stats['total_time'] += m.total_time
        
        # Compute derived metrics | 计算派生指标
        result = {}
        for name, stats in behavior_stats.items():
            result[name] = {
                'count': stats['count'],
                'success_rate': stats['successes'] / stats['count'],
                'avg_time': stats['total_time'] / stats['count'],
            }
        
        return result
    
    def get_error_rate_trend(self, bucket_size: int = 100) -> List[float]:
        """
        Get error rate trend over time.
        
        获取随时间变化的错误率趋势。
        
        Args | 参数:
            bucket_size: Number of thoughts per bucket | 每个桶的思考数量
            
        Returns | 返回:
            List[float]: Error rate for each bucket | 每个桶的错误率
        """
        if not self._time_series:
            return []
        
        trends = []
        for i in range(0, len(self._time_series), bucket_size):
            bucket = self._time_series[i:i + bucket_size]
            error_count = sum(1 for m in bucket if not m.success)
            trends.append(error_count / len(bucket))
        
        return trends
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of all metrics.
        
        获取所有指标的全面摘要。
        
        Returns | 返回:
            Dict: Comprehensive metrics summary | 全面的指标摘要
        """
        uptime = time.time() - self._start_time
        
        return {
            'aggregated': self._aggregated.to_dict(),
            'uptime_seconds': uptime,
            'thoughts_per_second': self._aggregated.total_thoughts / max(1, uptime),
            'behavior_performance': self.get_behavior_performance(),
            'window_size': self.window_size,
            'time_series_length': len(self._time_series),
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._recent.clear()
        self._time_series.clear()
        self._aggregated = AggregatedMetrics()
        self._start_time = time.time()
        self._logger.info("Metrics reset")
    
    # ============================================================
    # Export Methods | 导出方法
    # ============================================================
    
    def export_json(self, filepath: Union[str, Path]) -> bool:
        """
        Export metrics to JSON file.
        
        导出指标到 JSON 文件。
        
        Args | 参数:
            filepath: Output file path | 输出文件路径
            
        Returns | 返回:
            bool: True if successful | 成功返回 True
        """
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'summary': self.get_summary(),
                'time_series': self.get_time_series(),
                'exported_at': datetime.now().isoformat(),
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self._logger.info(f"Exported metrics to {filepath}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to export JSON: {e}")
            return False
    
    def export_csv(self, filepath: Union[str, Path]) -> bool:
        """
        Export time series to CSV file.
        
        导出时间序列到 CSV 文件。
        
        Args | 参数:
            filepath: Output file path | 输出文件路径
            
        Returns | 返回:
            bool: True if successful | 成功返回 True
        """
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', newline='', encoding='utf-8') as f:
                if not self._time_series:
                    return True
                
                fieldnames = [
                    'timestamp', 'success', 'decision_time', 'execution_time',
                    'total_time', 'selected_behavior', 'learning_occurred', 'error_type'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for m in self._time_series:
                    writer.writerow({
                        'timestamp': m.timestamp,
                        'success': m.success,
                        'decision_time': m.decision_time,
                        'execution_time': m.execution_time,
                        'total_time': m.total_time,
                        'selected_behavior': m.selected_behavior,
                        'learning_occurred': m.learning_occurred,
                        'error_type': m.error_type or '',
                    })
            
            self._logger.info(f"Exported time series to {filepath}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to export CSV: {e}")
            return False


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    'ThoughtMetrics',     # Single thought metrics | 单次思考指标
    'AggregatedMetrics',  # Aggregated metrics | 聚合指标
    'MetricsCollector',   # Metrics collector service | 指标收集器服务
]