"""
History Tracker Service - Thought Record Management | 历史追踪器服务 - 思考记录管理
===================================================================================

Provides service for tracking and managing thought history.
Records each thinking cycle for debugging, analysis, and visualization.

提供追踪和管理思考历史的服务。
记录每次思考周期，用于调试、分析和可视化。

Features | 特性:
    - Automatic thought recording | 自动记录思考
    - Configurable history size limit | 可配置的历史大小限制
    - Filtering and query capabilities | 过滤和查询能力
    - Export to JSON/pickle formats | 导出为 JSON/pickle 格式
    - Statistics on success/failure rates | 成功/失败率统计

Design Philosophy | 设计理念:
    - Non-intrusive: Records without affecting performance
      非侵入式：记录不影响性能
    - Queryable: Rich filtering and search capabilities
      可查询：丰富的过滤和搜索能力
    - Exportable: Easy to save for offline analysis
      可导出：易于保存用于离线分析
"""

import json
import pickle
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging

from ..engine.context import ThinkingContext


# ============================================================
# Thought Record | 思考记录
# ============================================================

@dataclass
class ThoughtRecord:
    """
    Complete record of a single thinking cycle.
    
    单次思考周期的完整记录。
    
    This immutable record captures all relevant information about a think()
    execution for later analysis, debugging, or visualization.
    
    这个不可变记录捕获了关于 think() 执行的所有相关信息，
    用于后续分析、调试或可视化。
    
    Attributes | 属性:
        id: Unique identifier | 唯一标识符
        timestamp: When the thought occurred | 思考发生的时间
        raw_input: Original input | 原始输入
        expected: Expected result (if any) | 期望结果（如果有）
        features: Extracted feature vector | 提取的特征向量
        selected_behavior: Name of selected behavior | 选中的行为名称
        fitness_scores: Fitness scores for all behaviors | 所有行为的适应度分数
        candidate_behaviors: List of candidate behavior names | 候选行为名称列表
        result: Execution result | 执行结果
        success: Whether execution succeeded | 执行是否成功
        error_message: Error message if failed | 失败时的错误消息
        decision_time: Time spent in decision stage (seconds) | 决策阶段耗时（秒）
        execution_time: Time spent in execution stage (seconds) | 执行阶段耗时（秒）
        total_time: Total thinking time (seconds) | 总思考时间（秒）
        learning_occurred: Whether learning was performed | 是否执行了学习
        parameter_changes: Parameter changes from learning | 学习产生的参数变化
        metadata: Additional metadata | 额外的元数据
    """
    
    id: str
    timestamp: datetime
    raw_input: Any
    expected: Any
    features: Optional[List[float]]
    selected_behavior: Optional[str]
    fitness_scores: Dict[str, float]
    candidate_behaviors: List[str]
    result: Any
    success: bool
    error_message: Optional[str]
    decision_time: float
    execution_time: float
    total_time: float
    learning_occurred: bool
    parameter_changes: Dict[str, float]
    metadata: Dict[str, Any]
    
    @classmethod
    def from_context(cls, context: ThinkingContext, thought_id: str) -> 'ThoughtRecord':
        """
        Create a ThoughtRecord from a ThinkingContext.
        
        从 ThinkingContext 创建 ThoughtRecord。
        
        Args | 参数:
            context: Thinking context after pipeline execution | 流水线执行后的思考上下文
            thought_id: Unique identifier for this thought | 此思考的唯一标识符
            
        Returns | 返回:
            ThoughtRecord: New thought record | 新的思考记录
        """
        return cls(
            id=thought_id,
            timestamp=datetime.now(),
            raw_input=context.raw_input,
            expected=context.expected,
            features=context.features.tolist() if context.features is not None else None,
            selected_behavior=context.selected_behavior,
            fitness_scores=context.fitness_scores.copy(),
            candidate_behaviors=context.candidate_behaviors.copy(),
            result=context.result,
            success=context.error is None,
            error_message=str(context.error) if context.error else None,
            decision_time=context.stage_times.get('decide', 0.0),
            execution_time=context.stage_times.get('execute', 0.0),
            total_time=context.total_time(),
            learning_occurred=context.learning_occurred,
            parameter_changes=context.parameter_changes.copy(),
            metadata=context.metadata.copy(),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'raw_input': self._serialize_value(self.raw_input),
            'expected': self._serialize_value(self.expected),
            'features': self.features,
            'selected_behavior': self.selected_behavior,
            'fitness_scores': self.fitness_scores,
            'candidate_behaviors': self.candidate_behaviors,
            'result': self._serialize_value(self.result),
            'success': self.success,
            'error_message': self.error_message,
            'decision_time': self.decision_time,
            'execution_time': self.execution_time,
            'total_time': self.total_time,
            'learning_occurred': self.learning_occurred,
            'parameter_changes': self.parameter_changes,
            'metadata': self.metadata,
        }
    
    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Serialize a value for JSON storage."""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [ThoughtRecord._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): ThoughtRecord._serialize_value(v) for k, v in value.items()}
        try:
            return str(value)
        except Exception:
            return repr(value)


# ============================================================
# History Tracker Service | 历史追踪器服务
# ============================================================

class HistoryTracker:
    """
    History Tracker Service - Manages thought history.
    
    历史追踪器服务 - 管理思考历史。
    
    This service records, stores, and provides access to thought history.
    
    这个服务记录、存储并提供对思考历史的访问。
    
    Example | 示例:
        >>> tracker = HistoryTracker(max_size=1000)
        >>> 
        >>> # Record a thought | 记录一次思考
        >>> tracker.record(context, thought_id)
        >>> 
        >>> # Get recent thoughts | 获取最近的思考
        >>> recent = tracker.get_recent(10)
        >>> 
        >>> # Get statistics | 获取统计信息
        >>> stats = tracker.get_stats()
        >>> print(f"Success rate: {stats['success_rate']:.2%}")
        >>> 
        >>> # Export history | 导出历史
        >>> tracker.export_json("history.json")
    """
    
    def __init__(
        self,
        max_size: Optional[int] = 1000,
        keep_successful: bool = True,
        keep_failed: bool = True,
    ):
        """
        Initialize history tracker.
        
        初始化历史追踪器。
        
        Args | 参数:
            max_size: Maximum number of thoughts to keep, None for unlimited
                      最大保留思考数，None 表示无限制
            keep_successful: Whether to keep successful thoughts | 是否保留成功的思考
            keep_failed: Whether to keep failed thoughts | 是否保留失败的思考
        """
        self.max_size = max_size
        self.keep_successful = keep_successful
        self.keep_failed = keep_failed
        
        self._history: List[ThoughtRecord] = []
        self._thought_counter: int = 0
        self._logger = logging.getLogger("metathin.services.HistoryTracker")
        
        self._logger.info(
            f"HistoryTracker initialized: max_size={max_size}, "
            f"keep_successful={keep_successful}, keep_failed={keep_failed}"
        )
    
    def record(self, context: ThinkingContext, thought_id: Optional[str] = None) -> ThoughtRecord:
        """
        Record a thought from the pipeline context.
        
        从流水线上下文记录一次思考。
        
        Args | 参数:
            context: Thinking context after pipeline execution | 流水线执行后的思考上下文
            thought_id: Optional custom ID (auto-generated if None) | 可选的自定义 ID
            
        Returns | 返回:
            ThoughtRecord: The created record | 创建的记录
        """
        # Check if we should keep this thought | 检查是否应该保留此思考
        if context.error is not None and not self.keep_failed:
            self._logger.debug("Skipping failed thought (keep_failed=False)")
            return None
        
        if context.error is None and not self.keep_successful:
            self._logger.debug("Skipping successful thought (keep_successful=False)")
            return None
        
        # Generate ID if not provided | 如果未提供 ID，生成一个
        if thought_id is None:
            self._thought_counter += 1
            thought_id = f"thought_{self._thought_counter:08d}"
        
        # Create record | 创建记录
        record = ThoughtRecord.from_context(context, thought_id)
        
        # Add to history | 添加到历史
        self._history.append(record)
        
        # Enforce size limit | 强制执行大小限制
        if self.max_size is not None and len(self._history) > self.max_size:
            removed = len(self._history) - self.max_size
            self._history = self._history[-self.max_size:]
            self._logger.debug(f"Trimmed {removed} old records")
        
        self._logger.debug(f"Recorded thought: {thought_id}")
        return record
    
    def get_recent(self, n: int = 10) -> List[ThoughtRecord]:
        """
        Get the n most recent thoughts.
        
        获取最近的 n 次思考。
        
        Args | 参数:
            n: Number of thoughts to retrieve | 要检索的思考数量
            
        Returns | 返回:
            List[ThoughtRecord]: Recent thoughts (most recent first)
                                 最近的思考（最新的在前）
        """
        return list(reversed(self._history[-n:]))
    
    def get_by_id(self, thought_id: str) -> Optional[ThoughtRecord]:
        """
        Get a thought by its ID.
        
        根据 ID 获取思考。
        
        Args | 参数:
            thought_id: Thought identifier | 思考标识符
            
        Returns | 返回:
            Optional[ThoughtRecord]: The thought, or None if not found
                                     思考记录，未找到时返回 None
        """
        for record in self._history:
            if record.id == thought_id:
                return record
        return None
    
    def get_successful(self, limit: Optional[int] = None) -> List[ThoughtRecord]:
        """
        Get only successful thoughts.
        
        只获取成功的思考。
        
        Args | 参数:
            limit: Maximum number to return | 最大返回数量
            
        Returns | 返回:
            List[ThoughtRecord]: Successful thoughts | 成功的思考
        """
        successful = [r for r in self._history if r.success]
        if limit:
            successful = successful[-limit:]
        return successful
    
    def get_failed(self, limit: Optional[int] = None) -> List[ThoughtRecord]:
        """
        Get only failed thoughts.
        
        只获取失败的思考。
        
        Args | 参数:
            limit: Maximum number to return | 最大返回数量
            
        Returns | 返回:
            List[ThoughtRecord]: Failed thoughts | 失败的思考
        """
        failed = [r for r in self._history if not r.success]
        if limit:
            failed = failed[-limit:]
        return failed
    
    def filter(
        self,
        behavior_name: Optional[str] = None,
        success_only: bool = False,
        failure_only: bool = False,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[ThoughtRecord]:
        """
        Filter thoughts by criteria.
        
        按条件过滤思考。
        
        Args | 参数:
            behavior_name: Filter by selected behavior name | 按选中的行为名称过滤
            success_only: Only successful thoughts | 仅成功的思考
            failure_only: Only failed thoughts | 仅失败的思考
            start_time: Minimum timestamp | 最小时间戳
            end_time: Maximum timestamp | 最大时间戳
            
        Returns | 返回:
            List[ThoughtRecord]: Filtered thoughts | 过滤后的思考
        """
        results = self._history
        
        if success_only:
            results = [r for r in results if r.success]
        elif failure_only:
            results = [r for r in results if not r.success]
        
        if behavior_name:
            results = [r for r in results if r.selected_behavior == behavior_name]
        
        if start_time:
            results = [r for r in results if r.timestamp >= start_time]
        
        if end_time:
            results = [r for r in results if r.timestamp <= end_time]
        
        return results
    
    def get_last(self) -> Optional[ThoughtRecord]:
        """
        Get the most recent thought.
        
        获取最近的思考。
        
        Returns | 返回:
            Optional[ThoughtRecord]: Last thought, or None if empty
                                     最后一次思考，空时返回 None
        """
        return self._history[-1] if self._history else None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get history statistics.
        
        获取历史统计信息。
        
        Returns | 返回:
            Dict: Statistics including counts, success rate, etc.
                  包含计数、成功率等的统计字典
        """
        total = len(self._history)
        successful = len([r for r in self._history if r.success])
        failed = total - successful
        
        # Calculate average times | 计算平均时间
        avg_decision = 0.0
        avg_execution = 0.0
        avg_total = 0.0
        
        if total > 0:
            avg_decision = sum(r.decision_time for r in self._history) / total
            avg_execution = sum(r.execution_time for r in self._history) / total
            avg_total = sum(r.total_time for r in self._history) / total
        
        # Behavior usage statistics | 行为使用统计
        behavior_usage = {}
        for record in self._history:
            if record.selected_behavior:
                behavior_usage[record.selected_behavior] = \
                    behavior_usage.get(record.selected_behavior, 0) + 1
        
        return {
            'total_thoughts': total,
            'successful_thoughts': successful,
            'failed_thoughts': failed,
            'success_rate': successful / max(1, total),
            'avg_decision_time': avg_decision,
            'avg_execution_time': avg_execution,
            'avg_total_time': avg_total,
            'behavior_usage': behavior_usage,
            'max_history_size': self.max_size,
            'current_history_size': total,
        }
    
    def clear(self) -> None:
        """Clear all history."""
        self._history.clear()
        self._logger.info("History cleared")
    
    # ============================================================
    # Export Methods | 导出方法
    # ============================================================
    
    def export_json(self, filepath: Union[str, Path]) -> bool:
        """
        Export history to JSON file.
        
        导出历史到 JSON 文件。
        
        Args | 参数:
            filepath: Output file path | 输出文件路径
            
        Returns | 返回:
            bool: True if successful | 成功返回 True
        """
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = [r.to_dict() for r in self._history]
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            self._logger.info(f"Exported {len(data)} thoughts to {filepath}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to export JSON: {e}")
            return False
    
    def export_pickle(self, filepath: Union[str, Path]) -> bool:
        """
        Export history to pickle file.
        
        导出历史到 pickle 文件。
        
        Args | 参数:
            filepath: Output file path | 输出文件路径
            
        Returns | 返回:
            bool: True if successful | 成功返回 True
        """
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'wb') as f:
                pickle.dump(self._history, f)
            
            self._logger.info(f"Exported {len(self._history)} thoughts to {filepath}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to export pickle: {e}")
            return False
    
    def load_pickle(self, filepath: Union[str, Path]) -> bool:
        """
        Load history from pickle file.
        
        从 pickle 文件加载历史。
        
        Args | 参数:
            filepath: Input file path | 输入文件路径
            
        Returns | 返回:
            bool: True if successful | 成功返回 True
        """
        try:
            path = Path(filepath)
            if not path.exists():
                self._logger.error(f"File not found: {filepath}")
                return False
            
            with open(path, 'rb') as f:
                loaded = pickle.load(f)
            
            if isinstance(loaded, list):
                self._history = loaded
                self._logger.info(f"Loaded {len(self._history)} thoughts from {filepath}")
                return True
            else:
                self._logger.error("Invalid pickle format")
                return False
                
        except Exception as e:
            self._logger.error(f"Failed to load pickle: {e}")
            return False
    
    def __len__(self) -> int:
        """Return number of thoughts in history."""
        return len(self._history)
    
    def __iter__(self):
        """Iterate over thoughts (oldest to newest)."""
        return iter(self._history)
    
    def __getitem__(self, index: int) -> ThoughtRecord:
        """Get thought by index."""
        return self._history[index]


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    'ThoughtRecord',    # Thought record data class | 思考记录数据类
    'HistoryTracker',   # History tracker service | 历史追踪器服务
]