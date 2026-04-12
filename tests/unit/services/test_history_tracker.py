"""
Unit tests for HistoryTracker service.
历史追踪器服务单元测试。
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from metathin.services import HistoryTracker, ThoughtRecord
from metathin.engine import ThinkingContext


class TestThoughtRecord:
    """Test ThoughtRecord data class."""
    
    def test_thought_record_creation(self):
        """Thought record should be creatable."""
        record = ThoughtRecord(
            id="test_001",
            timestamp=datetime.now(),
            raw_input="test",
            expected=None,
            features=[1.0, 2.0],
            selected_behavior="greet",
            fitness_scores={"greet": 0.9, "echo": 0.5},
            candidate_behaviors=["greet", "echo"],
            result="Hello!",
            success=True,
            error_message=None,
            decision_time=0.001,
            execution_time=0.002,
            total_time=0.003,
            learning_occurred=False,
            parameter_changes={},
            metadata={}
        )
        
        assert record.id == "test_001"
        assert record.success is True
        assert record.selected_behavior == "greet"
    
    def test_thought_record_from_context(self):
        """Should create ThoughtRecord from ThinkingContext."""
        features = np.array([1.0, 2.0], dtype=np.float64)
        context = ThinkingContext(
            raw_input="test_input",
            expected=42.0,
            features=features,
            selected_behavior="greet",
            fitness_scores={"greet": 0.9, "echo": 0.5},
            candidate_behaviors=["greet", "echo"],
            result="Hello!",
            stage_times={"decide": 0.001, "execute": 0.002},
            metadata={"key": "value"}
        )
        
        record = ThoughtRecord.from_context(context, "test_001")
        
        assert record.id == "test_001"
        assert record.raw_input == "test_input"
        assert record.expected == 42.0
        assert record.features == [1.0, 2.0]
        assert record.selected_behavior == "greet"
        assert record.fitness_scores == {"greet": 0.9, "echo": 0.5}
        assert record.result == "Hello!"
        assert record.success is True
        assert record.decision_time == 0.001
        assert record.execution_time == 0.002
        assert record.metadata == {"key": "value"}
    
    def test_thought_record_from_failed_context(self):
        """Should create ThoughtRecord from failed context."""
        error = ValueError("Something went wrong")
        context = ThinkingContext(
            raw_input="test",
            error=error,
            stage_times={"perceive": 0.001}
        )
        
        record = ThoughtRecord.from_context(context, "failed_001")
        
        assert record.success is False
        assert "Something went wrong" in record.error_message
        assert record.total_time > 0
    
    def test_thought_record_to_dict(self):
        """Should convert to dictionary for serialization."""
        record = ThoughtRecord(
            id="test_001",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            raw_input="test",
            expected=42,
            features=[1.0, 2.0],
            selected_behavior="greet",
            fitness_scores={"greet": 0.9},
            candidate_behaviors=["greet"],
            result="Hello!",
            success=True,
            error_message=None,
            decision_time=0.001,
            execution_time=0.002,
            total_time=0.003,
            learning_occurred=False,
            parameter_changes={},
            metadata={"key": "value"}
        )
        
        d = record.to_dict()
        
        assert d['id'] == "test_001"
        assert d['raw_input'] == "test"
        assert d['selected_behavior'] == "greet"
        assert d['success'] is True
        assert 'timestamp' in d


class TestHistoryTracker:
    """Test HistoryTracker service."""
    
    def test_initialization(self):
        """History tracker should initialize correctly."""
        tracker = HistoryTracker(max_size=100)
        
        assert tracker.max_size == 100
        assert tracker.keep_successful is True
        assert tracker.keep_failed is True
        assert len(tracker) == 0
    
    def test_record_successful_thought(self):
        """Should record successful thoughts."""
        tracker = HistoryTracker()
        context = ThinkingContext(raw_input="test", result="success")
        
        record = tracker.record(context)
        
        assert record is not None
        assert record.success is True
        assert len(tracker) == 1
    
    def test_record_failed_thought(self):
        """Should record failed thoughts."""
        tracker = HistoryTracker()
        error = ValueError("test error")
        context = ThinkingContext(raw_input="test", error=error)
        
        record = tracker.record(context)
        
        assert record is not None
        assert record.success is False
        assert len(tracker) == 1
    
    def test_skip_failed_when_keep_failed_false(self):
        """Should skip failed thoughts when keep_failed is False."""
        tracker = HistoryTracker(keep_failed=False)
        error = ValueError("test error")
        context = ThinkingContext(raw_input="test", error=error)
        
        record = tracker.record(context)
        
        assert record is None
        assert len(tracker) == 0
    
    def test_skip_successful_when_keep_successful_false(self):
        """Should skip successful thoughts when keep_successful is False."""
        tracker = HistoryTracker(keep_successful=False)
        context = ThinkingContext(raw_input="test", result="success")
        
        record = tracker.record(context)
        
        assert record is None
        assert len(tracker) == 0
    
    def test_max_size_limit(self):
        """Should respect max size limit."""
        tracker = HistoryTracker(max_size=3)
        
        for i in range(5):
            context = ThinkingContext(raw_input=f"test_{i}", result=f"result_{i}")
            tracker.record(context)
        
        assert len(tracker) == 3
    
    def test_get_recent_returns_most_recent(self):
        """get_recent() should return most recent thoughts."""
        tracker = HistoryTracker()
        
        for i in range(10):
            context = ThinkingContext(raw_input=f"test_{i}", result=f"result_{i}")
            tracker.record(context)
        
        recent = tracker.get_recent(3)
        
        assert len(recent) == 3
        # Most recent first
        assert recent[0].raw_input == "test_9"
        assert recent[1].raw_input == "test_8"
        assert recent[2].raw_input == "test_7"
    
    def test_get_by_id_returns_correct_record(self):
        """get_by_id() should return record by ID."""
        tracker = HistoryTracker()
        
        context = ThinkingContext(raw_input="test", result="result")
        record = tracker.record(context)
        
        found = tracker.get_by_id(record.id)
        assert found is not None
        assert found.id == record.id
    
    def test_get_by_id_returns_none_for_not_found(self):
        """get_by_id() should return None for non-existent ID."""
        tracker = HistoryTracker()
        
        found = tracker.get_by_id("nonexistent")
        assert found is None
    
    def test_get_successful_returns_only_successful(self):
        """get_successful() should return only successful thoughts."""
        tracker = HistoryTracker()
        
        # Record mix of successful and failed
        tracker.record(ThinkingContext(raw_input="success1", result="ok"))
        error = ValueError("fail")
        tracker.record(ThinkingContext(raw_input="fail1", error=error))
        tracker.record(ThinkingContext(raw_input="success2", result="ok"))
        
        successful = tracker.get_successful()
        
        assert len(successful) == 2
        assert all(r.success for r in successful)
    
    def test_get_failed_returns_only_failed(self):
        """get_failed() should return only failed thoughts."""
        tracker = HistoryTracker()
        
        tracker.record(ThinkingContext(raw_input="success1", result="ok"))
        error = ValueError("fail1")
        tracker.record(ThinkingContext(raw_input="fail1", error=error))
        error2 = ValueError("fail2")
        tracker.record(ThinkingContext(raw_input="fail2", error=error2))
        
        failed = tracker.get_failed()
        
        assert len(failed) == 2
        assert all(not r.success for r in failed)
    
    def test_filter_by_behavior(self):
        """Should filter thoughts by behavior name."""
        tracker = HistoryTracker()
        
        for i in range(5):
            context = ThinkingContext(
                raw_input=f"test_{i}",
                selected_behavior=f"behavior_{i % 2}",
                result="ok"
            )
            tracker.record(context)
        
        filtered = tracker.filter(behavior_name="behavior_0")
        
        assert len(filtered) == 3
        assert all(r.selected_behavior == "behavior_0" for r in filtered)
    
    def test_filter_by_success_only(self):
        """Should filter by success only."""
        tracker = HistoryTracker()
        
        tracker.record(ThinkingContext(raw_input="success", result="ok"))
        error = ValueError("fail")
        tracker.record(ThinkingContext(raw_input="fail", error=error))
        tracker.record(ThinkingContext(raw_input="success2", result="ok"))
        
        filtered = tracker.filter(success_only=True)
        
        assert len(filtered) == 2
        assert all(r.success for r in filtered)
    
    def test_filter_by_failure_only(self):
        """Should filter by failure only."""
        tracker = HistoryTracker()
        
        tracker.record(ThinkingContext(raw_input="success", result="ok"))
        error = ValueError("fail")
        tracker.record(ThinkingContext(raw_input="fail", error=error))
        
        filtered = tracker.filter(failure_only=True)
        
        assert len(filtered) == 1
        assert not filtered[0].success
    
    def test_filter_by_time_range(self):
        """Should filter by time range."""
        tracker = HistoryTracker()
        
        now = datetime.now()
        
        # Record with specific timestamps
        for i, hours_ago in enumerate([10, 5, 2, 1]):
            context = ThinkingContext(raw_input=f"test_{i}", result="ok")
            record = tracker.record(context)
            # Manually modify timestamp (for testing)
            record.timestamp = now - timedelta(hours=hours_ago)
        
        start_time = now - timedelta(hours=3)
        end_time = now - timedelta(hours=1)
        
        filtered = tracker.filter(start_time=start_time, end_time=end_time)
        
        # Should get records from 2 and 1 hours ago
        assert len(filtered) == 2
    
    def test_get_last_returns_most_recent(self):
        """get_last() should return the most recent thought."""
        tracker = HistoryTracker()
        
        tracker.record(ThinkingContext(raw_input="first", result="ok"))
        tracker.record(ThinkingContext(raw_input="second", result="ok"))
        tracker.record(ThinkingContext(raw_input="third", result="ok"))
        
        last = tracker.get_last()
        
        assert last is not None
        assert last.raw_input == "third"
    
    def test_get_last_returns_none_when_empty(self):
        """get_last() should return None when history is empty."""
        tracker = HistoryTracker()
        
        last = tracker.get_last()
        assert last is None
    
    def test_get_stats_returns_statistics(self):
        """get_stats() should return comprehensive statistics."""
        tracker = HistoryTracker()
        
        # Record 3 successes, 2 failures
        for i in range(3):
            tracker.record(ThinkingContext(raw_input=f"success_{i}", result="ok"))
        for i in range(2):
            error = ValueError(f"fail_{i}")
            tracker.record(ThinkingContext(raw_input=f"fail_{i}", error=error))
        
        stats = tracker.get_stats()
        
        assert stats['total_thoughts'] == 5
        assert stats['successful_thoughts'] == 3
        assert stats['failed_thoughts'] == 2
        assert stats['success_rate'] == 0.6
        assert 'behavior_usage' in stats
    
    def test_clear_removes_all_records(self):
        """clear() should remove all records."""
        tracker = HistoryTracker()
        
        for i in range(5):
            tracker.record(ThinkingContext(raw_input=f"test_{i}", result="ok"))
        
        assert len(tracker) == 5
        
        tracker.clear()
        
        assert len(tracker) == 0
        assert tracker.get_last() is None
    
    def test_export_json_creates_file(self, tmp_path):
        """export_json() should create a JSON file."""
        tracker = HistoryTracker()
        
        tracker.record(ThinkingContext(raw_input="test1", result="ok"))
        tracker.record(ThinkingContext(raw_input="test2", result="ok"))
        
        filepath = tmp_path / "history.json"
        success = tracker.export_json(filepath)
        
        assert success is True
        assert filepath.exists()
        
        # Verify file contains data
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        assert len(data) == 2
    
    def test_export_pickle_creates_file(self, tmp_path):
        """export_pickle() should create a pickle file."""
        tracker = HistoryTracker()
        
        tracker.record(ThinkingContext(raw_input="test1", result="ok"))
        tracker.record(ThinkingContext(raw_input="test2", result="ok"))
        
        filepath = tmp_path / "history.pkl"
        success = tracker.export_pickle(filepath)
        
        assert success is True
        assert filepath.exists()
    
    def test_load_pickle_restores_history(self, tmp_path):
        """load_pickle() should restore history from file."""
        tracker1 = HistoryTracker()
        
        tracker1.record(ThinkingContext(raw_input="test1", result="ok"))
        tracker1.record(ThinkingContext(raw_input="test2", result="ok"))
        
        filepath = tmp_path / "history.pkl"
        tracker1.export_pickle(filepath)
        
        tracker2 = HistoryTracker()
        success = tracker2.load_pickle(filepath)
        
        assert success is True
        assert len(tracker2) == 2
        assert tracker2.get_recent(1)[0].raw_input == "test2"
    
    def test_iteration(self):
        """Should support iteration over thoughts."""
        tracker = HistoryTracker()
        
        for i in range(3):
            tracker.record(ThinkingContext(raw_input=f"test_{i}", result="ok"))
        
        thoughts = list(tracker)
        assert len(thoughts) == 3
        assert thoughts[0].raw_input == "test_0"
        assert thoughts[1].raw_input == "test_1"
        assert thoughts[2].raw_input == "test_2"
    
    def test_indexing(self):
        """Should support indexing."""
        tracker = HistoryTracker()
        
        tracker.record(ThinkingContext(raw_input="first", result="ok"))
        tracker.record(ThinkingContext(raw_input="second", result="ok"))
        
        assert tracker[0].raw_input == "first"
        assert tracker[1].raw_input == "second"