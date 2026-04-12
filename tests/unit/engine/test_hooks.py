"""
Unit tests for thinking hooks system.
思考钩子系统单元测试。
"""

import pytest
from metathin.engine import (
    ThinkingContext,
    HookManager,
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
    OnErrorHook
)
from metathin.core.types import FeatureVector
from metathin.core.b_behavior import MetaBehavior
import numpy as np


class SimpleBehavior(MetaBehavior):
    """Simple behavior for testing."""
    
    def __init__(self, name: str = "test"):
        super().__init__()
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
    
    def execute(self, features: FeatureVector, **kwargs):
        return f"Executed {self._name}"


# ============================================================
# Test Hook Implementations
# ============================================================

class TrackingBeforePerceiveHook(BeforePerceiveHook):
    def __init__(self):
        self.called = False
        self.context_received = None
    
    def on_before_perceive(self, context: ThinkingContext) -> ThinkingContext:
        self.called = True
        self.context_received = context
        return context


class ModifyingAfterPerceiveHook(AfterPerceiveHook):
    def on_after_perceive(self, context: ThinkingContext, features: FeatureVector) -> ThinkingContext:
        # Add custom metadata
        new_metadata = context.metadata.copy()
        new_metadata['after_perceive_called'] = True
        return ThinkingContext(
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
            metadata=new_metadata,
            error=context.error,
        )


class FilteringBeforeHypothesizeHook(BeforeHypothesizeHook):
    def on_before_hypothesize(self, context: ThinkingContext, behaviors):
        # Filter out behaviors with name starting with 'skip'
        filtered = [b for b in behaviors if not b.name.startswith('skip')]
        return context, filtered


class ModifyingAfterHypothesizeHook(AfterHypothesizeHook):
    def on_after_hypothesize(self, context: ThinkingContext, fitness_scores):
        # Boost fitness for 'boost' behavior
        new_scores = fitness_scores.copy()
        if 'boost' in new_scores:
            new_scores['boost'] = min(1.0, new_scores['boost'] + 0.2)
        return ThinkingContext(
            raw_input=context.raw_input,
            expected=context.expected,
            context_params=context.context_params,
            features=context.features,
            fitness_scores=new_scores,
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


class ValidatingBeforeDecideHook(BeforeDecideHook):
    def on_before_decide(self, context: ThinkingContext, candidates, fitness_scores):
        # Ensure no NaN in fitness scores
        assert all(0 <= s <= 1 for s in fitness_scores)
        return context, candidates, fitness_scores


class LoggingAfterDecideHook(AfterDecideHook):
    def __init__(self):
        self.selected = None
    
    def on_after_decide(self, context: ThinkingContext, selected):
        self.selected = selected
        return context


class ResourcePreparingBeforeExecuteHook(BeforeExecuteHook):
    def __init__(self):
        self.prepared = False
    
    def on_before_execute(self, context: ThinkingContext, behavior):
        self.prepared = True
        return context


class ResultValidatingAfterExecuteHook(AfterExecuteHook):
    def on_after_execute(self, context: ThinkingContext, result):
        # Ensure result is not None
        if result is None:
            result = "default_result"
        return ThinkingContext(
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


class SkippingBeforeLearnHook(BeforeLearnHook):
    def on_before_learn(self, context: ThinkingContext, expected, actual):
        # Skip learning if error is too small
        if abs(expected - actual) < 0.01:
            return context, None, None  # Skip by returning None for expected
        return context, expected, actual


class RecordingAfterLearnHook(AfterLearnHook):
    def __init__(self):
        self.changes_recorded = None
    
    def on_after_learn(self, context: ThinkingContext, parameter_changes):
        self.changes_recorded = parameter_changes
        return context


class ErrorRecoveryHook(OnErrorHook):
    def __init__(self):
        self.error_handled = False
        self.recovered_context = None
    
    def on_error(self, context: ThinkingContext, error, stage):
        self.error_handled = True
        # Create recovery context with error flag
        new_metadata = context.metadata.copy()
        new_metadata['error_recovered'] = True
        self.recovered_context = ThinkingContext(
            raw_input=context.raw_input,
            expected=context.expected,
            context_params=context.context_params,
            features=context.features,
            fitness_scores=context.fitness_scores,
            candidate_behaviors=context.candidate_behaviors,
            selected_behavior=context.selected_behavior,
            decision_confidence=context.decision_confidence,
            result="recovered_result",
            learning_occurred=context.learning_occurred,
            parameter_changes=context.parameter_changes,
            start_time=context.start_time,
            stage_times=context.stage_times,
            metadata=new_metadata,
            error=error,
        )
        return self.recovered_context


# ============================================================
# Test HookManager
# ============================================================

class TestHookManager:
    """Test HookManager registration and management."""
    
    def test_register_before_perceive(self):
        """Should register before-perceive hooks."""
        manager = HookManager()
        hook = TrackingBeforePerceiveHook()
        
        manager.register_before_perceive(hook)
        
        assert len(manager.before_perceive_hooks) == 1
        assert manager.before_perceive_hooks[0] is hook
    
    def test_register_after_perceive(self):
        """Should register after-perceive hooks."""
        manager = HookManager()
        hook = ModifyingAfterPerceiveHook()
        
        manager.register_after_perceive(hook)
        
        assert len(manager.after_perceive_hooks) == 1
    
    def test_register_before_hypothesize(self):
        """Should register before-hypothesize hooks."""
        manager = HookManager()
        hook = FilteringBeforeHypothesizeHook()
        
        manager.register_before_hypothesize(hook)
        
        assert len(manager.before_hypothesize_hooks) == 1
    
    def test_register_after_hypothesize(self):
        """Should register after-hypothesize hooks."""
        manager = HookManager()
        hook = ModifyingAfterHypothesizeHook()
        
        manager.register_after_hypothesize(hook)
        
        assert len(manager.after_hypothesize_hooks) == 1
    
    def test_register_before_decide(self):
        """Should register before-decide hooks."""
        manager = HookManager()
        hook = ValidatingBeforeDecideHook()
        
        manager.register_before_decide(hook)
        
        assert len(manager.before_decide_hooks) == 1
    
    def test_register_after_decide(self):
        """Should register after-decide hooks."""
        manager = HookManager()
        hook = LoggingAfterDecideHook()
        
        manager.register_after_decide(hook)
        
        assert len(manager.after_decide_hooks) == 1
    
    def test_register_before_execute(self):
        """Should register before-execute hooks."""
        manager = HookManager()
        hook = ResourcePreparingBeforeExecuteHook()
        
        manager.register_before_execute(hook)
        
        assert len(manager.before_execute_hooks) == 1
    
    def test_register_after_execute(self):
        """Should register after-execute hooks."""
        manager = HookManager()
        hook = ResultValidatingAfterExecuteHook()
        
        manager.register_after_execute(hook)
        
        assert len(manager.after_execute_hooks) == 1
    
    def test_register_before_learn(self):
        """Should register before-learn hooks."""
        manager = HookManager()
        hook = SkippingBeforeLearnHook()
        
        manager.register_before_learn(hook)
        
        assert len(manager.before_learn_hooks) == 1
    
    def test_register_after_learn(self):
        """Should register after-learn hooks."""
        manager = HookManager()
        hook = RecordingAfterLearnHook()
        
        manager.register_after_learn(hook)
        
        assert len(manager.after_learn_hooks) == 1
    
    def test_register_on_error(self):
        """Should register error hooks."""
        manager = HookManager()
        hook = ErrorRecoveryHook()
        
        manager.register_on_error(hook)
        
        assert len(manager.error_hooks) == 1
    
    def test_chaining_registration(self):
        """Should support method chaining."""
        manager = HookManager()
        
        result = (manager
            .register_before_perceive(TrackingBeforePerceiveHook())
            .register_after_perceive(ModifyingAfterPerceiveHook())
            .register_on_error(ErrorRecoveryHook()))
        
        assert result is manager
        assert len(manager.before_perceive_hooks) == 1
        assert len(manager.after_perceive_hooks) == 1
        assert len(manager.error_hooks) == 1


# ============================================================
# Test Hook Functionality
# ============================================================

class TestHookFunctionality:
    """Test that hooks actually work when called."""
    
    def test_before_perceive_hook_called(self):
        """Before-perceive hook should be called."""
        hook = TrackingBeforePerceiveHook()
        manager = HookManager()
        manager.register_before_perceive(hook)
        
        context = ThinkingContext(raw_input="test")
        
        for h in manager.before_perceive_hooks:
            context = h.on_before_perceive(context)
        
        assert hook.called is True
        assert hook.context_received is context
    
    def test_after_perceive_hook_modifies_context(self):
        """After-perceive hook should be able to modify context."""
        hook = ModifyingAfterPerceiveHook()
        manager = HookManager()
        manager.register_after_perceive(hook)
        
        context = ThinkingContext(raw_input="test", metadata={})
        features = np.array([1.0], dtype=np.float64)
        
        for h in manager.after_perceive_hooks:
            context = h.on_after_perceive(context, features)
        
        assert context.metadata.get('after_perceive_called') is True
    
    def test_before_hypothesize_hook_filters_behaviors(self):
        """Before-hypothesize hook should filter behaviors."""
        hook = FilteringBeforeHypothesizeHook()
        manager = HookManager()
        manager.register_before_hypothesize(hook)
        
        context = ThinkingContext(raw_input="test")
        behaviors = [
            SimpleBehavior("keep1"),
            SimpleBehavior("skip_this"),
            SimpleBehavior("keep2"),
            SimpleBehavior("skip_that")
        ]
        
        for h in manager.before_hypothesize_hooks:
            context, behaviors = h.on_before_hypothesize(context, behaviors)
        
        assert len(behaviors) == 2
        assert behaviors[0].name == "keep1"
        assert behaviors[1].name == "keep2"
    
    def test_after_hypothesize_hook_modifies_fitness(self):
        """After-hypothesize hook should modify fitness scores."""
        hook = ModifyingAfterHypothesizeHook()
        manager = HookManager()
        manager.register_after_hypothesize(hook)
        
        context = ThinkingContext(raw_input="test")
        fitness_scores = {"normal": 0.5, "boost": 0.6, "other": 0.4}
        
        for h in manager.after_hypothesize_hooks:
            context = h.on_after_hypothesize(context, fitness_scores)
        
        assert context.fitness_scores['boost'] == 0.8  # 0.6 + 0.2 = 0.8
        assert context.fitness_scores['normal'] == 0.5
        assert context.fitness_scores['other'] == 0.4
    
    def test_before_decide_hook_validates(self):
        """Before-decide hook should validate fitness scores."""
        hook = ValidatingBeforeDecideHook()
        manager = HookManager()
        manager.register_before_decide(hook)
        
        context = ThinkingContext(raw_input="test")
        behaviors = [SimpleBehavior("b1"), SimpleBehavior("b2")]
        fitness_scores = [0.8, 0.3]
        
        # Should not raise
        for h in manager.before_decide_hooks:
            context, behaviors, fitness_scores = h.on_before_decide(
                context, behaviors, fitness_scores
            )
    
    def test_after_decide_hook_records_selection(self):
        """After-decide hook should record selected behavior."""
        hook = LoggingAfterDecideHook()
        manager = HookManager()
        manager.register_after_decide(hook)
        
        context = ThinkingContext(raw_input="test")
        selected = SimpleBehavior("selected_behavior")
        
        for h in manager.after_decide_hooks:
            context = h.on_after_decide(context, selected)
        
        assert hook.selected is selected
        assert hook.selected.name == "selected_behavior"
    
    def test_before_execute_hook_prepares_resources(self):
        """Before-execute hook should prepare resources."""
        hook = ResourcePreparingBeforeExecuteHook()
        manager = HookManager()
        manager.register_before_execute(hook)
        
        context = ThinkingContext(raw_input="test")
        behavior = SimpleBehavior("test")
        
        for h in manager.before_execute_hooks:
            context = h.on_before_execute(context, behavior)
        
        assert hook.prepared is True
    
    def test_after_execute_hook_validates_result(self):
        """After-execute hook should validate/modify result."""
        hook = ResultValidatingAfterExecuteHook()
        manager = HookManager()
        manager.register_after_execute(hook)
        
        context = ThinkingContext(raw_input="test", result=None)
        
        for h in manager.after_execute_hooks:
            context = h.on_after_execute(context, None)
        
        assert context.result == "default_result"
    
    def test_before_learn_hook_can_skip_learning(self):
        """Before-learn hook should be able to skip learning."""
        hook = SkippingBeforeLearnHook()
        manager = HookManager()
        manager.register_before_learn(hook)
        
        context = ThinkingContext(raw_input="test")
        expected = 10.0
        actual = 10.001  # Small error
        
        for h in manager.before_learn_hooks:
            context, expected, actual = h.on_before_learn(context, expected, actual)
        
        # Should return None for expected to indicate skip
        assert expected is None
        assert actual is None
    
    def test_after_learn_hook_records_changes(self):
        """After-learn hook should record parameter changes."""
        hook = RecordingAfterLearnHook()
        manager = HookManager()
        manager.register_after_learn(hook)
        
        context = ThinkingContext(raw_input="test")
        changes = {'w_0_0': 0.05, 'b_0': -0.01}
        
        for h in manager.after_learn_hooks:
            context = h.on_after_learn(context, changes)
        
        assert hook.changes_recorded == changes
    
    def test_error_hook_recovers(self):
        """Error hook should handle errors and provide recovery."""
        hook = ErrorRecoveryHook()
        manager = HookManager()
        manager.register_on_error(hook)
        
        context = ThinkingContext(raw_input="test")
        error = ValueError("Test error")
        stage = "execute"
        
        recovered_context = None
        for h in manager.error_hooks:
            recovered_context = h.on_error(context, error, stage)
        
        assert hook.error_handled is True
        assert recovered_context is not None
        assert recovered_context.metadata.get('error_recovered') is True
        assert recovered_context.result == "recovered_result"


# ============================================================
# Test Multiple Hooks
# ============================================================

class TestMultipleHooks:
    """Test that multiple hooks can be registered and executed in order."""
    
    def test_multiple_before_perceive_hooks(self):
        """Multiple before-perceive hooks should execute in order."""
        class SequentialHook(BeforePerceiveHook):
            def __init__(self, order: int):
                self.order = order
                self.executed = False
            
            def on_before_perceive(self, context):
                self.executed = True
                # Add order to metadata
                orders = context.metadata.get('orders', [])
                orders.append(self.order)
                new_metadata = context.metadata.copy()
                new_metadata['orders'] = orders
                return ThinkingContext(
                    raw_input=context.raw_input,
                    expected=context.expected,
                    context_params=context.context_params,
                    features=context.features,
                    fitness_scores=context.fitness_scores,
                    candidate_behaviors=context.candidate_behaviors,
                    selected_behavior=context.selected_behavior,
                    decision_confidence=context.decision_confidence,
                    result=context.result,
                    learning_occurred=context.learning_occurred,
                    parameter_changes=context.parameter_changes,
                    start_time=context.start_time,
                    stage_times=context.stage_times,
                    metadata=new_metadata,
                    error=context.error,
                )
        
        manager = HookManager()
        hook1 = SequentialHook(1)
        hook2 = SequentialHook(2)
        hook3 = SequentialHook(3)
        
        manager.register_before_perceive(hook1)
        manager.register_before_perceive(hook2)
        manager.register_before_perceive(hook3)
        
        context = ThinkingContext(raw_input="test", metadata={})
        
        for h in manager.before_perceive_hooks:
            context = h.on_before_perceive(context)
        
        assert hook1.executed is True
        assert hook2.executed is True
        assert hook3.executed is True
        assert context.metadata['orders'] == [1, 2, 3]
    
    def test_multiple_error_hooks(self):
        """Multiple error hooks should all execute."""
        class CountingErrorHook(OnErrorHook):
            def __init__(self):
                self.count = 0
            
            def on_error(self, context, error, stage):
                self.count += 1
                return context
        
        manager = HookManager()
        hook1 = CountingErrorHook()
        hook2 = CountingErrorHook()
        hook3 = CountingErrorHook()
        
        manager.register_on_error(hook1)
        manager.register_on_error(hook2)
        manager.register_on_error(hook3)
        
        context = ThinkingContext(raw_input="test")
        error = ValueError("Test")
        
        for h in manager.error_hooks:
            context = h.on_error(context, error, "test")
        
        assert hook1.count == 1
        assert hook2.count == 1
        assert hook3.count == 1