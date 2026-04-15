# metathin_plus/chaos/__init__.py
"""
Metathin+ Chaos Module - Adapted to Metathin Quintuple Interface
================================================================

Chaos prediction module refactored to use Metathin core interfaces (P, B, S, D, Ψ).

Version: 0.5.0 (Refactored)
"""

__version__ = '0.5.0'
__author__ = 'Lydian-Zhu'
__license__ = 'MIT'

# ============================================================
# Core Exports | 核心导出
# ============================================================

from .base import SystemState, PredictionResult, ChaosModel

# ============================================================
# Quintuple Components | 五元组件
# ============================================================

# P - Pattern Space | 感知层
from .pattern_space import ChaosPatternSpace

# B - Behaviors | 行动层
from .behaviors import (
    PhaseSpaceBehavior,
    VolterraBehavior,
    NeuralBehavior,
    SpectralBehavior,
    PersistentBehavior,
    LinearTrendBehavior,
)

# S - Selector | 评估层
from .selector import ChaosSelector

# D - Decision Strategy | 决策层
from .decision import (
    MinErrorStrategy,
    WeightedVoteStrategy,
    AdaptiveStrategy,
)

# Ψ - Learning Mechanism | 学习层
from .learning import ErrorLearning, RewardLearning

# ============================================================
# Main Agent | 主智能体
# ============================================================

from .metachaos import MetaChaos

# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    # Version
    '__version__',
    
    # Base
    'SystemState',
    'PredictionResult',
    'ChaosModel',
    
    # P - Pattern Space
    'ChaosPatternSpace',
    
    # B - Behaviors
    'PhaseSpaceBehavior',
    'VolterraBehavior',
    'NeuralBehavior',
    'SpectralBehavior',
    'PersistentBehavior',
    'LinearTrendBehavior',
    
    # S - Selector
    'ChaosSelector',
    
    # D - Decision Strategy
    'MinErrorStrategy',
    'WeightedVoteStrategy',
    'AdaptiveStrategy',
    
    # Ψ - Learning
    'ErrorLearning',
    'RewardLearning',
    
    # Main Agent
    'MetaChaos',
]

print(f"✅ Metathin+Chaos v{__version__} loaded (Metathin interface)")