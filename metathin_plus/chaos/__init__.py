"""
Metathin+ Chaos Module - Adapted to Metathin Quintuple Interface
================================================================
"""

__version__ = '0.5.0'
__author__ = 'Lydian-Zhu'
__license__ = 'MIT'

# Base types
from .base import SystemState, PredictionResult, ChaosModel

# P - Pattern Space
from .pattern_space import ChaosPatternSpace

# B - Behaviors
from .behaviors import (
    PhaseSpaceBehavior,
    VolterraBehavior,
    NeuralBehavior,
    SpectralBehavior,
    PersistentBehavior,
    LinearTrendBehavior,
)

# S - Selector
from .selector import ChaosSelector

# D - Decision Strategy
from .decision import (
    MinErrorStrategy,
    WeightedVoteStrategy,
    AdaptiveStrategy,
)

# Ψ - Learning
from .learning import ErrorLearning, RewardLearning

# Main Agent
from .metachaos import MetaChaos

__all__ = [
    '__version__',
    'SystemState',
    'PredictionResult',
    'ChaosModel',
    'ChaosPatternSpace',
    'PhaseSpaceBehavior',
    'VolterraBehavior',
    'NeuralBehavior',
    'SpectralBehavior',
    'PersistentBehavior',
    'LinearTrendBehavior',
    'ChaosSelector',
    'MinErrorStrategy',
    'WeightedVoteStrategy',
    'AdaptiveStrategy',
    'ErrorLearning',
    'RewardLearning',
    'MetaChaos',
]

print(f"✅ Metathin+Chaos v{__version__} loaded")