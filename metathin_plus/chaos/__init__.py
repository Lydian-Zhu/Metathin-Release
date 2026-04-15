# metathin_plus/chaos/__init__.py - 修复导入顺序

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
# 1. Base types (no dependencies) | 基础类型（无依赖）
# ============================================================

from .base import SystemState, PredictionResult, ChaosModel

# ============================================================
# 2. Pattern Space (P) | 感知层
# ============================================================

from .pattern_space import ChaosPatternSpace

# ============================================================
# 3. Behaviors (B) | 行动层
# ============================================================

from .behaviors import (
    PhaseSpaceBehavior,
    VolterraBehavior,
    NeuralBehavior,
    SpectralBehavior,
    PersistentBehavior,
    LinearTrendBehavior,
)

# ============================================================
# 4. Selector (S) | 评估层 - 已修复
# ============================================================

from .selector import ChaosSelector

# ============================================================
# 5. Decision Strategy (D) | 决策层
# ============================================================

from .decision import (
    MinErrorStrategy,
    WeightedVoteStrategy,
    AdaptiveStrategy,
)

# ============================================================
# 6. Learning Mechanism (Ψ) | 学习层
# ============================================================

from .learning import ErrorLearning, RewardLearning

# ============================================================
# 7. Main Agent (depends on all above) | 主智能体
# ============================================================

from .metachaos import MetaChaos

# ============================================================
# Export Interface | 导出接口
# ============================================================

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

print(f"✅ Metathin+Chaos v{__version__} loaded (Metathin interface)")