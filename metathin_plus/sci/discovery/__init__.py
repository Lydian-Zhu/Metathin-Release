# metathin_sci/discovery/__init__.py
"""
Metathin+Sci Discovery Module | 发现模块
=========================================

Scientific discovery components for the Metathin framework.
Adapted to the quintuple (P, B, S, D, Ψ) architecture.

科学发现组件，适配五元组 (P, B, S, D, Ψ) 架构。
"""

from .adaptive_extrapolator import AdaptiveExtrapolator, SymbolicForm, SymbolicLibrary
from .scientific_metathin import ScientificMetathin
from .report_generator import DiscoveryReport, DiscoveryPhase

__all__ = [
    'AdaptiveExtrapolator',
    'SymbolicForm',
    'SymbolicLibrary',
    'ScientificMetathin',
    'DiscoveryReport',
    'DiscoveryPhase',
]