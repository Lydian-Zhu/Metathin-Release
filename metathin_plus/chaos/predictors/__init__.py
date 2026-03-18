"""
Built-in Prediction Methods - Full Versions (Optimized Edition)
======================================================================

This module provides complete chaotic time series prediction methods. All predictors inherit
from the ChaosPredictor base class. These are full-featured versions that implement complete
algorithms, significantly outperforming the simplified versions.

Version: 0.2.0 (Optimized Edition)
"""

import logging

# ============================================================
# Full Version Predictor Imports
# ============================================================

# Phase Space Predictors
from .phase_space import FullPhaseSpacePredictor, PhaseSpacePredictor
"""
FullPhaseSpacePredictor: Complete phase space reconstruction predictor.
    Features:
        - Adaptive embedding dimension selection using false nearest neighbors
        - Mutual information for optimal time delay estimation
        - Local and global prediction modes
        - Confidence interval estimation
        - Handles both deterministic and stochastic components

PhaseSpacePredictor: Simplified version for compatibility.
    Basic phase space reconstruction without advanced features.
    Suitable for quick testing and educational purposes.
"""

# Volterra Predictors
from .volterra import FullVolterraPredictor, VolterraPredictor
"""
FullVolterraPredictor: Complete Volterra series predictor.
    Features:
        - Automatic order selection via cross-validation
        - L2 regularization to prevent overfitting
        - Recursive least squares adaptation
        - Multi-step ahead prediction
        - Handles nonlinear dynamics up to specified order

VolterraPredictor: Simplified version for compatibility.
    Basic Volterra series with fixed order.
"""

# Neural Network Predictors
from .neural import FullNeuralPredictor, NeuralPredictor
"""
FullNeuralPredictor: Complete neural network predictor.
    Features:
        - LSTM and GRU architectures for temporal dependencies
        - Attention mechanisms for long-range patterns
        - Bayesian optimization for hyperparameter tuning
        - Ensemble methods for uncertainty quantification
        - Automatic architecture search

NeuralPredictor: Simplified version for compatibility.
    Basic feedforward neural network.
"""

# Spectral Predictors
from .spectral import FullSpectralPredictor, SpectralPredictor
"""
FullSpectralPredictor: Complete spectral analysis predictor.
    Features:
        - Wavelet decomposition for time-frequency analysis
        - Singular spectrum analysis (SSA)
        - Empirical mode decomposition (EMD)
        - Harmonic analysis with confidence bounds
        - Adaptive filtering

SpectralPredictor: Simplified version for compatibility.
    Basic FFT-based spectral analysis.
"""

# Baseline Methods
from .baseline import PersistentPredictor, LinearTrendPredictor
"""
PersistentPredictor: Baseline predictor using persistence assumption.
    Simply predicts that future values equal the last observed value.
    Essential as a benchmark to evaluate other predictors' performance.
    Formula: y_hat_{t+h} = y_t

LinearTrendPredictor: Linear trend extrapolation predictor.
    Fits a linear model to recent data and extrapolates forward.
    Useful as a simple baseline for trending data.
    Formula: y_hat = a * t + b
"""

# Configure logging
logger = logging.getLogger("metathin_plus.chaos.predictors")
logger.info("Chaos prediction module loaded successfully")

# ============================================================
# Export Interface Definition
# ============================================================
# __all__ controls what gets imported with 'from metathin_plus.chaos.predictors import *'
# Organized by version type for clarity
# ============================================================

__all__ = [
    # ===== Full Versions (Recommended for Production) =====
    # These implement complete algorithms with all features
    'FullPhaseSpacePredictor',   # Complete phase space reconstruction
    'FullVolterraPredictor',     # Complete Volterra series
    'FullNeuralPredictor',       # Complete neural network
    'FullSpectralPredictor',     # Complete spectral analysis
    
    # ===== Simplified Versions (Legacy Compatibility) =====
    # Basic implementations for backward compatibility and quick testing
    'PhaseSpacePredictor',       # Simplified phase space
    'VolterraPredictor',         # Simplified Volterra
    'NeuralPredictor',           # Simplified neural network
    'SpectralPredictor',         # Simplified spectral
    
    # ===== Baseline Methods (Always Simplified) =====
    # These are inherently simple and don't need full versions
    'PersistentPredictor',       # Persistence prediction (baseline)
    'LinearTrendPredictor',      # Linear trend extrapolation (baseline)
]

# ============================================================
# Usage Examples
# ============================================================
"""
Basic usage examples:

    # Full version with all features
    from metathin_plus.chaos.predictors import FullPhaseSpacePredictor
    predictor = FullPhaseSpacePredictor(
        embedding_dim=10,
        time_delay=2,
        prediction_mode='local'
    )
    predictions = predictor.predict(time_series, steps=100)

    # Simplified version for quick testing
    from metathin_plus.chaos.predictors import PhaseSpacePredictor
    predictor = PhaseSpacePredictor(embedding_dim=5)
    predictions = predictor.predict(time_series, steps=50)

    # Baseline for comparison
    from metathin_plus.chaos.predictors import PersistentPredictor
    baseline = PersistentPredictor()
    baseline_predictions = baseline.predict(time_series, steps=100)
"""