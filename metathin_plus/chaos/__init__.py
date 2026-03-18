"""
Chaos Agent Module (Optimized Version)
===================================================

Provides agents, predictors, and analysis tools for chaotic systems.
This module is built on the Metathin core framework and specialized for
chaotic time series prediction and analysis.

Version: 0.2.0 (Optimized Version)
"""

# ============================================================
# Base Class Imports
# ============================================================
from .base import (
    ChaosModel,
    """ChaosModel: Abstract base class defining the interface for all chaos prediction models.
       Provides common functionality for training, predicting, and evaluating predictions."""
    
    SystemState,
    """SystemState: Represents the state of a dynamical system in phase space.
       Encapsulates the coordinates, timestamps, and associated metadata."""
    
    StateProcessor,
    """StateProcessor: Processes raw time series data into system states.
       Handles embedding dimension selection, time delay estimation, and state normalization."""
    
    PredictionResult,
    """PredictionResult: Container for prediction results and associated metrics.
       Includes predicted values, confidence intervals, and model diagnostics."""
    
    ChaosPredictor,
    """ChaosPredictor: High-level predictor interface combining multiple prediction strategies.
       Integrates different prediction methods and selects the most appropriate one."""
)

# ============================================================
# Main Agent Import
# ============================================================
from .metachaos_basic import MetaChaosBasic
"""MetaChaosBasic: Main chaos prediction agent that orchestrates multiple predictors.
   Integrates phase space reconstruction, Volterra series, neural networks, and spectral analysis.
   Provides a unified interface for chaotic time series prediction."""

# ============================================================
# Predictor Imports - From predictors submodule
# ============================================================
from .predictors import (
    # Simplified Versions (Legacy Compatibility)
    PhaseSpacePredictor,
    """PhaseSpacePredictor (Simplified): Basic phase space reconstruction predictor.
       Uses Takens' embedding theorem to reconstruct attractor dynamics.
       Suitable for quick prototyping and educational purposes."""
    
    VolterraPredictor,
    """VolterraPredictor (Simplified): Basic Volterra series predictor.
       Uses polynomial expansion of past states for nonlinear prediction.
       Good for systems with moderate nonlinearity."""
    
    NeuralPredictor,
    """NeuralPredictor (Simplified): Basic neural network predictor.
       Simple feedforward network for chaotic time series prediction.
       Serves as a baseline neural approach."""
    
    SpectralPredictor,
    """SpectralPredictor (Simplified): Basic spectral analysis predictor.
       Uses FFT and dominant frequency extraction for prediction.
       Effective for systems with strong periodic components."""
    
    PersistentPredictor,
    """PersistentPredictor: Baseline predictor using persistence assumption.
       Simply predicts that future values equal the last observed value.
       Used as a benchmark to evaluate other predictors' performance."""
    
    LinearTrendPredictor,
    """LinearTrendPredictor: Linear trend extrapolation predictor.
       Fits a linear model to recent data and extrapolates forward.
       Useful as a simple baseline for trending data."""
    
    # Full Versions (Recommended for Production)
    FullPhaseSpacePredictor,
    """FullPhaseSpacePredictor (Complete): Advanced phase space reconstruction.
       Features:
           - Adaptive embedding dimension selection
           - Mutual information for time delay estimation
           - False nearest neighbors algorithm
           - Local and global prediction modes
       Recommended for production use."""
    
    FullVolterraPredictor,
    """FullVolterraPredictor (Complete): Advanced Volterra series predictor.
       Features:
           - Automatic order selection via cross-validation
           - Regularization to prevent overfitting
           - Recursive least squares adaptation
           - Multi-step ahead prediction capabilities
       Recommended for production use."""
    
    FullNeuralPredictor,
    """FullNeuralPredictor (Complete): Advanced neural network predictor.
       Features:
           - LSTM and GRU architectures for temporal dynamics
           - Attention mechanisms for long-range dependencies
           - Bayesian optimization for hyperparameter tuning
           - Ensemble methods for uncertainty quantification
       Recommended for production use."""
    
    FullSpectralPredictor,
    """FullSpectralPredictor (Complete): Advanced spectral analysis predictor.
       Features:
           - Wavelet decomposition for time-frequency analysis
           - Singular spectrum analysis (SSA)
           - Adaptive mode decomposition
           - Harmonic analysis with confidence bounds
       Recommended for production use."""
)

# ============================================================
# Analyzer Imports
# ============================================================
from .analyzers import (
    LyapunovEstimator,
    """LyapunovEstimator: Estimates the largest Lyapunov exponent.
       The Lyapunov exponent quantifies the rate of divergence of nearby trajectories.
       Positive exponent indicates chaos, zero indicates periodic motion,
       negative indicates stable fixed points.
       Uses algorithms: Rosenstein, Kantz, and Wolf methods."""
    
    DimensionEstimator
    """DimensionEstimator: Estimates the correlation dimension of the attractor.
       The correlation dimension measures the fractal dimension of the strange attractor.
       Helps characterize the complexity and embedding requirements.
       Uses Grassberger-Procaccia algorithm and box-counting methods."""
)

# ============================================================
# Export Interface Definition
# ============================================================
# __all__ controls what gets imported with 'from metathin_plus.chaos import *'
# Organized by category for easy reference
# ============================================================

__all__ = [
    # ===== Base Classes =====
    # Foundation for all chaos analysis components
    'ChaosModel',               # Abstract base for predictors
    'SystemState',              # Phase space state representation
    'StateProcessor',           # State processing utilities
    'PredictionResult',         # Prediction results container
    'ChaosPredictor',           # High-level predictor interface
    
    # ===== Main Agent =====
    # Primary entry point for chaos prediction
    'MetaChaosBasic',           # Main chaos prediction agent
    
    # ===== Predictors (Simplified Versions) =====
    # Lightweight predictors for quick prototyping and baseline comparisons
    'PhaseSpacePredictor',       # Basic phase space reconstruction
    'VolterraPredictor',         # Basic Volterra series
    'NeuralPredictor',           # Basic neural network
    'SpectralPredictor',         # Basic spectral analysis
    'PersistentPredictor',       # Baseline persistence model
    'LinearTrendPredictor',      # Baseline linear trend
    
    # ===== Predictors (Full Versions) =====
    # Advanced predictors with full feature sets for production use
    'FullPhaseSpacePredictor',   # Advanced phase space reconstruction
    'FullVolterraPredictor',     # Advanced Volterra series
    'FullNeuralPredictor',       # Advanced neural network
    'FullSpectralPredictor',     # Advanced spectral analysis
    
    # ===== Analyzers =====
    # Tools for characterizing chaotic dynamics
    'LyapunovEstimator',         # Lyapunov exponent estimation
    'DimensionEstimator',        # Correlation dimension estimation
]

# ============================================================
# Version Information
# ============================================================

__version__ = '0.2.0'
__author__ = 'Lydian-Zhu'