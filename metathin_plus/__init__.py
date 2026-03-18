"""
Metathin+ - Pre-built Agent Library
Provides ready-to-use specialized agents
===================================================

Contains two core modules:
    - chaos: Chaos prediction module for analyzing and forecasting chaotic systems
    - sci: Scientific discovery module for automated knowledge discovery and function finding

Version: 0.3.0
"""

__version__ = '0.3.0'
__author__ = 'Lydian-Zhu'
__license__ = 'MIT'

# ============================================================
# Chaos Module Imports
# ============================================================
# The chaos module provides tools for analyzing and predicting chaotic dynamical systems.
# It includes various prediction algorithms based on different mathematical principles.

from .chaos import MetaChaosBasic
"""MetaChaosBasic: Main chaos prediction agent that integrates multiple predictors"""

from .chaos import (
    PhaseSpacePredictor,
    """Phase Space Predictor: Uses phase space reconstruction for prediction
       Based on Takens' embedding theorem, reconstructs the system's attractor
       from time series data"""
    
    VolterraPredictor,
    """Volterra Predictor: Uses Volterra series expansion for nonlinear prediction
       Captures nonlinear dynamics through polynomial expansion of past states"""
    
    NeuralPredictor,
    """Neural Predictor: Uses neural networks for chaotic time series prediction
       Leverages deep learning to capture complex nonlinear patterns"""
    
    SpectralPredictor,
    """Spectral Predictor: Uses spectral analysis for prediction
       Analyzes frequency components and identifies dominant oscillatory modes"""
    
    PersistentPredictor,
    """Persistent Predictor: Baseline predictor using persistence assumption
       Assumes the system will remain in its current state - useful as benchmark"""
    
    LinearTrendPredictor
    """Linear Trend Predictor: Uses linear extrapolation for prediction
       Fits a linear trend to recent data and extrapolates forward"""
)

from .chaos.base import (
    ChaosModel,
    """ChaosModel: Abstract base class for all chaos prediction models
       Defines the interface for training and predicting"""
    
    SystemState,
    """SystemState: Represents the state of a dynamical system
       Encapsulates the phase space coordinates and associated metadata"""
    
    StateProcessor,
    """StateProcessor: Processes raw time series into system states
       Handles embedding, normalization, and feature extraction"""
    
    PredictionResult
    """PredictionResult: Encapsulates prediction results and confidence metrics
       Contains predicted values, uncertainty estimates, and model diagnostics"""
)

# ============================================================
# Scientific Discovery Module Imports
# ============================================================
# The sci module provides tools for automated scientific discovery,
# including function finding, pattern recognition, and hypothesis generation.

from .sci import (
    # Core Components
    FunctionGenerator,
    """FunctionGenerator: Generates candidate mathematical functions
       Uses genetic programming or symbolic regression to create function expressions"""
    
    FeatureExtractor,
    """FeatureExtractor: Extracts mathematical features from data
       Identifies relevant patterns, symmetries, and invariants in the data"""
    
    SimilarityMatcher,
    """SimilarityMatcher: Matches patterns against known function families
       Compares extracted features with library of known mathematical forms"""
    
    # Memory Components
    FunctionMemoryBank,
    """FunctionMemoryBank: Persistent storage for discovered functions
       Maintains a library of previously discovered mathematical relationships"""
    
    FunctionMemory,
    """FunctionMemory: Individual memory entry for a discovered function
       Stores function expression, parameters, and discovery metadata"""
    
    # Discovery Components
    AdaptiveExtrapolator,
    """AdaptiveExtrapolator: Adaptively extrapolates discovered patterns
       Projects discovered functions forward to predict unseen behavior"""
    
    ScientificMetathin,
    """ScientificMetathin: Main agent for automated scientific discovery
       Orchestrates the entire discovery process from data to hypothesis"""
    
    DiscoveryReport,
    """DiscoveryReport: Comprehensive report of discovery results
       Contains discovered functions, confidence metrics, and visualizations"""
    
    DiscoveryPhase
    """DiscoveryPhase: Enumeration of discovery process phases
       Tracks progress through exploration, pattern recognition, and validation"""
)

# ============================================================
# Export Interface Definition
# ============================================================
# __all__ controls what gets imported with 'from metathin_plus import *'
# Organized by module for easy reference
# ============================================================

__all__ = [
    # ===== Chaos Module =====
    # Main Agent
    'MetaChaosBasic',           # Main chaos prediction agent
    
    # Predictors
    'PhaseSpacePredictor',       # Phase space reconstruction predictor
    'VolterraPredictor',         # Volterra series predictor
    'NeuralPredictor',           # Neural network predictor
    'SpectralPredictor',         # Spectral analysis predictor
    'PersistentPredictor',       # Baseline persistence predictor
    'LinearTrendPredictor',      # Linear trend predictor
    
    # Base Classes and Utilities
    'ChaosModel',                # Abstract base class for predictors
    'SystemState',               # System state representation
    'StateProcessor',            # State processing utilities
    'PredictionResult',          # Prediction results container
    
    # ===== Scientific Discovery Module =====
    # Core Components
    'FunctionGenerator',         # Mathematical function generator
    'FeatureExtractor',          # Mathematical feature extractor
    'SimilarityMatcher',         # Pattern matching utilities
    
    # Memory Components
    'FunctionMemoryBank',        # Function memory storage
    'FunctionMemory',            # Individual function memory entry
    
    # Discovery Components
    'AdaptiveExtrapolator',      # Adaptive pattern extrapolation
    'ScientificMetathin',        # Main scientific discovery agent
    'DiscoveryReport',           # Discovery results report
    'DiscoveryPhase',            # Discovery phase enumeration
]

# ============================================================
# Module Information
# ============================================================
# Provides feedback about successful module loading
# and available submodules

print(f"✅ Metathin+ v{__version__} loaded successfully")
print(f"   Available modules: chaos (chaos prediction), sci (scientific discovery)")