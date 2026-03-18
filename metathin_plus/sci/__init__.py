"""
Metathin+Sci - Scientific Discovery Agent
=====================================================

A specialized agent for automated scientific discovery, capable of identifying mathematical
patterns, discovering functional relationships, and generating hypotheses from data.

Version: 0.1.0
"""

__version__ = '0.1.0'
__author__ = 'Lydian-Zhu'
__license__ = 'MIT'

# ============================================================
# Core Module Imports
# ============================================================

from .core import (
    FunctionGenerator,
    """
    FunctionGenerator: Generates candidate mathematical functions.
    
    Uses genetic programming and symbolic regression techniques to evolve mathematical
    expressions that fit the observed data. Capable of discovering both simple and
    complex functional forms.
    
    Features:
        - Genetic programming for expression evolution
        - Multiple expression representations (trees, strings)
        - Fitness evaluation based on data fit
        - Complexity regularization to avoid overfitting
    """,
    
    FeatureExtractor,
    """
    FeatureExtractor: Extracts mathematical features from data.
    
    Analyzes time series to identify characteristic patterns, symmetries, and invariants.
    These features guide the discovery process by suggesting candidate function families.
    
    Features:
        - Statistical features (mean, variance, skewness, kurtosis)
        - Spectral features (dominant frequencies, power distribution)
        - Dynamical features (Lyapunov exponents, correlation dimension)
        - Geometric features (curvature, monotonicity, periodicity)
    """,
    
    SimilarityMatcher
    """
    SimilarityMatcher: Matches patterns against known function families.
    
    Compares extracted features with a library of known mathematical forms to identify
    potential matches. Accelerates discovery by leveraging existing mathematical knowledge.
    
    Features:
        - Multi-scale pattern matching
        - Feature-based similarity metrics
        - Known function library (polynomials, exponentials, trigonometric, etc.)
        - Confidence scoring for matches
    """
)

# ============================================================
# Memory Module Imports
# ============================================================

from .memory import (
    FunctionMemory,
    """
    FunctionMemory: Individual memory entry for a discovered function.
    
    Stores a complete record of a discovered mathematical relationship, including the
    expression, parameters, fitting quality, and discovery metadata.
    
    Attributes:
        - expression: Symbolic representation of the function
        - parameters: Estimated parameters
        - fit_quality: Goodness-of-fit metrics (R², MSE, etc.)
        - timestamp: Discovery time
        - context: Conditions under which the function was discovered
    """,
    
    FunctionMemoryBank
    """
    FunctionMemoryBank: Persistent storage for discovered functions.
    
    Maintains a library of previously discovered mathematical relationships, enabling
    reuse and cross-referencing across different discovery tasks.
    
    Features:
        - Persistent storage (JSON/SQLite backends)
        - Semantic search by pattern similarity
        - Version tracking for function evolution
        - Export/import capabilities for sharing discoveries
    """
)

# ============================================================
# Discovery Module Imports
# ============================================================

from .discovery import (
    AdaptiveExtrapolator,
    """
    AdaptiveExtrapolator: Adaptively extrapolates discovered patterns.
    
    Projects discovered functional relationships forward to predict unseen behavior.
    Adjusts extrapolation strategy based on pattern type and uncertainty estimates.
    
    Features:
        - Multiple extrapolation modes (analytic, numerical, hybrid)
        - Uncertainty quantification
        - Adaptive strategy selection
        - Confidence bounds for predictions
    """,
    
    SymbolicForm,
    """
    SymbolicForm: Symbolic representation of mathematical expressions.
    
    Encapsulates mathematical expressions in a form suitable for manipulation,
    evaluation, and transformation. Supports basic arithmetic operations, elementary
    functions, and composition.
    
    Features:
        - Tree-based representation
        - Evaluation with numpy backend
        - Symbolic differentiation
        - Expression simplification
        - LaTeX generation for publication
    """,
    
    ScientificMetathin,
    """
    ScientificMetathin: Main agent for automated scientific discovery.
    
    Orchestrates the entire discovery process, from data ingestion to hypothesis
    generation. Integrates all components (generator, extractor, matcher, memory)
    into a coherent discovery pipeline.
    
    Discovery Pipeline:
        1. Ingest and preprocess data
        2. Extract features and patterns
        3. Search for known patterns in memory
        4. Generate candidate functions
        5. Evaluate and validate candidates
        6. Extrapolate and predict
        7. Generate discovery report
    """,
    
    DiscoveryReport,
    """
    DiscoveryReport: Comprehensive report of discovery results.
    
    Encapsulates all findings from a discovery run, including discovered functions,
    confidence metrics, visualizations, and supporting evidence.
    
    Components:
        - Discovered functions with parameters
        - Fit quality metrics (R², AIC, BIC)
        - Confidence intervals
        - Visualizations of data and fits
        - Comparison with known functions
        - Recommendations for further investigation
    """,
    
    DiscoveryPhase
    """
    DiscoveryPhase: Enumeration of discovery process phases.
    
    Tracks the progress through the scientific discovery pipeline, providing
    visibility into the current stage and enabling phase-specific logging.
    
    Phases:
        - INITIALIZATION: Setting up the discovery task
        - FEATURE_EXTRACTION: Analyzing input data
        - PATTERN_MATCHING: Searching known patterns
        - HYPOTHESIS_GENERATION: Creating candidate functions
        - VALIDATION: Testing candidate functions
        - EXTRAPOLATION: Extending discovered patterns
        - REPORTING: Generating discovery outputs
        - COMPLETED: Discovery process finished
        - ERROR: Error occurred during discovery
    """
)

# ============================================================
# Export Interface Definition
# ============================================================
# __all__ controls what gets imported with 'from metathin_plus.sci import *'
# Organized by module for easy reference
# ============================================================

__all__ = [
    # Version Information
    '__version__',
    '__author__',
    '__license__',
    
    # Core Modules
    'FunctionGenerator',      # Genetic programming function generator
    'FeatureExtractor',       # Mathematical feature extractor
    'SimilarityMatcher',      # Pattern matching utilities
    
    # Memory Modules
    'FunctionMemory',         # Individual function memory entry
    'FunctionMemoryBank',     # Persistent function library
    
    # Discovery Modules
    'AdaptiveExtrapolator',   # Adaptive pattern extrapolation
    'SymbolicForm',           # Symbolic expression representation
    'ScientificMetathin',     # Main scientific discovery agent
    'DiscoveryReport',        # Discovery results report
    'DiscoveryPhase',         # Discovery phase enumeration
]

# ============================================================
# Initialization Information
# ============================================================

print(f"✅ Metathin+Sci v{__version__} loaded successfully")
print(f"   Scientific discovery agent ready. Type 'help(sci)' for documentation")