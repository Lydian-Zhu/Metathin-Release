"""
Metathin+Sci Discovery Module
=====================================================

The discovery module provides high-level components for automated scientific discovery,
including adaptive extrapolation, symbolic representation, and the main discovery agent.

This module orchestrates the entire discovery pipeline:
    1. Data ingestion and preprocessing
    2. Feature extraction and pattern recognition
    3. Function generation and hypothesis formation
    4. Validation and extrapolation
    5. Report generation and visualization
"""

from .adaptive_extrapolator import AdaptiveExtrapolator, SymbolicForm
"""
AdaptiveExtrapolator: Adaptively extrapolates discovered patterns.
    
Projects discovered functional relationships forward to predict unseen behavior.
Adjusts extrapolation strategy based on pattern type and uncertainty estimates.

Features:
    - Multiple extrapolation modes (analytic, numerical, hybrid)
    - Uncertainty quantification
    - Adaptive strategy selection
    - Confidence bounds for predictions

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
"""

from .scientific_metathin import ScientificMetathin
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

Key Capabilities:
    - Automated hypothesis generation
    - Pattern discovery in time series
    - Function fitting and validation
    - Knowledge reuse via function memory
    - Explainable AI through symbolic expressions
"""

from .report_generator import DiscoveryReport, DiscoveryPhase
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

# ============================================================
# Export Interface Definition
# ============================================================
# __all__ controls what gets imported with 'from metathin_plus.sci.discovery import *'

__all__ = [
    # Adaptive Extrapolation
    'AdaptiveExtrapolator',   # Adaptive pattern extrapolation
    'SymbolicForm',           # Symbolic expression representation
    
    # Main Discovery Agent
    'ScientificMetathin',     # Main scientific discovery agent
    
    # Reporting
    'DiscoveryReport',        # Discovery results report
    'DiscoveryPhase',         # Discovery phase enumeration
]

# ============================================================
# Usage Examples
# ============================================================
"""
>>> from metathin_plus.sci.discovery import ScientificMetathin, DiscoveryPhase
>>> from metathin_plus.sci.core import FunctionGenerator, FeatureExtractor
>>> 
>>> # 1. Initialize components
>>> generator = FunctionGenerator()
>>> extractor = FeatureExtractor()
>>> 
>>> # 2. Create discovery agent
>>> agent = ScientificMetathin(
...     function_generator=generator,
...     feature_extractor=extractor,
...     name="MyDiscoveryAgent"
... )
>>> 
>>> # 3. Generate training data
>>> X_list, y_list, labels = generator.generate_batch(1000)
>>> 
>>> # 4. Train the agent
>>> agent.fit(X_list, y_list, labels)
>>> 
>>> # 5. Discover patterns in new data
>>> new_data = np.sin(np.linspace(0, 10, 100)) + 0.1 * np.random.randn(100)
>>> report = agent.discover(new_data)
>>> 
>>> # 6. Examine results
>>> print(f"Discovered function: {report.best_function}")
>>> print(f"Confidence: {report.confidence:.2%}")
>>> print(f"Discovery phase: {report.phase.value}")
>>> 
>>> # 7. Generate visualizations
>>> report.plot_data_and_fit()
>>> report.save_report("discovery_results.json")
"""