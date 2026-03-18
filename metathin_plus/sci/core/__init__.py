"""
Metathin+Sci Core Module
=====================================================

The core module provides fundamental functional components for scientific discovery,
including function generation, feature extraction, and similarity matching.

Design Philosophy:
    - Modular: Each component has a single, well-defined responsibility
    - Extensible: Supports adding new function templates and feature types
    - Efficient: Vectorized operations and caching mechanisms
    - Reproducible: Random seed control ensures results can be replicated

Core Components:
    1. FunctionGenerator - Function data generator
       Creates synthetic datasets from various mathematical function families
       
    2. FeatureExtractor - Feature extractor
       Extracts characteristic features from time series and functional data
       
    3. SimilarityMatcher - Similarity matcher
       Finds similar functions based on extracted features using distance metrics
"""

# ============================================================
# Core Component Imports
# ============================================================

# Function Generator
from .function_generator import FunctionGenerator, FunctionType, FunctionTemplate, FunctionSample
"""
FunctionGenerator: Generates synthetic data from mathematical functions.
    - Supports multiple function families (polynomial, trigonometric, exponential, etc.)
    - Configurable parameters and noise levels
    - Batch generation for training datasets
    
FunctionType: Enumeration of supported function families.
    - POLYNOMIAL: a0 + a1*x + a2*x^2 + ...
    - TRIGONOMETRIC: sin, cos, tan and combinations
    - EXPONENTIAL: exp, log, power laws
    - RATIONAL: ratios of polynomials
    - COMPOSITE: combinations of elementary functions
    
FunctionTemplate: Template for generating specific function types.
    - Parameter ranges and distributions
    - Domain specifications
    - Noise characteristics
    
FunctionSample: A single generated sample with metadata.
    - x_values: Input domain points
    - y_values: Function values
    - true_function: Underlying mathematical expression
    - parameters: Actual parameters used
    - noise_level: Amount of noise added
"""

# Feature Extractor
from .feature_extractor import FeatureExtractor, FeatureType, FeatureDefinition
"""
FeatureExtractor: Extracts characteristic features from time series.
    - Statistical features (mean, variance, skewness, kurtosis)
    - Spectral features (dominant frequencies, power distribution)
    - Dynamical features (Lyapunov exponents, correlation dimension)
    - Geometric features (curvature, monotonicity, periodicity)
    
FeatureType: Enumeration of available feature categories.
    - STATISTICAL: Basic statistical moments
    - SPECTRAL: Frequency domain features
    - DYNAMICAL: Chaos and complexity measures
    - GEOMETRIC: Shape and structure features
    
FeatureDefinition: Definition of a specific feature.
    - name: Feature identifier
    - compute_fn: Function to compute the feature
    - bounds: Expected value range
    - description: Human-readable description
"""

# Similarity Matcher
from .similarity_matcher import SimilarityMatcher, MatchResult, DistanceMetric
"""
SimilarityMatcher: Finds similar functions based on feature vectors.
    - Builds efficient index structures (KD-Tree, Ball Tree)
    - Supports multiple distance metrics
    - Returns ranked matches with confidence scores
    
MatchResult: Result of a similarity search.
    - function_type: Matched function family
    - similarity_score: Distance or similarity measure
    - confidence: Confidence in the match (0-1)
    - metadata: Additional information about the match
    
DistanceMetric: Supported distance metrics for comparison.
    - EUCLIDEAN: Standard L2 distance
    - MANHATTAN: L1 distance
    - COSINE: Cosine similarity
    - CORRELATION: Correlation-based distance
    - MAHALANOBIS: Mahalanobis distance (requires covariance)
"""

# ============================================================
# Version Information
# ============================================================

__all__ = [
    # Function Generator Related
    'FunctionGenerator',      # Main function generator class
    'FunctionType',           # Enum of function families
    'FunctionTemplate',       # Template for function generation
    'FunctionSample',         # Single generated sample
    
    # Feature Extractor Related
    'FeatureExtractor',       # Main feature extractor class
    'FeatureType',            # Enum of feature categories
    'FeatureDefinition',      # Individual feature definition
    
    # Similarity Matcher Related
    'SimilarityMatcher',      # Main similarity matcher class
    'MatchResult',            # Result of similarity search
    'DistanceMetric',         # Supported distance metrics
]

# ============================================================
# Usage Examples
# ============================================================
"""
>>> from metathin_sci.core import FunctionGenerator, FeatureExtractor, SimilarityMatcher
>>> import numpy as np
>>> 
>>> # 1. Generate training data
>>> generator = FunctionGenerator(seed=42)
>>> X, y, labels = generator.generate_batch(1000)  # X: domain, y: values, labels: function types
>>> 
>>> # 2. Extract features
>>> extractor = FeatureExtractor()
>>> features = []
>>> for i in range(len(y)):
...     feat = extractor.extract(y[i])  # Extract features from each time series
...     features.append(feat)
>>> features = np.array(features)  # Shape: (n_samples, n_features)
>>> 
>>> # 3. Build index and train matcher
>>> matcher = SimilarityMatcher(metric=DistanceMetric.EUCLIDEAN)
>>> matcher.build_index(features, labels)  # Build search index
>>> 
>>> # 4. Find similar functions for new data
>>> # Generate a test sample
>>> new_data = np.sin(np.linspace(0, 10, 100)) + 0.1 * np.random.randn(100)
>>> new_features = extractor.extract(new_data)
>>> 
>>> # Find most similar known functions
>>> similar = matcher.find_similar(new_features, k=3)
>>> for match in similar:
...     print(f"Function type: {match.function_type}, Similarity: {match.similarity_score:.4f}")
... 
>>> # 5. Full pipeline example
>>> def discover_function_type(data):
...     # Extract features
...     features = extractor.extract(data)
...     # Find matches
...     matches = matcher.find_similar(features, k=1)
...     if matches:
...         return matches[0].function_type, matches[0].confidence
...     return None, 0.0
>>> 
>>> func_type, confidence = discover_function_type(new_data)
>>> print(f"Discovered function type: {func_type} (confidence: {confidence:.2%})")
"""

# ============================================================
# Module Initialization Logging
# ============================================================

import logging
logger = logging.getLogger(__name__)
logger.debug("Metathin+Sci core module initialized successfully")