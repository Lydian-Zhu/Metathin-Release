"""
Pattern Space Components
========================

Provides various feature extractors that convert raw input into feature vectors.
The pattern space serves as the agent's "eyes," responsible for perceiving and understanding the external world.

Design Philosophy:
    - Flexibility: Supports arbitrary input types (text, numerical, images, etc.)
    - Composability: Multiple pattern spaces can be combined to handle multimodal input
    - Observable: Provides feature names and dimension information for debugging and visualization
    - Robustness: Handles exceptional inputs and edge cases to ensure system stability

The feature vector q = (q1, q2, ..., qs) forms the foundation for agent decision-making,
and all subsequent components (selectors, decision strategies, etc.) operate on these features.

Feature Extractor Types:
    - SimplePatternSpace: Flexible extraction based on user-defined functions
    - StatisticalPatternSpace: Automatically computes statistical features (mean, variance, etc.)
    - NormalizedPatternSpace: Normalizes features to a consistent range
    - CompositePatternSpace: Combines multiple feature extractors
    - CachedPatternSpace: Caches feature extraction results for efficiency
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from abc import ABC, abstractmethod
import logging
import warnings

from ..core.interfaces import PatternSpace, FeatureVector, PatternExtractionError


# ============================================================
# Helper Functions
# ============================================================
# These functions safely handle various types of input data
# ============================================================

def safe_float_convert(value: Any) -> float:
    """
    Safely convert any value to float.
    
    Handles various input types, ensuring a valid float is returned.
    
    Args:
        value: Value to convert
        
    Returns:
        float: Converted float value, 0.0 if conversion fails
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_array_convert(data: Any) -> np.ndarray:
    """
    Safely convert any data to numpy array.
    
    Handles different data types, ensuring a float64 numpy array is returned.
    
    Args:
        data: Data to convert
        
    Returns:
        np.ndarray: Converted numpy array
    """
    if isinstance(data, np.ndarray):
        return data.astype(np.float64)
    elif isinstance(data, (list, tuple)):
        return np.array([safe_float_convert(x) for x in data], dtype=np.float64)
    elif isinstance(data, (int, float)):
        return np.array([float(data)], dtype=np.float64)
    elif isinstance(data, str):
        # Special handling for strings: return string length
        return np.array([float(len(data))], dtype=np.float64)
    else:
        return np.array([0.0], dtype=np.float64)


# ============================================================
# Simple Pattern Space
# ============================================================

class SimplePatternSpace(PatternSpace):
    """
    Simple Pattern Space: Function-based feature extraction.
    
    Uses user-provided functions to extract features from input, offering maximum flexibility.
    Users can define arbitrary feature extraction logic, returning a list of feature values.
    
    Characteristics:
        - Flexible: Users can define any feature extraction function
        - Simple: Only need to implement a single function
        - Describable: Can specify feature names
    
    Example:
        >>> # Extract text length and word count
        >>> pattern = SimplePatternSpace(
        ...     lambda text: [len(text), len(text.split())],
        ...     feature_names=['char_count', 'word_count'],
        ...     name="text_features"
        ... )
        >>> 
        >>> features = pattern.extract("hello world")
        >>> print(features)  # [11, 2]
        >>> print(pattern.get_feature_names())  # ['char_count', 'word_count']
    """
    
    def __init__(self, 
                 extract_func: Callable[[Any], List[float]],
                 feature_names: Optional[List[str]] = None,
                 name: str = "SimplePattern"):
        """
        Initialize simple pattern space.
        
        Args:
            extract_func: Feature extraction function, receives raw input and returns list of feature values
            feature_names: Feature names list, length must match extract_func's return value
            name: Pattern space name for logging and debugging
            
        Raises:
            TypeError: If extract_func is not callable
        """
        super().__init__()
        
        if not callable(extract_func):
            raise TypeError(f"extract_func must be callable, got {type(extract_func)}")
        
        self._extract_func = extract_func
        self._name = name
        self._feature_names = feature_names or []
        self._cached_dim = None
        self._cached_names = None
        self._logger = logging.getLogger(f"metathin.pattern.{name}")
    
    def extract(self, raw_input: Any) -> FeatureVector:
        """
        Execute feature extraction.
        
        Calls the user-provided function and performs necessary type conversion and validation.
        
        Args:
            raw_input: Raw input data
            
        Returns:
            FeatureVector: Extracted feature vector
            
        Raises:
            PatternExtractionError: If feature extraction fails
        """
        try:
            self._logger.debug(f"Extracting features: input type={type(raw_input).__name__}")
            
            # Call user function
            features = self._extract_func(raw_input)
            
            # Ensure it's a list
            if not isinstance(features, (list, tuple)):
                features = [features]
            
            # Convert to floats
            features = [safe_float_convert(f) for f in features]
            
            # Convert to numpy array
            result = np.array(features, dtype=np.float64)
            
            # Update cached dimension
            if self._cached_dim is None:
                self._cached_dim = len(result)
                self._logger.debug(f"Feature dimension: {self._cached_dim}")
            
            return result
            
        except Exception as e:
            self._logger.error(f"Feature extraction failed: {e}")
            raise PatternExtractionError(f"Feature extraction failed: {e}") from e
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        if self._feature_names:
            return self._feature_names
        
        # Generate default names
        if self._cached_dim is None:
            return []
        
        return [f"{self._name}_{i}" for i in range(self._cached_dim)]


# ============================================================
# Statistical Pattern Space
# ============================================================

class StatisticalPatternSpace(PatternSpace):
    """
    Statistical Pattern Space: Extracts statistical features from numerical sequences.
    
    Automatically computes various statistics from input data, such as mean, standard deviation, max, etc.
    Particularly suitable for time series data or numerical arrays.
    
    Characteristics:
        - Automated: Automatically computes various statistical measures
        - Rich: Provides multiple statistical feature options
        - Customizable: Can select which statistics to include
    
    Available statistical features:
        - 'mean': Arithmetic mean
        - 'std': Standard deviation
        - 'var': Variance
        - 'max': Maximum value
        - 'min': Minimum value
        - 'median': Median value
        - 'sum': Sum of all values
        - 'count': Number of values
        - 'skew': Skewness (asymmetry of distribution)
        - 'kurtosis': Kurtosis (peakedness of distribution)
        - 'q1': First quartile (25th percentile)
        - 'q3': Third quartile (75th percentile)
        - 'range': Range (max - min)
        - 'iqr': Interquartile range (q3 - q1)
    """
    
    # Available statistical features and their computation functions
    _FEATURE_FUNCS = {
        'mean': np.mean,
        'std': np.std,
        'var': np.var,
        'max': np.max,
        'min': np.min,
        'median': np.median,
        'sum': np.sum,
        'count': len,
        'skew': lambda x: float(np.mean((x - np.mean(x)) ** 3) / (np.std(x) ** 3 + 1e-8)),
        'kurtosis': lambda x: float(np.mean((x - np.mean(x)) ** 4) / (np.var(x) ** 2 + 1e-8) - 3),
        'q1': lambda x: float(np.percentile(x, 25)),
        'q3': lambda x: float(np.percentile(x, 75)),
        'range': lambda x: float(np.max(x) - np.min(x)),
        'iqr': lambda x: float(np.percentile(x, 75) - np.percentile(x, 25)),
    }
    
    def __init__(self, 
                 features: Optional[List[str]] = None,
                 name: str = "StatisticalPattern",
                 handle_empty: str = 'zeros'):
        """
        Initialize statistical pattern space.
        
        Args:
            features: List of statistical features to extract, None means use all features
            name: Pattern space name
            handle_empty: How to handle empty input
                - 'zeros': Return zero vector
                - 'raise': Raise an exception
                
        Raises:
            ValueError: If an unknown feature name is provided
        """
        super().__init__()
        
        # Set feature list
        if features is None:
            self._features = list(self._FEATURE_FUNCS.keys())
        else:
            # Validate feature names
            for f in features:
                if f not in self._FEATURE_FUNCS:
                    raise ValueError(f"Unknown statistical feature: {f}")
            self._features = features
        
        self._name = name
        self._handle_empty = handle_empty
        self._logger = logging.getLogger(f"metathin.pattern.{name}")
    
    def _prepare_data(self, raw_input: Any) -> np.ndarray:
        """Prepare data: convert input to numpy array."""
        # Convert to array
        if isinstance(raw_input, np.ndarray):
            data = raw_input.flatten()
        elif isinstance(raw_input, (list, tuple)):
            data = np.array([safe_float_convert(x) for x in raw_input], dtype=np.float64)
        elif isinstance(raw_input, (int, float)):
            data = np.array([float(raw_input)], dtype=np.float64)
        else:
            data = np.array([0.0], dtype=np.float64)
        
        # Handle empty data
        if len(data) == 0:
            if self._handle_empty == 'raise':
                raise PatternExtractionError("Input data is empty")
            else:
                self._logger.warning("Input data is empty, returning zero vector")
                return np.zeros(1, dtype=np.float64)
        
        return data
    
    def extract(self, raw_input: Any) -> FeatureVector:
        """
        Extract statistical features.
        
        Args:
            raw_input: Raw input data
            
        Returns:
            FeatureVector: Extracted statistical features
            
        Raises:
            PatternExtractionError: If feature extraction fails
        """
        try:
            self._logger.debug(f"Extracting statistical features: input type={type(raw_input).__name__}")
            
            # Prepare data
            data = self._prepare_data(raw_input)
            
            # Compute each statistical feature
            features = []
            for f_name in self._features:
                func = self._FEATURE_FUNCS[f_name]
                try:
                    value = func(data)
                    features.append(float(value))
                except Exception as e:
                    self._logger.debug(f"Computing feature '{f_name}' failed: {e}")
                    features.append(0.0)
            
            result = np.array(features, dtype=np.float64)
            self._logger.debug(f"Statistical features: {dict(zip(self._features, result))}")
            
            return result
            
        except Exception as e:
            self._logger.error(f"Statistical feature extraction failed: {e}")
            raise PatternExtractionError(f"Statistical feature extraction failed: {e}") from e
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return [f"{self._name}.{f}" for f in self._features]


# ============================================================
# Normalized Pattern Space
# ============================================================

class NormalizedPatternSpace(PatternSpace):
    """
    Normalized Pattern Space: Normalizes features from another pattern space.
    
    Maps feature values to a consistent range (e.g., [0,1]), improving compatibility across
    features with different scales. Supports multiple normalization methods.
    
    Characteristics:
        - Composable: Wraps another pattern space
        - Multi-method: Supports various normalization strategies
        - Adaptive: Can dynamically adjust normalization ranges based on data
    
    Example:
        >>> base = SimplePatternSpace(lambda x: [x, x**2])
        >>> norm = NormalizedPatternSpace(base, method='standard')
        >>> features = norm.extract(5)
    """
    
    def __init__(self, 
                 base_pattern: PatternSpace,
                 ranges: Optional[List[Tuple[float, float]]] = None,
                 method: str = 'fixed'):
        """
        Initialize normalized pattern space.
        
        Args:
            base_pattern: Base pattern space
            ranges: List of normalization ranges, each feature gets (min, max), only for 'fixed' method
            method: Normalization method
                - 'fixed': Fixed range normalization
                - 'adaptive': Adaptive range normalization (dynamically updated)
                - 'standard': Standardization (z-score)
                
        Raises:
            TypeError: If base_pattern is not a PatternSpace instance
            ValueError: If ranges length doesn't match base feature dimension
        """
        super().__init__()
        
        if not isinstance(base_pattern, PatternSpace):
            raise TypeError(f"base_pattern must be a PatternSpace instance")
        
        self._base = base_pattern
        self._method = method
        self._logger = logging.getLogger("metathin.pattern.NormalizedPatternSpace")
        
        # Get base feature dimension
        try:
            base_dim = base_pattern.get_feature_dimension()
            if base_dim == 0:
                sample_features = base_pattern.extract("sample")
                base_dim = len(sample_features)
        except:
            base_dim = 1
        
        # Initialize ranges
        if method == 'fixed':
            if ranges is None:
                self._ranges = [(0.0, 1.0)] * base_dim
                self._logger.warning(f"method='fixed' but no ranges provided, using default")
            else:
                if len(ranges) != base_dim:
                    raise ValueError(f"Number of ranges ({len(ranges)}) doesn't match base feature dimension ({base_dim})")
                self._ranges = ranges
            
            self._mins = np.array([r[0] for r in self._ranges])
            self._maxs = np.array([r[1] for r in self._ranges])
        else:
            self._ranges = [(0.0, 1.0)] * base_dim
            self._mins = np.zeros(base_dim)
            self._maxs = np.ones(base_dim)
            self._means = np.zeros(base_dim)
            self._stds = np.ones(base_dim)
            self._count = 0
    
    def extract(self, raw_input: Any) -> FeatureVector:
        """
        Extract and normalize features.
        
        Args:
            raw_input: Raw input data
            
        Returns:
            FeatureVector: Normalized feature vector
            
        Raises:
            PatternExtractionError: If normalization fails
        """
        try:
            # Extract base features
            features = self._base.extract(raw_input)
            
            # Ensure 1-dimensional
            if features.ndim > 1:
                features = features.flatten()
            
            # Normalize based on method
            if self._method == 'fixed':
                normalized = self._normalize_fixed(features)
            elif self._method == 'adaptive':
                normalized = self._normalize_adaptive(features)
            elif self._method == 'standard':
                normalized = self._normalize_standard(features)
            else:
                normalized = features
            
            # Ensure within [0,1] range
            normalized = np.clip(normalized, 0.0, 1.0)
            
            return normalized
            
        except Exception as e:
            self._logger.error(f"Normalization failed: {e}")
            raise PatternExtractionError(f"Normalization failed: {e}") from e
    
    def _normalize_fixed(self, features: np.ndarray) -> np.ndarray:
        """Fixed range normalization: linear mapping to [0,1]."""
        normalized = np.zeros_like(features)
        for i in range(min(len(features), len(self._mins))):
            min_val, max_val = self._mins[i], self._maxs[i]
            if max_val > min_val:
                normalized[i] = (features[i] - min_val) / (max_val - min_val)
            else:
                normalized[i] = 0.5
        return normalized
    
    def _normalize_adaptive(self, features: np.ndarray) -> np.ndarray:
        """Adaptive range normalization: dynamically update min/max."""
        # Update ranges
        self._count += 1
        for i in range(min(len(features), len(self._mins))):
            self._mins[i] = min(self._mins[i], features[i])
            self._maxs[i] = max(self._maxs[i], features[i])
        
        # Normalize
        normalized = np.zeros_like(features)
        for i in range(min(len(features), len(self._mins))):
            if self._maxs[i] > self._mins[i]:
                normalized[i] = (features[i] - self._mins[i]) / (self._maxs[i] - self._mins[i])
            else:
                normalized[i] = 0.5
        
        return normalized
    
    def _normalize_standard(self, features: np.ndarray) -> np.ndarray:
        """Standardization (z-score): transform to mean 0, std 1."""
        self._count += 1
        
        # Update mean and variance
        for i in range(min(len(features), len(self._means))):
            delta = features[i] - self._means[i]
            self._means[i] = self._means[i] + delta / self._count
            delta2 = features[i] - self._means[i]
            self._stds[i] = np.sqrt((self._stds[i]**2 * (self._count-1) + delta * delta2) / max(1, self._count))
        
        # Standardize
        normalized = np.zeros_like(features)
        for i in range(min(len(features), len(self._means))):
            if self._stds[i] > 0:
                normalized[i] = (features[i] - self._means[i]) / self._stds[i]
                # Clip to reasonable range [-3,3] and map to [0,1]
                normalized[i] = np.clip(normalized[i], -3, 3) / 3
                normalized[i] = (normalized[i] + 1) / 2
            else:
                normalized[i] = 0.5
        
        return normalized


# ============================================================
# Composite Pattern Space
# ============================================================

class CompositePatternSpace(PatternSpace):
    """
    Composite Pattern Space: Concatenates features from multiple pattern spaces.
    
    Combines multiple feature extractors into one, suitable for multimodal input.
    For example: simultaneously processing text and numerical features, then concatenating them.
    
    Characteristics:
        - Multimodal: Supports combining different types of features
        - Flexible: Can assign names to each subspace
        - Robust: Zero-padding when individual subspaces fail
    
    Example:
        >>> # Create multiple pattern spaces
        >>> text_pat = SimplePatternSpace(
        ...     lambda x: [len(x)],
        ...     feature_names=['length']
        ... )
        >>> 
        >>> num_pat = StatisticalPatternSpace(
        ...     features=['mean', 'std'],
        ...     name="numbers"
        >>> )
        >>> 
        >>> # Combine them
        >>> composite = CompositePatternSpace([
        ...     ('text', text_pat),
        ...     ('stats', num_pat)
        ... ])
        >>> 
        >>> # Input can be multipart data
        >>> features = composite.extract("hello")
        >>> print(len(features))  # 3 (1 text feature + 2 statistical features)
    """
    
    def __init__(self, patterns: List[Union[PatternSpace, Tuple[str, PatternSpace]]]):
        """
        Initialize composite pattern space.
        
        Args:
            patterns: List of pattern spaces, can be:
                - PatternSpace instance: automatically generates name (pattern_0, pattern_1, etc.)
                - (name, PatternSpace) tuple: specifies custom name
        """
        super().__init__()
        
        self._patterns = []
        self._names = []
        self._dims = []
        
        for i, p in enumerate(patterns):
            if isinstance(p, tuple):
                name, pattern = p
            else:
                name = f"pattern_{i}"
                pattern = p
            
            if not isinstance(pattern, PatternSpace):
                raise TypeError(f"Item {i} must be a PatternSpace instance")
            
            self._patterns.append(pattern)
            self._names.append(name)
            self._dims.append(pattern.get_feature_dimension())
        
        self._logger = logging.getLogger("metathin.pattern.CompositePatternSpace")
    
    def extract(self, raw_input: Any) -> FeatureVector:
        """
        Extract composite features.
        
        Supports two input modes:
            - Single input: Passed to all subspaces
            - List input: Distributed to subspaces in order
        
        Args:
            raw_input: Raw input data (single value or list)
            
        Returns:
            FeatureVector: Concatenated feature vector
        """
        try:
            all_features = []
            
            # Handle multipart input
            if isinstance(raw_input, (list, tuple)) and len(raw_input) == len(self._patterns):
                # Distribute by position
                inputs = raw_input
            else:
                # All subspaces use same input
                inputs = [raw_input] * len(self._patterns)
            
            for i, (pattern, input_data) in enumerate(zip(self._patterns, inputs)):
                try:
                    features = pattern.extract(input_data)
                    
                    # Ensure 1-dimensional
                    if features.ndim > 1:
                        features = features.flatten()
                    
                    all_features.append(features)
                    
                except Exception as e:
                    self._logger.error(f"Pattern '{self._names[i]}' extraction failed: {e}")
                    # Zero-pad
                    all_features.append(np.zeros(self._dims[i] if self._dims[i] > 0 else 1))
            
            # Concatenate all features
            result = np.concatenate(all_features)
            
            return result
            
        except Exception as e:
            self._logger.error(f"Composite feature extraction failed: {e}")
            raise PatternExtractionError(f"Composite feature extraction failed: {e}") from e
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names.
        
        Returns feature names with prefix in format "subspace_name.feature_name".
        """
        names = []
        for name, pattern in zip(self._names, self._patterns):
            sub_names = pattern.get_feature_names()
            if sub_names:
                names.extend([f"{name}.{sub}" for sub in sub_names])
            else:
                dim = pattern.get_feature_dimension()
                names.extend([f"{name}_{i}" for i in range(dim)])
        
        return names


# ============================================================
# Cached Pattern Space
# ============================================================

class CachedPatternSpace(PatternSpace):
    """
    Cached Pattern Space: Caches extraction results from another pattern space.
    
    For computationally expensive feature extraction, caching improves efficiency.
    Supports cache size limits and LRU (Least Recently Used) eviction policy.
    
    Characteristics:
        - Performance optimization: Avoids redundant computation
        - LRU policy: Automatically evicts oldest cache entries
        - Statistical: Provides cache hit rate statistics
    
    Example:
        >>> # Create expensive pattern space
        >>> expensive = ExpensivePatternSpace()
        >>> 
        >>> # Add caching
        >>> cached = CachedPatternSpace(expensive, cache_size=100)
        >>> 
        >>> # First extraction computes, subsequent identical inputs use cache
        >>> f1 = cached.extract("same_input")  # Computes
        >>> f2 = cached.extract("same_input")  # Cache hit
    """
    
    def __init__(self, 
                 base_pattern: PatternSpace,
                 cache_size: int = 100):
        """
        Initialize cached pattern space.
        
        Args:
            base_pattern: Base pattern space to wrap
            cache_size: Cache size limit
        """
        super().__init__()
        
        self._base = base_pattern
        self._cache_size = cache_size
        
        # Cache: {input_hash: (feature_vector, timestamp)}
        self._cache: Dict[int, Tuple[FeatureVector, float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        self._logger = logging.getLogger("metathin.pattern.CachedPatternSpace")
    
    def _hash_input(self, raw_input: Any) -> int:
        """Compute input hash for cache key generation."""
        try:
            return hash(raw_input)
        except:
            # If unhashable, use string representation
            return hash(str(raw_input))
    
    def _clean_cache(self):
        """Clean oldest cache entries (LRU policy)."""
        if len(self._cache) <= self._cache_size:
            return
        
        # Sort by timestamp, delete oldest
        items = list(self._cache.items())
        items.sort(key=lambda x: x[1][1])
        
        to_remove = len(self._cache) - self._cache_size
        for key, _ in items[:to_remove]:
            del self._cache[key]
    
    def extract(self, raw_input: Any) -> FeatureVector:
        """
        Extract features with caching.
        
        Implementation steps:
            1. Compute input hash
            2. Check cache hit
            3. If hit, return cached result
            4. If miss, call base pattern space and cache result
        
        Args:
            raw_input: Raw input data
            
        Returns:
            FeatureVector: Extracted feature vector
        """
        input_hash = self._hash_input(raw_input)
        
        # Check cache
        if input_hash in self._cache:
            features, _ = self._cache[input_hash]
            self._cache_hits += 1
            self._logger.debug(f"Cache hit: {input_hash}")
            return features.copy()  # Return copy to prevent modification
        
        self._cache_misses += 1
        self._logger.debug(f"Cache miss: {input_hash}")
        
        # Compute features
        features = self._base.extract(raw_input)
        
        # Store in cache
        self._cache[input_hash] = (features.copy(), time.time())
        self._clean_cache()
        
        return features
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            'size': len(self._cache),
            'capacity': self._cache_size,
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': self._cache_hits / max(1, total),
        }