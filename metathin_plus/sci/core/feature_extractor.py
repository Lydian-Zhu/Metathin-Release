"""
Feature Extractor - Discriminative Feature Extraction from Time Series
==========================================================================

Extracts multiple features from functional data for function identification and similarity matching.
Features are designed to distinguish different function types while remaining robust to parameter
variations and noise.

Design Philosophy:
    - Completeness: Covers all aspects of functions (statistical, geometric, frequency, complexity)
    - Robustness: Insensitive to noise and parameter variations
    - Interpretable: Each feature has clear physical meaning
    - Efficient: Fast feature computation suitable for batch processing

Feature Types:
    1. Statistical Features: mean, variance, skewness, kurtosis, quantiles, etc.
    2. Geometric Features: slope, curvature, monotonicity, convexity
    3. Frequency Features: FFT dominant frequency, power spectrum entropy, frequency distribution
    4. Complexity Features: sample entropy, LZ complexity, fractal dimension
    5. Temporal Features: autocorrelation, cross-correlation, trend
"""

from turtle import pd

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from scipy import stats, signal, fftpack
from scipy.spatial.distance import pdist
import logging
from enum import Enum
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


class FeatureType(Enum):
    """Feature type enumeration."""
    STATISTICAL = "statistical"      # Statistical features
    GEOMETRIC = "geometric"          # Geometric features
    FREQUENCY = "frequency"          # Frequency domain features
    COMPLEXITY = "complexity"        # Complexity features
    TEMPORAL = "temporal"            # Temporal features


@dataclass
class FeatureDefinition:
    """
    Feature definition data class.

    Attributes:
        name: Feature name
        type: Feature type
        func: Computation function
        description: Feature description
        min_val: Theoretical minimum (for normalization)
        max_val: Theoretical maximum (for normalization)
    """
    name: str
    type: FeatureType
    func: Callable[[np.ndarray], float]
    description: str = ""
    min_val: float = -np.inf
    max_val: float = np.inf
    
    def __call__(self, data: np.ndarray) -> float:
        """Compute feature value."""
        try:
            val = self.func(data)
            return float(val)
        except Exception as e:
            return 0.0


class FeatureExtractor:
    """
    Feature Extractor - Extracts multiple features from time series.

    Main Features:
        1. Statistical features (mean, variance, skewness, etc.)
        2. Geometric features (slope, curvature, monotonicity)
        3. Frequency features (FFT dominant frequency, power spectrum entropy)
        4. Complexity features (sample entropy, fractal dimension)
        5. Feature normalization and dimensionality reduction

    Parameters:
        normalize: Whether to normalize features
        include_types: Feature types to include, None means all
        custom_features: List of custom feature definitions

    Example:
        >>> extractor = FeatureExtractor(normalize=True)
        >>> x = np.linspace(0, 10, 100)
        >>> y = np.sin(x)
        >>> features = extractor.extract(y)
        >>> print(features.shape)  # (n_features,)
    """
    
    def __init__(self,
                 normalize: bool = True,
                 include_types: Optional[List[FeatureType]] = None,
                 custom_features: Optional[List[FeatureDefinition]] = None):
        """
        Initialize feature extractor.

        Args:
            normalize: Whether to normalize features
            include_types: Feature types to include, None means all
            custom_features: List of custom feature definitions
        """
        self.normalize = normalize
        self.include_types = include_types
        
        # Logger
        self.logger = logging.getLogger("metathin_sci.core.FeatureExtractor")
        
        # Feature definitions list
        self.features: List[FeatureDefinition] = []
        self._init_default_features()
        
        # Add custom features
        if custom_features:
            self.features.extend(custom_features)
        
        # Feature statistics (for normalization)
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        
        self.logger.info(f"Feature extractor initialized with {len(self.features)} features")
    
    def _init_default_features(self):
        """Initialize default features."""
        
        # ===== 1. Statistical Features =====
        
        # Mean
        self.features.append(FeatureDefinition(
            name="mean",
            type=FeatureType.STATISTICAL,
            func=lambda x: np.mean(x),
            description="Arithmetic mean"
        ))
        
        # Standard deviation
        self.features.append(FeatureDefinition(
            name="std",
            type=FeatureType.STATISTICAL,
            func=lambda x: np.std(x),
            description="Standard deviation"
        ))
        
        # Variance
        self.features.append(FeatureDefinition(
            name="var",
            type=FeatureType.STATISTICAL,
            func=lambda x: np.var(x),
            description="Variance"
        ))
        
        # Skewness (third moment)
        self.features.append(FeatureDefinition(
            name="skewness",
            type=FeatureType.STATISTICAL,
            func=lambda x: float(stats.skew(x)) if len(x) > 3 else 0,
            description="Skewness, measures distribution asymmetry"
        ))
        
        # Kurtosis (fourth moment)
        self.features.append(FeatureDefinition(
            name="kurtosis",
            type=FeatureType.STATISTICAL,
            func=lambda x: float(stats.kurtosis(x)) if len(x) > 4 else 0,
            description="Kurtosis, measures distribution peakedness"
        ))
        
        # Median
        self.features.append(FeatureDefinition(
            name="median",
            type=FeatureType.STATISTICAL,
            func=lambda x: np.median(x),
            description="Median"
        ))
        
        # Interquartile range
        self.features.append(FeatureDefinition(
            name="iqr",
            type=FeatureType.STATISTICAL,
            func=lambda x: float(np.percentile(x, 75) - np.percentile(x, 25)),
            description="Interquartile range"
        ))
        
        # Maximum
        self.features.append(FeatureDefinition(
            name="max",
            type=FeatureType.STATISTICAL,
            func=lambda x: np.max(x),
            description="Maximum value"
        ))
        
        # Minimum
        self.features.append(FeatureDefinition(
            name="min",
            type=FeatureType.STATISTICAL,
            func=lambda x: np.min(x),
            description="Minimum value"
        ))
        
        # Range
        self.features.append(FeatureDefinition(
            name="range",
            type=FeatureType.STATISTICAL,
            func=lambda x: np.max(x) - np.min(x),
            description="Range (max - min)"
        ))
        
        # Positive value ratio
        self.features.append(FeatureDefinition(
            name="positive_ratio",
            type=FeatureType.STATISTICAL,
            func=lambda x: np.sum(x > 0) / len(x),
            description="Ratio of positive values"
        ))
        
        # Zero value ratio
        self.features.append(FeatureDefinition(
            name="zero_ratio",
            type=FeatureType.STATISTICAL,
            func=lambda x: np.sum(np.abs(x) < 1e-6) / len(x),
            description="Ratio of near-zero values"
        ))
        
        # ===== 2. Geometric Features =====
        
        # Mean slope (first difference mean)
        self.features.append(FeatureDefinition(
            name="mean_slope",
            type=FeatureType.GEOMETRIC,
            func=lambda x: np.mean(np.diff(x)) if len(x) > 1 else 0,
            description="Mean slope"
        ))
        
        # Slope standard deviation
        self.features.append(FeatureDefinition(
            name="slope_std",
            type=FeatureType.GEOMETRIC,
            func=lambda x: np.std(np.diff(x)) if len(x) > 1 else 0,
            description="Standard deviation of slopes"
        ))
        
        # Mean curvature (second difference)
        self.features.append(FeatureDefinition(
            name="mean_curvature",
            type=FeatureType.GEOMETRIC,
            func=lambda x: np.mean(np.diff(x, 2)) if len(x) > 2 else 0,
            description="Mean curvature"
        ))
        
        # Monotonicity indicator
        self.features.append(FeatureDefinition(
            name="monotonicity",
            type=FeatureType.GEOMETRIC,
            func=lambda x: np.sum(np.diff(x) > 0) / len(x) - 
                           np.sum(np.diff(x) < 0) / len(x),
            description="Monotonicity indicator (positive: increasing, negative: decreasing)"
        ))
        
        # Zero crossing rate
        self.features.append(FeatureDefinition(
            name="zero_crossing_rate",
            type=FeatureType.GEOMETRIC,
            func=self._compute_zero_crossing_rate,
            description="Rate of zero crossings"
        ))
        
        # Peak count
        self.features.append(FeatureDefinition(
            name="peak_count",
            type=FeatureType.GEOMETRIC,
            func=self._count_peaks,
            description="Number of peaks (normalized)"
        ))
        
        # Peak mean height
        self.features.append(FeatureDefinition(
            name="peak_mean_height",
            type=FeatureType.GEOMETRIC,
            func=self._compute_peak_mean_height,
            description="Mean height of peaks"
        ))
        
        # ===== 3. Frequency Features =====
        
        # Dominant frequency position
        self.features.append(FeatureDefinition(
            name="dominant_freq",
            type=FeatureType.FREQUENCY,
            func=self._compute_dominant_frequency,
            description="Dominant frequency position (normalized)"
        ))
        
        # Dominant frequency amplitude
        self.features.append(FeatureDefinition(
            name="dominant_amplitude",
            type=FeatureType.FREQUENCY,
            func=self._compute_dominant_amplitude,
            description="Dominant frequency amplitude"
        ))
        
        # Spectral centroid
        self.features.append(FeatureDefinition(
            name="spectral_centroid",
            type=FeatureType.FREQUENCY,
            func=self._compute_spectral_centroid,
            description="Spectral centroid"
        ))
        
        # Spectral bandwidth
        self.features.append(FeatureDefinition(
            name="spectral_bandwidth",
            type=FeatureType.FREQUENCY,
            func=self._compute_spectral_bandwidth,
            description="Spectral bandwidth"
        ))
        
        # Power spectrum entropy
        self.features.append(FeatureDefinition(
            name="spectral_entropy",
            type=FeatureType.FREQUENCY,
            func=self._compute_spectral_entropy,
            description="Power spectrum entropy (frequency distribution uniformity)"
        ))
        
        # Harmonic richness
        self.features.append(FeatureDefinition(
            name="harmonic_richness",
            type=FeatureType.FREQUENCY,
            func=self._compute_harmonic_richness,
            description="Harmonic richness (higher harmonic energy ratio)"
        ))
        
        # ===== 4. Complexity Features =====
        
        # Sample entropy (approximate entropy)
        self.features.append(FeatureDefinition(
            name="sample_entropy",
            type=FeatureType.COMPLEXITY,
            func=self._compute_sample_entropy,
            description="Sample entropy, measures sequence complexity"
        ))
        
        # Hurst exponent
        self.features.append(FeatureDefinition(
            name="hurst_exponent",
            type=FeatureType.COMPLEXITY,
            func=self._compute_hurst_exponent,
            description="Hurst exponent, measures long-range correlation"
        ))
        
        # Fractal dimension
        self.features.append(FeatureDefinition(
            name="fractal_dimension",
            type=FeatureType.COMPLEXITY,
            func=self._compute_fractal_dimension,
            description="Fractal dimension (Higuchi algorithm)"
        ))
        
        # LZ complexity
        self.features.append(FeatureDefinition(
            name="lz_complexity",
            type=FeatureType.COMPLEXITY,
            func=self._compute_lz_complexity,
            description="Lempel-Ziv complexity"
        ))
        
        # ===== 5. Temporal Features =====
        
        # Autocorrelation lag 1
        self.features.append(FeatureDefinition(
            name="autocorr_lag1",
            type=FeatureType.TEMPORAL,
            func=lambda x: float(pd.Series(x).autocorr(lag=1)) if len(x) > 10 else 0,
            description="Autocorrelation at lag 1"
        ))
        
        # Autocorrelation lag 2
        self.features.append(FeatureDefinition(
            name="autocorr_lag2",
            type=FeatureType.TEMPORAL,
            func=lambda x: float(pd.Series(x).autocorr(lag=2)) if len(x) > 10 else 0,
            description="Autocorrelation at lag 2"
        ))
        
        # Trend strength
        self.features.append(FeatureDefinition(
            name="trend_strength",
            type=FeatureType.TEMPORAL,
            func=self._compute_trend_strength,
            description="Trend strength (R² of linear fit)"
        ))
        
        # Seasonal strength
        self.features.append(FeatureDefinition(
            name="seasonal_strength",
            type=FeatureType.TEMPORAL,
            func=self._compute_seasonal_strength,
            description="Seasonal strength"
        ))
    
    # ============================================================
    # Feature Computation Helper Methods
    # ============================================================
    
    def _compute_zero_crossing_rate(self, x: np.ndarray) -> float:
        """Compute zero crossing rate."""
        if len(x) < 2:
            return 0.0
        signs = np.sign(x)
        crossings = np.sum(np.diff(signs) != 0)
        return crossings / (len(x) - 1)
    
    def _count_peaks(self, x: np.ndarray) -> float:
        """Count peaks (normalized)."""
        if len(x) < 3:
            return 0.0
        peaks = 0
        for i in range(1, len(x)-1):
            if x[i] > x[i-1] and x[i] > x[i+1]:
                peaks += 1
        return peaks / len(x)
    
    def _compute_peak_mean_height(self, x: np.ndarray) -> float:
        """Compute mean height of peaks."""
        if len(x) < 3:
            return 0.0
        peaks = []
        for i in range(1, len(x)-1):
            if x[i] > x[i-1] and x[i] > x[i+1]:
                peaks.append(x[i])
        if peaks:
            return np.mean(peaks) - np.mean(x)
        return 0.0
    
    def _compute_dominant_frequency(self, x: np.ndarray) -> float:
        """Compute dominant frequency position (normalized)."""
        if len(x) < 4:
            return 0.0
        # Detrend
        x_detrend = x - np.mean(x)
        # FFT
        n = len(x_detrend)
        fft_vals = np.fft.fft(x_detrend)
        fft_mag = np.abs(fft_vals[:n//2])
        if len(fft_mag) == 0:
            return 0.0
        # Find dominant frequency
        dominant_idx = np.argmax(fft_mag[1:]) + 1  # Skip DC
        return dominant_idx / (n//2)  # Normalize
    
    def _compute_dominant_amplitude(self, x: np.ndarray) -> float:
        """Compute dominant frequency amplitude (normalized)."""
        if len(x) < 4:
            return 0.0
        x_detrend = x - np.mean(x)
        n = len(x_detrend)
        fft_vals = np.fft.fft(x_detrend)
        fft_mag = np.abs(fft_vals[:n//2])
        if len(fft_mag) == 0:
            return 0.0
        max_amp = np.max(fft_mag[1:])  # Skip DC
        return max_amp / (np.sum(fft_mag[1:]) + 1e-8)
    
    def _compute_spectral_centroid(self, x: np.ndarray) -> float:
        """Compute spectral centroid."""
        if len(x) < 4:
            return 0.0
        x_detrend = x - np.mean(x)
        n = len(x_detrend)
        fft_vals = np.fft.fft(x_detrend)
        fft_mag = np.abs(fft_vals[:n//2])
        freqs = np.arange(len(fft_mag))
        if np.sum(fft_mag) == 0:
            return 0.0
        centroid = np.sum(freqs * fft_mag) / np.sum(fft_mag)
        return centroid / (n//2)  # Normalize
    
    def _compute_spectral_bandwidth(self, x: np.ndarray) -> float:
        """Compute spectral bandwidth."""
        if len(x) < 4:
            return 0.0
        x_detrend = x - np.mean(x)
        n = len(x_detrend)
        fft_vals = np.fft.fft(x_detrend)
        fft_mag = np.abs(fft_vals[:n//2])
        freqs = np.arange(len(fft_mag))
        if np.sum(fft_mag) == 0:
            return 0.0
        centroid = self._compute_spectral_centroid(x)
        bandwidth = np.sqrt(np.sum(((freqs/(n//2) - centroid)**2) * fft_mag) / np.sum(fft_mag))
        return bandwidth
    
    def _compute_spectral_entropy(self, x: np.ndarray) -> float:
        """Compute power spectrum entropy."""
        if len(x) < 4:
            return 0.0
        x_detrend = x - np.mean(x)
        n = len(x_detrend)
        fft_vals = np.fft.fft(x_detrend)
        power = np.abs(fft_vals[:n//2])**2
        power = power / (np.sum(power) + 1e-10)
        entropy = -np.sum(power * np.log(power + 1e-10))
        # Normalize to [0,1]
        max_entropy = np.log(n//2)
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def _compute_harmonic_richness(self, x: np.ndarray) -> float:
        """Compute harmonic richness."""
        if len(x) < 4:
            return 0.0
        x_detrend = x - np.mean(x)
        n = len(x_detrend)
        fft_vals = np.fft.fft(x_detrend)
        fft_mag = np.abs(fft_vals[:n//2])
        if len(fft_mag) < 10:
            return 0.0
        
        # Find dominant frequency
        dominant_idx = np.argmax(fft_mag[1:]) + 1
        
        # Calculate harmonic energy
        harmonic_indices = [dominant_idx * k for k in range(2, 5) 
                           if dominant_idx * k < len(fft_mag)]
        if not harmonic_indices:
            return 0.0
        
        harmonic_energy = np.sum([fft_mag[i] for i in harmonic_indices])
        total_energy = np.sum(fft_mag) - fft_mag[0]  # Remove DC
        
        return harmonic_energy / (total_energy + 1e-8)
    
    def _compute_sample_entropy(self, x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Compute sample entropy."""
        if len(x) < m + 10:
            return 0.0
        
        N = len(x)
        # Normalize
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        
        # Build templates
        def _maxdist(xi, xj):
            return np.max(np.abs(xi - xj))
        
        def _phi(m):
            templates = np.array([x[i:i+m] for i in range(N-m+1)])
            B = 0
            for i in range(len(templates)):
                for j in range(len(templates)):
                    if i != j and _maxdist(templates[i], templates[j]) <= r:
                        B += 1
            return B / (len(templates) * (len(templates)-1))
        
        try:
            phi_m = _phi(m)
            phi_m1 = _phi(m+1)
            if phi_m == 0 or phi_m1 == 0:
                return 0.0
            return -np.log(phi_m1 / phi_m)
        except:
            return 0.0
    
    def _compute_hurst_exponent(self, x: np.ndarray) -> float:
        """Compute Hurst exponent using R/S analysis."""
        if len(x) < 100:
            return 0.5
        
        x = x - np.mean(x)
        n = len(x)
        
        # Calculate R/S at different scales
        scales = [int(n/4), int(n/2), int(3*n/4), n]
        scales = [s for s in scales if s > 10]
        
        if len(scales) < 2:
            return 0.5
        
        rs_values = []
        for scale in scales:
            # Segment
            n_segments = n // scale
            rs_segment = []
            
            for i in range(n_segments):
                segment = x[i*scale:(i+1)*scale]
                # Cumulative deviation
                cumsum = np.cumsum(segment - np.mean(segment))
                R = np.max(cumsum) - np.min(cumsum)
                S = np.std(segment)
                if S > 0:
                    rs_segment.append(R / S)
            
            if rs_segment:
                rs_values.append(np.mean(rs_segment))
            else:
                rs_values.append(0)
        
        # Fit log-log
        if len(rs_values) >= 2:
            log_scales = np.log(scales)
            log_rs = np.log(rs_values)
            coeffs = np.polyfit(log_scales, log_rs, 1)
            return coeffs[0]
        
        return 0.5
    
    def _compute_fractal_dimension(self, x: np.ndarray) -> float:
        """Compute fractal dimension using Higuchi algorithm."""
        if len(x) < 50:
            return 1.0
        
        n = len(x)
        k_max = min(10, n//4)
        
        lengths = []
        for k in range(1, k_max+1):
            length = 0
            for m in range(k):
                indices = np.arange(m, n-1, k)
                if len(indices) > 1:
                    seg = x[indices]
                    length += np.sum(np.abs(np.diff(seg))) * (n-1) / (len(indices) * k)
            if k > 0:
                lengths.append(length / k)
        
        if len(lengths) > 1:
            log_k = np.log(1.0 / np.arange(1, len(lengths)+1))
            log_l = np.log(lengths)
            coeffs = np.polyfit(log_k, log_l, 1)
            return -coeffs[0]
        
        return 1.0
    
    def _compute_lz_complexity(self, x: np.ndarray, threshold: str = 'median') -> float:
        """Compute Lempel-Ziv complexity."""
        if len(x) < 10:
            return 0.0
        
        # Binarize
        if threshold == 'median':
            thresh = np.median(x)
        elif threshold == 'mean':
            thresh = np.mean(x)
        else:
            thresh = 0.0
        
        binary = (x > thresh).astype(int)
        s = ''.join(map(str, binary))
        
        # LZ complexity algorithm
        c = 1
        l = 1
        i = 0
        k = 1
        n = len(s)
        
        while True:
            if s[i+k] not in s[i:i+k]:
                c += 1
                i += k
                k = 1
            else:
                k += 1
            
            if i + k >= n:
                break
        
        # Normalize
        norm = n / np.log2(n) if n > 1 else 1
        return c / norm
    
    def _compute_trend_strength(self, x: np.ndarray) -> float:
        """Compute trend strength."""
        if len(x) < 3:
            return 0.0
        
        t = np.arange(len(x))
        coeffs = np.polyfit(t, x, 1)
        trend = coeffs[0] * t + coeffs[1]
        residual = x - trend
        
        # R²
        ss_tot = np.sum((x - np.mean(x))**2)
        ss_res = np.sum(residual**2)
        
        if ss_tot == 0:
            return 0.0
        
        r2 = 1 - ss_res / ss_tot
        return r2
    
    def _compute_seasonal_strength(self, x: np.ndarray) -> float:
        """Compute seasonal strength (assuming period is 1/4 of data length)."""
        if len(x) < 20:
            return 0.0
        
        # Simple detrending
        t = np.arange(len(x))
        coeffs = np.polyfit(t, x, 1)
        detrended = x - (coeffs[0] * t + coeffs[1])
        
        # Estimate period
        n = len(detrended)
        period = n // 4  # Assume period is 1/4 of data length
        if period < 2:
            return 0.0
        
        # Compute seasonal component
        seasonal = np.zeros(n)
        for i in range(n):
            seasonal[i] = np.mean(detrended[i % period::period])
        
        residual = detrended - seasonal
        
        # Compute strength
        var_total = np.var(detrended)
        var_residual = np.var(residual)
        
        if var_total == 0:
            return 0.0
        
        strength = 1 - var_residual / var_total
        return strength
    
    # ============================================================
    # Main Interface
    # ============================================================
    
    def extract(self, data: np.ndarray) -> np.ndarray:
        """
        Extract features from a single sequence.

        Args:
            data: Input time series (n_points,)

        Returns:
            np.ndarray: Feature vector
        """
        data = np.asarray(data).flatten()
        
        if len(data) == 0:
            return np.zeros(len(self.features))
        
        features = []
        for feat_def in self.features:
            # Filter by type
            if (self.include_types and 
                feat_def.type not in self.include_types):
                features.append(0.0)
                continue
            
            # Compute feature
            val = feat_def(data)
            features.append(val)
        
        return np.array(features, dtype=np.float64)
    
    def extract_batch(self, data_list: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from multiple sequences.

        Args:
            data_list: List of data arrays, each (n_points,)

        Returns:
            np.ndarray: Feature matrix (n_samples, n_features)
        """
        features = []
        for data in data_list:
            feat = self.extract(data)
            features.append(feat)
        
        feature_matrix = np.array(features)
        
        # Normalize
        if self.normalize:
            feature_matrix = self.normalize_features(feature_matrix)
        
        return feature_matrix
    
    def normalize_features(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Normalize feature matrix.

        Args:
            feature_matrix: Raw feature matrix

        Returns:
            np.ndarray: Normalized feature matrix
        """
        normalized = np.zeros_like(feature_matrix)
        
        for j in range(feature_matrix.shape[1]):
            col = feature_matrix[:, j]
            
            # Skip constant columns
            if np.std(col) < 1e-10:
                normalized[:, j] = 0.5
                continue
            
            # Z-score normalization
            mean = np.mean(col)
            std = np.std(col)
            normalized[:, j] = (col - mean) / (std + 1e-8)
            
            # Record statistics
            self.feature_stats[self.features[j].name] = {
                'mean': float(mean),
                'std': float(std)
            }
        
        return normalized
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return [f.name for f in self.features]
    
    def get_feature_types(self) -> List[str]:
        """Get feature types."""
        return [f.type.value for f in self.features]
    
    def get_feature_descriptions(self) -> List[str]:
        """Get feature descriptions."""
        return [f.description for f in self.features]
    
    def get_feature_count(self) -> int:
        """Get number of features."""
        return len(self.features)
    
    def add_custom_feature(self, feature: FeatureDefinition) -> None:
        """
        Add custom feature.

        Args:
            feature: Feature definition
        """
        self.features.append(feature)
        self.logger.info(f"Added custom feature: {feature.name}")
    
    def remove_feature(self, name: str) -> bool:
        """
        Remove feature by name.

        Args:
            name: Feature name

        Returns:
            bool: Success status
        """
        for i, f in enumerate(self.features):
            if f.name == name:
                self.features.pop(i)
                self.logger.info(f"Removed feature: {name}")
                return True
        return False
    
    def save(self, filename: str) -> bool:
        """
        Save extractor configuration.

        Args:
            filename: Output filename

        Returns:
            bool: Success status
        """
        try:
            config = {
                'normalize': self.normalize,
                'include_types': [t.value for t in self.include_types] if self.include_types else None,
                'feature_names': self.get_feature_names(),
                'feature_stats': self.feature_stats
            }
            import json
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Save failed: {e}")
            return False
    
    def load(self, filename: str) -> bool:
        """
        Load extractor configuration.

        Args:
            filename: Input filename

        Returns:
            bool: Success status
        """
        try:
            import json
            with open(filename, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.normalize = config.get('normalize', True)
            if config.get('include_types'):
                self.include_types = [FeatureType(t) for t in config['include_types']]
            self.feature_stats = config.get('feature_stats', {})
            
            return True
        except Exception as e:
            self.logger.error(f"Load failed: {e}")
            return False


# ============================================================
# Usage Examples
# ============================================================
"""
>>> from metathin_plus.sci.core import FeatureExtractor
>>> import numpy as np
>>> 
>>> # 1. Create feature extractor
>>> extractor = FeatureExtractor(normalize=True)
>>> print(f"Number of features: {extractor.get_feature_count()}")
>>> 
>>> # 2. Generate test data
>>> x = np.linspace(0, 10, 100)
>>> y1 = np.sin(x)                     # Sine function
>>> y2 = 2*x + 1                        # Linear function
>>> y3 = np.exp(-0.5*x) * np.sin(2*x)   # Damped oscillation
>>> 
>>> # 3. Extract features
>>> f1 = extractor.extract(y1)
>>> f2 = extractor.extract(y2)
>>> f3 = extractor.extract(y3)
>>> 
>>> # 4. View feature differences
>>> names = extractor.get_feature_names()
>>> for i, name in enumerate(names[:5]):  # Only first 5
...     print(f"{name:15s}: {f1[i]:.3f} | {f2[i]:.3f} | {f3[i]:.3f}")
>>> 
>>> # 5. Batch extraction
>>> features = extractor.extract_batch([y1, y2, y3])
>>> print(features.shape)  # (3, n_features)
"""