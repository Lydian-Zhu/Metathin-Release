
# metathin_sci/core/feature_extractor.py
"""
Feature Extractor - Extract Discriminative Features from Time Series | 特征提取器 - 从时间序列中提取判别特征
=============================================================================================================

Extracts various features from function data for function identification and similarity matching.
Features are designed to distinguish different function types while being robust to parameter
variations and noise.

从函数数据中提取多种特征，用于函数识别和相似度匹配。
特征设计要能区分不同类型的函数，同时对参数变化和噪声鲁棒。

Design Philosophy | 设计理念:
    - Completeness: Cover all aspects (statistical, geometric, frequency, complexity)
      完备性：覆盖函数的各个方面（统计、几何、频域、复杂度）
    - Robustness: Insensitive to noise and parameter changes | 鲁棒性：对噪声和参数变化不敏感
    - Interpretability: Each feature has clear physical meaning | 可解释：每个特征都有明确的物理意义
    - Efficiency: Fast feature computation for batch processing | 高效：特征计算速度快，适合批量处理

Feature Types | 特征类型:
    1. Statistical: mean, variance, skewness, kurtosis, quantiles | 统计特征
    2. Geometric: slope, curvature, monotonicity, zero-crossing | 几何特征
    3. Frequency: FFT dominant frequency, spectral centroid, spectral entropy | 频域特征
    4. Complexity: sample entropy, Hurst exponent, fractal dimension | 复杂度特征
    5. Temporal: autocorrelation, trend strength, seasonality | 时序特征
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Union
from scipy import stats, signal
import logging
from enum import Enum
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Feature Type Enum | 特征类型枚举
# ============================================================

class FeatureType(Enum):
    """Feature type enumeration | 特征类型枚举"""
    STATISTICAL = "statistical"      # Statistical features | 统计特征
    GEOMETRIC = "geometric"          # Geometric features | 几何特征
    FREQUENCY = "frequency"          # Frequency domain features | 频域特征
    COMPLEXITY = "complexity"        # Complexity features | 复杂度特征
    TEMPORAL = "temporal"            # Temporal features | 时序特征


# ============================================================
# Feature Definition Data Class | 特征定义数据类
# ============================================================

@dataclass
class FeatureDefinition:
    """
    Feature definition data class | 特征定义数据类
    
    Defines a single feature with its name, type, and computation function.
    
    定义单个特征，包含名称、类型和计算函数。
    
    Attributes | 属性:
        name: Feature name | 特征名称
        type: Feature type category | 特征类型
        func: Computation function | 计算函数
        description: Feature description | 特征描述
        min_val: Theoretical minimum (for normalization) | 理论最小值（用于归一化）
        max_val: Theoretical maximum (for normalization) | 理论最大值（用于归一化）
    """
    name: str
    type: FeatureType
    func: Callable[[np.ndarray], float]
    description: str = ""
    min_val: float = -np.inf
    max_val: float = np.inf
    
    def __call__(self, data: np.ndarray) -> float:
        """Compute feature value | 计算特征值"""
        try:
            val = self.func(data)
            if np.isnan(val) or np.isinf(val):
                return 0.0
            return float(val)
        except Exception as e:
            return 0.0


# ============================================================
# Feature Extractor | 特征提取器
# ============================================================

class FeatureExtractor:
    """
    Feature Extractor - Extracts various features from time series | 特征提取器 - 从时间序列中提取多种特征
    
    Main features | 主要功能:
        1. Extract statistical features (mean, variance, skewness, etc.) | 提取统计特征
        2. Extract geometric features (slope, curvature, monotonicity) | 提取几何特征
        3. Extract frequency features (FFT dominant frequency, spectral entropy) | 提取频域特征
        4. Extract complexity features (sample entropy, fractal dimension) | 提取复杂度特征
        5. Feature normalization and dimensionality reduction | 特征归一化和降维
    
    Parameters | 参数:
        normalize: Whether to normalize features | 是否归一化特征
        include_types: Feature types to include (None for all) | 包含的特征类型，None表示全部
        custom_features: Custom feature list | 自定义特征列表
    
    Example | 示例:
        >>> extractor = FeatureExtractor(normalize=True)
        >>> 
        >>> # Extract features from single time series | 从单个时间序列提取特征
        >>> features = extractor.extract(y_data)
        >>> 
        >>> # Extract features from multiple series | 从多个序列批量提取特征
        >>> features_batch = extractor.extract_batch([y1, y2, y3])
        >>> 
        >>> # Get feature names | 获取特征名称
        >>> names = extractor.get_feature_names()
    """
    
    def __init__(self,
                 normalize: bool = True,
                 include_types: Optional[List[FeatureType]] = None,
                 custom_features: Optional[List[FeatureDefinition]] = None):
        """
        Initialize feature extractor | 初始化特征提取器
        
        Args | 参数:
            normalize: Whether to normalize features | 是否归一化特征
            include_types: Feature types to include | 包含的特征类型
            custom_features: Custom feature definitions | 自定义特征列表
        """
        self.normalize = normalize
        self.include_types = include_types
        
        # Logger | 日志
        self.logger = logging.getLogger("metathin_sci.core.FeatureExtractor")
        
        # Feature definitions | 特征定义列表
        self.features: List[FeatureDefinition] = []
        self._init_default_features()
        
        # Add custom features | 添加自定义特征
        if custom_features:
            self.features.extend(custom_features)
        
        # Feature statistics for normalization | 特征统计信息（用于归一化）
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        
        self.logger.info(f"FeatureExtractor initialized: {len(self.features)} features | 初始化完成")
    
    def _init_default_features(self):
        """Initialize default features | 初始化默认特征"""
        
        # ===== 1. Statistical Features | 统计特征 =====
        
        # Mean | 均值
        self.features.append(FeatureDefinition(
            name="mean",
            type=FeatureType.STATISTICAL,
            func=lambda x: np.mean(x),
            description="Arithmetic mean | 算术平均值"
        ))
        
        # Standard deviation | 标准差
        self.features.append(FeatureDefinition(
            name="std",
            type=FeatureType.STATISTICAL,
            func=lambda x: np.std(x),
            description="Standard deviation | 标准差"
        ))
        
        # Variance | 方差
        self.features.append(FeatureDefinition(
            name="var",
            type=FeatureType.STATISTICAL,
            func=lambda x: np.var(x),
            description="Variance | 方差"
        ))
        
        # Skewness (third moment) | 偏度（三阶矩）
        self.features.append(FeatureDefinition(
            name="skewness",
            type=FeatureType.STATISTICAL,
            func=lambda x: float(stats.skew(x)) if len(x) > 3 else 0,
            description="Skewness, measures distribution asymmetry | 偏度，衡量分布不对称性"
        ))
        
        # Kurtosis (fourth moment) | 峰度（四阶矩）
        self.features.append(FeatureDefinition(
            name="kurtosis",
            type=FeatureType.STATISTICAL,
            func=lambda x: float(stats.kurtosis(x)) if len(x) > 4 else 0,
            description="Kurtosis, measures distribution peakedness | 峰度，衡量分布尖峭程度"
        ))
        
        # Median | 中位数
        self.features.append(FeatureDefinition(
            name="median",
            type=FeatureType.STATISTICAL,
            func=lambda x: np.median(x),
            description="Median | 中位数"
        ))
        
        # Interquartile range | 四分位距
        self.features.append(FeatureDefinition(
            name="iqr",
            type=FeatureType.STATISTICAL,
            func=lambda x: float(np.percentile(x, 75) - np.percentile(x, 25)),
            description="Interquartile range (Q3 - Q1) | 四分位距"
        ))
        
        # Maximum | 最大值
        self.features.append(FeatureDefinition(
            name="max",
            type=FeatureType.STATISTICAL,
            func=lambda x: np.max(x),
            description="Maximum value | 最大值"
        ))
        
        # Minimum | 最小值
        self.features.append(FeatureDefinition(
            name="min",
            type=FeatureType.STATISTICAL,
            func=lambda x: np.min(x),
            description="Minimum value | 最小值"
        ))
        
        # Range | 极差
        self.features.append(FeatureDefinition(
            name="range",
            type=FeatureType.STATISTICAL,
            func=lambda x: np.max(x) - np.min(x),
            description="Range (max - min) | 极差"
        ))
        
        # Positive ratio | 正值比例
        self.features.append(FeatureDefinition(
            name="positive_ratio",
            type=FeatureType.STATISTICAL,
            func=lambda x: np.sum(x > 0) / len(x),
            description="Proportion of positive values | 正值比例"
        ))
        
        # Zero ratio | 零值比例
        self.features.append(FeatureDefinition(
            name="zero_ratio",
            type=FeatureType.STATISTICAL,
            func=lambda x: np.sum(np.abs(x) < 1e-6) / len(x),
            description="Proportion of near-zero values | 接近零的比例"
        ))
        
        # ===== 2. Geometric Features | 几何特征 =====
        
        # Mean slope (first difference mean) | 平均斜率（一阶差分均值）
        self.features.append(FeatureDefinition(
            name="mean_slope",
            type=FeatureType.GEOMETRIC,
            func=lambda x: np.mean(np.diff(x)) if len(x) > 1 else 0,
            description="Mean slope | 平均斜率"
        ))
        
        # Slope standard deviation | 斜率标准差
        self.features.append(FeatureDefinition(
            name="slope_std",
            type=FeatureType.GEOMETRIC,
            func=lambda x: np.std(np.diff(x)) if len(x) > 1 else 0,
            description="Slope standard deviation | 斜率标准差"
        ))
        
        # Mean curvature (second difference) | 平均曲率（二阶差分）
        self.features.append(FeatureDefinition(
            name="mean_curvature",
            type=FeatureType.GEOMETRIC,
            func=lambda x: np.mean(np.diff(x, 2)) if len(x) > 2 else 0,
            description="Mean curvature | 平均曲率"
        ))
        
        # Monotonicity | 单调性指标
        self.features.append(FeatureDefinition(
            name="monotonicity",
            type=FeatureType.GEOMETRIC,
            func=self._compute_monotonicity,
            description="Monotonicity (positive: increasing, negative: decreasing) | 单调性指标"
        ))
        
        # Zero crossing rate | 过零点率
        self.features.append(FeatureDefinition(
            name="zero_crossing_rate",
            type=FeatureType.GEOMETRIC,
            func=self._compute_zero_crossing_rate,
            description="Zero crossing rate | 过零点率"
        ))
        
        # Peak count (normalized) | 峰值数（归一化）
        self.features.append(FeatureDefinition(
            name="peak_count",
            type=FeatureType.GEOMETRIC,
            func=self._count_peaks,
            description="Peak count (normalized) | 峰值数量（归一化）"
        ))
        
        # Peak mean height | 峰值平均高度
        self.features.append(FeatureDefinition(
            name="peak_mean_height",
            type=FeatureType.GEOMETRIC,
            func=self._compute_peak_mean_height,
            description="Peak mean height | 峰值平均高度"
        ))
        
        # ===== 3. Frequency Domain Features | 频域特征 =====
        
        # Dominant frequency position | 主频位置
        self.features.append(FeatureDefinition(
            name="dominant_freq",
            type=FeatureType.FREQUENCY,
            func=self._compute_dominant_frequency,
            description="Dominant frequency position (normalized) | 主频位置（归一化）"
        ))
        
        # Dominant frequency amplitude | 主频幅值
        self.features.append(FeatureDefinition(
            name="dominant_amplitude",
            type=FeatureType.FREQUENCY,
            func=self._compute_dominant_amplitude,
            description="Dominant frequency amplitude | 主频幅值"
        ))
        
        # Spectral centroid | 频谱重心
        self.features.append(FeatureDefinition(
            name="spectral_centroid",
            type=FeatureType.FREQUENCY,
            func=self._compute_spectral_centroid,
            description="Spectral centroid | 频谱重心"
        ))
        
        # Spectral bandwidth | 频谱带宽
        self.features.append(FeatureDefinition(
            name="spectral_bandwidth",
            type=FeatureType.FREQUENCY,
            func=self._compute_spectral_bandwidth,
            description="Spectral bandwidth | 频谱带宽"
        ))
        
        # Spectral entropy | 功率谱熵
        self.features.append(FeatureDefinition(
            name="spectral_entropy",
            type=FeatureType.FREQUENCY,
            func=self._compute_spectral_entropy,
            description="Spectral entropy (frequency distribution uniformity) | 功率谱熵"
        ))
        
        # Harmonic richness | 谐波丰富度
        self.features.append(FeatureDefinition(
            name="harmonic_richness",
            type=FeatureType.FREQUENCY,
            func=self._compute_harmonic_richness,
            description="Harmonic richness (higher harmonic energy ratio) | 谐波丰富度"
        ))
        
        # ===== 4. Complexity Features | 复杂度特征 =====
        
        # Sample entropy | 样本熵
        self.features.append(FeatureDefinition(
            name="sample_entropy",
            type=FeatureType.COMPLEXITY,
            func=self._compute_sample_entropy,
            description="Sample entropy, measures sequence complexity | 样本熵"
        ))
        
        # Hurst exponent | Hurst指数
        self.features.append(FeatureDefinition(
            name="hurst_exponent",
            type=FeatureType.COMPLEXITY,
            func=self._compute_hurst_exponent,
            description="Hurst exponent, measures long-range correlation | Hurst指数"
        ))
        
        # Fractal dimension | 分形维数
        self.features.append(FeatureDefinition(
            name="fractal_dimension",
            type=FeatureType.COMPLEXITY,
            func=self._compute_fractal_dimension,
            description="Fractal dimension (Higuchi algorithm) | 分形维数"
        ))
        
        # Lempel-Ziv complexity | LZ复杂度
        self.features.append(FeatureDefinition(
            name="lz_complexity",
            type=FeatureType.COMPLEXITY,
            func=self._compute_lz_complexity,
            description="Lempel-Ziv complexity | Lempel-Ziv复杂度"
        ))
        
        # ===== 5. Temporal Features | 时序特征 =====
        
        # Autocorrelation lag 1 | 自相关系数（滞后1）
        self.features.append(FeatureDefinition(
            name="autocorr_lag1",
            type=FeatureType.TEMPORAL,
            func=self._compute_autocorr_lag1,
            description="Autocorrelation at lag 1 | 滞后1自相关系数"
        ))
        
        # Autocorrelation lag 2 | 自相关系数（滞后2）
        self.features.append(FeatureDefinition(
            name="autocorr_lag2",
            type=FeatureType.TEMPORAL,
            func=self._compute_autocorr_lag2,
            description="Autocorrelation at lag 2 | 滞后2自相关系数"
        ))
        
        # Trend strength | 趋势强度
        self.features.append(FeatureDefinition(
            name="trend_strength",
            type=FeatureType.TEMPORAL,
            func=self._compute_trend_strength,
            description="Trend strength (R² of linear fit) | 趋势强度（线性拟合的R²）"
        ))
        
        # Seasonal strength | 季节性强度
        self.features.append(FeatureDefinition(
            name="seasonal_strength",
            type=FeatureType.TEMPORAL,
            func=self._compute_seasonal_strength,
            description="Seasonal strength | 季节性强度"
        ))
    
    # ============================================================
    # Feature Computation Helpers | 特征计算辅助方法
    # ============================================================
    
    def _compute_monotonicity(self, x: np.ndarray) -> float:
        """Compute monotonicity | 计算单调性指标"""
        if len(x) < 2:
            return 0.0
        diffs = np.diff(x)
        pos_ratio = np.sum(diffs > 0) / len(diffs)
        neg_ratio = np.sum(diffs < 0) / len(diffs)
        return pos_ratio - neg_ratio
    
    def _compute_zero_crossing_rate(self, x: np.ndarray) -> float:
        """Compute zero crossing rate | 计算过零点率"""
        if len(x) < 2:
            return 0.0
        signs = np.sign(x)
        crossings = np.sum(np.diff(signs) != 0)
        return crossings / (len(x) - 1)
    
    def _count_peaks(self, x: np.ndarray) -> float:
        """Count peaks (normalized) | 计数峰值（归一化）"""
        if len(x) < 3:
            return 0.0
        peaks = 0
        for i in range(1, len(x)-1):
            if x[i] > x[i-1] and x[i] > x[i+1]:
                peaks += 1
        return peaks / len(x)
    
    def _compute_peak_mean_height(self, x: np.ndarray) -> float:
        """Compute peak mean height | 计算峰值平均高度"""
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
        """Compute dominant frequency position (normalized) | 计算主频位置（归一化）"""
        if len(x) < 4:
            return 0.0
        x_detrend = x - np.mean(x)
        n = len(x_detrend)
        fft_vals = np.fft.fft(x_detrend)
        fft_mag = np.abs(fft_vals[:n//2])
        if len(fft_mag) == 0:
            return 0.0
        # Skip DC component | 跳过直流
        dominant_idx = np.argmax(fft_mag[1:]) + 1
        return dominant_idx / (n//2)
    
    def _compute_dominant_amplitude(self, x: np.ndarray) -> float:
        """Compute dominant frequency amplitude (normalized) | 计算主频幅值（归一化）"""
        if len(x) < 4:
            return 0.0
        x_detrend = x - np.mean(x)
        n = len(x_detrend)
        fft_vals = np.fft.fft(x_detrend)
        fft_mag = np.abs(fft_vals[:n//2])
        if len(fft_mag) == 0:
            return 0.0
        max_amp = np.max(fft_mag[1:])  # Skip DC | 跳过直流
        return max_amp / (np.sum(fft_mag[1:]) + 1e-8)
    
    def _compute_spectral_centroid(self, x: np.ndarray) -> float:
        """Compute spectral centroid | 计算频谱重心"""
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
        return centroid / (n//2)
    
    def _compute_spectral_bandwidth(self, x: np.ndarray) -> float:
        """Compute spectral bandwidth | 计算频谱带宽"""
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
        centroid_norm = centroid * (n//2)
        bandwidth = np.sqrt(np.sum(((freqs - centroid_norm) ** 2) * fft_mag) / np.sum(fft_mag))
        return bandwidth / (n//2)
    
    def _compute_spectral_entropy(self, x: np.ndarray) -> float:
        """Compute spectral entropy | 计算功率谱熵"""
        if len(x) < 4:
            return 0.0
        x_detrend = x - np.mean(x)
        n = len(x_detrend)
        fft_vals = np.fft.fft(x_detrend)
        power = np.abs(fft_vals[:n//2]) ** 2
        power = power / (np.sum(power) + 1e-10)
        entropy = -np.sum(power * np.log(power + 1e-10))
        max_entropy = np.log(n//2)
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def _compute_harmonic_richness(self, x: np.ndarray) -> float:
        """Compute harmonic richness | 计算谐波丰富度"""
        if len(x) < 10:
            return 0.0
        x_detrend = x - np.mean(x)
        n = len(x_detrend)
        fft_vals = np.fft.fft(x_detrend)
        fft_mag = np.abs(fft_vals[:n//2])
        if len(fft_mag) < 10:
            return 0.0
        
        # Find dominant frequency | 找到主频
        dominant_idx = np.argmax(fft_mag[1:]) + 1
        
        # Calculate harmonic energy | 计算谐波能量
        harmonic_indices = [dominant_idx * k for k in range(2, 5) 
                           if dominant_idx * k < len(fft_mag)]
        if not harmonic_indices:
            return 0.0
        
        harmonic_energy = np.sum([fft_mag[i] for i in harmonic_indices])
        total_energy = np.sum(fft_mag) - fft_mag[0]  # Remove DC | 去掉直流
        
        return harmonic_energy / (total_energy + 1e-8)
    
    def _compute_sample_entropy(self, x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Compute sample entropy | 计算样本熵"""
        if len(x) < m + 10:
            return 0.0
        
        N = len(x)
        # Normalize | 归一化
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        
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
        """Compute Hurst exponent (R/S analysis) | 计算Hurst指数（R/S分析）"""
        if len(x) < 100:
            return 0.5
        
        x = x - np.mean(x)
        n = len(x)
        
        # Calculate R/S at different scales | 计算不同尺度的R/S
        scales = [int(n/4), int(n/2), int(3*n/4), n]
        scales = [s for s in scales if s > 10]
        
        if len(scales) < 2:
            return 0.5
        
        rs_values = []
        for scale in scales:
            n_segments = n // scale
            rs_segment = []
            
            for i in range(n_segments):
                segment = x[i*scale:(i+1)*scale]
                cumsum = np.cumsum(segment - np.mean(segment))
                R = np.max(cumsum) - np.min(cumsum)
                S = np.std(segment)
                if S > 0:
                    rs_segment.append(R / S)
            
            if rs_segment:
                rs_values.append(np.mean(rs_segment))
            else:
                rs_values.append(0)
        
        # Fit log-log | 拟合log-log
        if len(rs_values) >= 2:
            log_scales = np.log(scales)
            log_rs = np.log(rs_values)
            coeffs = np.polyfit(log_scales, log_rs, 1)
            return coeffs[0]
        
        return 0.5
    
    def _compute_fractal_dimension(self, x: np.ndarray) -> float:
        """Compute fractal dimension (Higuchi algorithm) | 计算分形维数（Higuchi算法）"""
        if len(x) < 50:
            return 1.0
        
        n = len(x)
        k_max = min(10, n // 4)
        
        lengths = []
        for k in range(1, k_max + 1):
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
        """Compute Lempel-Ziv complexity | 计算Lempel-Ziv复杂度"""
        if len(x) < 10:
            return 0.0
        
        # Binarize | 二值化
        if threshold == 'median':
            thresh = np.median(x)
        elif threshold == 'mean':
            thresh = np.mean(x)
        else:
            thresh = 0.0
        
        binary = (x > thresh).astype(int)
        s = ''.join(map(str, binary))
        
        # LZ complexity algorithm | LZ复杂度算法
        c = 1
        l = 1
        i = 0
        k = 1
        n = len(s)
        
        while True:
            if i + k >= n:
                break
            sub = s[i:i+k]
            if sub not in s[i:i+k-1]:
                c += 1
                i += k
                k = 1
            else:
                k += 1
            
            if i + k >= n:
                break
        
        # Normalize | 归一化
        norm = n / np.log2(n) if n > 1 else 1
        return c / norm
    
    def _compute_autocorr_lag1(self, x: np.ndarray) -> float:
        """Compute autocorrelation at lag 1 | 计算滞后1自相关系数"""
        if len(x) < 10:
            return 0.0
        try:
            from scipy import signal as sig
            result = sig.correlate(x, x, mode='full')
            if len(result) > 0:
                return float(result[len(result)//2] / (np.var(x) * len(x) + 1e-8))
            return 0.0
        except:
            return 0.0
    
    def _compute_autocorr_lag2(self, x: np.ndarray) -> float:
        """Compute autocorrelation at lag 2 | 计算滞后2自相关系数"""
        if len(x) < 11:
            return 0.0
        try:
            from scipy import signal as sig
            result = sig.correlate(x, x, mode='full')
            if len(result) > 1:
                return float(result[len(result)//2 + 1] / (np.var(x) * len(x) + 1e-8))
            return 0.0
        except:
            return 0.0
    
    def _compute_trend_strength(self, x: np.ndarray) -> float:
        """Compute trend strength | 计算趋势强度"""
        if len(x) < 3:
            return 0.0
        
        t = np.arange(len(x))
        coeffs = np.polyfit(t, x, 1)
        trend = coeffs[0] * t + coeffs[1]
        residual = x - trend
        
        ss_tot = np.sum((x - np.mean(x))**2)
        ss_res = np.sum(residual**2)
        
        if ss_tot == 0:
            return 0.0
        
        r2 = 1 - ss_res / ss_tot
        return r2
    
    def _compute_seasonal_strength(self, x: np.ndarray) -> float:
        """Compute seasonal strength | 计算季节性强度"""
        if len(x) < 20:
            return 0.0
        
        # Detrend | 去趋势
        t = np.arange(len(x))
        coeffs = np.polyfit(t, x, 1)
        detrended = x - (coeffs[0] * t + coeffs[1])
        
        # Estimate period (assume 1/4 of data length) | 估计周期（假设为1/4数据长度）
        n = len(detrended)
        period = n // 4
        if period < 2:
            return 0.0
        
        # Calculate seasonal component | 计算季节分量
        seasonal = np.zeros(n)
        for i in range(n):
            seasonal[i] = np.mean(detrended[i % period::period])
        
        residual = detrended - seasonal
        
        var_total = np.var(detrended)
        var_residual = np.var(residual)
        
        if var_total == 0:
            return 0.0
        
        strength = 1 - var_residual / var_total
        return strength
    
    # ============================================================
    # Main Interface | 主接口
    # ============================================================
    
    def extract(self, data: np.ndarray) -> np.ndarray:
        """
        Extract features from a single time series | 从单个序列提取特征
        
        Args | 参数:
            data: Input time series (n_points,) | 输入时间序列
            
        Returns | 返回:
            np.ndarray: Feature vector | 特征向量
        """
        data = np.asarray(data).flatten()
        
        if len(data) == 0:
            return np.zeros(len(self.features))
        
        features = []
        for feat_def in self.features:
            # Filter by type | 根据类型过滤
            if (self.include_types and 
                feat_def.type not in self.include_types):
                features.append(0.0)
                continue
            
            # Compute feature | 计算特征
            val = feat_def(data)
            features.append(val)
        
        return np.array(features, dtype=np.float64)
    
    def extract_batch(self, data_list: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from multiple time series | 批量提取特征
        
        Args | 参数:
            data_list: List of time series (n_samples, n_points) | 数据列表
            
        Returns | 返回:
            np.ndarray: Feature matrix (n_samples, n_features) | 特征矩阵
        """
        features = []
        for data in data_list:
            feat = self.extract(data)
            features.append(feat)
        
        feature_matrix = np.array(features)
        
        # Normalize | 归一化
        if self.normalize:
            feature_matrix = self._normalize_features(feature_matrix)
        
        return feature_matrix
    
    def _normalize_features(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Normalize feature matrix (Z-score) | 归一化特征矩阵（Z-score）
        
        Args | 参数:
            feature_matrix: Raw feature matrix | 原始特征矩阵
            
        Returns | 返回:
            np.ndarray: Normalized feature matrix | 归一化后的特征矩阵
        """
        normalized = np.zeros_like(feature_matrix)
        
        for j in range(feature_matrix.shape[1]):
            col = feature_matrix[:, j]
            
            # Skip constant columns | 跳过常数列
            if np.std(col) < 1e-10:
                normalized[:, j] = 0.5
                continue
            
            # Z-score normalization | Z-score归一化
            mean = np.mean(col)
            std = np.std(col)
            normalized[:, j] = (col - mean) / (std + 1e-8)
            
            # Record statistics | 记录统计信息
            self.feature_stats[self.features[j].name] = {
                'mean': float(mean),
                'std': float(std)
            }
        
        return normalized
    
    def get_feature_names(self) -> List[str]:
        """Get feature name list | 获取特征名称列表"""
        return [f.name for f in self.features]
    
    def get_feature_types(self) -> List[str]:
        """Get feature type list | 获取特征类型列表"""
        return [f.type.value for f in self.features]
    
    def get_feature_descriptions(self) -> List[str]:
        """Get feature description list | 获取特征描述列表"""
        return [f.description for f in self.features]
    
    def get_feature_count(self) -> int:
        """Get number of features | 获取特征数量"""
        return len(self.features)
    
    def add_custom_feature(self, feature: FeatureDefinition) -> None:
        """
        Add a custom feature | 添加自定义特征
        
        Args | 参数:
            feature: Feature definition | 特征定义
        """
        self.features.append(feature)
        self.logger.info(f"Added custom feature: {feature.name} | 添加自定义特征")
    
    def remove_feature(self, name: str) -> bool:
        """
        Remove a feature by name | 移除特征
        
        Args | 参数:
            name: Feature name | 特征名称
            
        Returns | 返回:
            bool: True if successful | 是否成功
        """
        for i, f in enumerate(self.features):
            if f.name == name:
                self.features.pop(i)
                self.logger.info(f"Removed feature: {name} | 移除特征")
                return True
        return False
    
    def save(self, filename: str) -> bool:
        """
        Save feature extractor configuration | 保存特征器配置
        
        Args | 参数:
            filename: Output filename | 文件名
            
        Returns | 返回:
            bool: True if successful | 是否成功
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
            self.logger.info(f"Configuration saved to {filename} | 配置已保存")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save: {e} | 保存失败")
            return False
    
    def load(self, filename: str) -> bool:
        """
        Load feature extractor configuration | 加载特征器配置
        
        Args | 参数:
            filename: Input filename | 文件名
            
        Returns | 返回:
            bool: True if successful | 是否成功
        """
        try:
            import json
            with open(filename, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.normalize = config.get('normalize', True)
            if config.get('include_types'):
                self.include_types = [FeatureType(t) for t in config['include_types']]
            self.feature_stats = config.get('feature_stats', {})
            
            self.logger.info(f"Configuration loaded from {filename} | 配置已加载")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load: {e} | 加载失败")
            return False