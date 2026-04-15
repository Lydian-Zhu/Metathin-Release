"""
Laurent Series Function Space | 洛朗级数向量空间
===================================================

Maps functions to vectors in a 41-dimensional space via Laurent series expansion.
通过洛朗级数展开将函数映射到41维向量空间。

Mathematical Foundation | 数学基础:
    f(z) = Σ_{n=-∞}^{∞} a_n (z - c)^n
    
    Vector representation | 向量表示:
        \vec{f} = [a_{-N}, ..., a_{-1}, a_0, a_1, ..., a_{N}]
    
    Similarity | 相似度:
        r = \frac{\vec{f} \cdot \vec{g}}{\|\vec{f}\| \|\vec{g}\|}

Design Principles | 设计原则:
    - Pure mathematics, no Metathin dependency | 纯数学，无 Metathin 依赖
    - Immutable vectors | 不可变向量
    - Serializable | 可序列化
"""

import numpy as np
from typing import Callable, Optional, List, Tuple, Any
from dataclasses import dataclass
import logging


@dataclass
class VectorSpaceConfig:
    """
    Vector space configuration | 向量空间配置
    
    Attributes:
        n_negative: Number of negative power terms (default 20) | 负幂项数量
        n_positive: Number of positive power terms (default 20) | 正幂项数量
        center: Expansion center point (default 0) | 展开中心点
        eps: Numerical tolerance | 数值容差
    """
    n_negative: int = 20
    n_positive: int = 20
    center: float = 0.0
    eps: float = 1e-10
    
    @property
    def dimension(self) -> int:
        """Total dimension of the vector space | 向量空间总维度"""
        return self.n_negative + self.n_positive + 1
    
    def to_dict(self) -> dict:
        """Convert to dictionary | 转换为字典"""
        return {
            'n_negative': self.n_negative,
            'n_positive': self.n_positive,
            'center': self.center,
            'eps': self.eps
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'VectorSpaceConfig':
        """Create from dictionary | 从字典创建"""
        return cls(
            n_negative=data.get('n_negative', 20),
            n_positive=data.get('n_positive', 20),
            center=data.get('center', 0.0),
            eps=data.get('eps', 1e-10)
        )


class FunctionVector:
    """
    Function Vector - Immutable representation in Laurent space | 函数向量
    
    Represents a function as a vector in the Laurent series coefficient space.
    将函数表示为洛朗级数系数空间中的向量。
    
    Features | 特性:
        - Immutable: All operations return new vectors | 不可变：所有操作返回新向量
        - Serializable: Can be saved to JSON | 可序列化：可保存到 JSON
        - Supports vector operations | 支持向量运算
    """
    
    def __init__(self, coefficients: np.ndarray, config: VectorSpaceConfig):
        """
        Initialize function vector | 初始化函数向量
        
        Args:
            coefficients: Laurent series coefficients | 洛朗级数系数
            config: Vector space configuration | 向量空间配置
        """
        self._config = config
        self._coefficients = self._normalize(coefficients)
        self._logger = logging.getLogger("metathin_sci.FunctionVector")
    
    def _normalize(self, coeffs: np.ndarray) -> np.ndarray:
        """Normalize coefficients to standard dimension | 归一化系数到标准维度"""
        arr = np.array(coeffs, dtype=np.float64)
        target_dim = self._config.dimension
        
        if len(arr) < target_dim:
            # Pad with zeros | 补零
            arr = np.pad(arr, (0, target_dim - len(arr)))
        elif len(arr) > target_dim:
            # Truncate | 截断
            arr = arr[:target_dim]
        
        # Replace NaN and Inf | 替换 NaN 和 Inf
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    
    @property
    def coefficients(self) -> np.ndarray:
        """Get coefficients (copy) | 获取系数（副本）"""
        return self._coefficients.copy()
    
    @property
    def config(self) -> VectorSpaceConfig:
        """Get configuration | 获取配置"""
        return self._config
    
    @property
    def norm(self) -> float:
        """Euclidean norm | 欧几里得范数"""
        return float(np.linalg.norm(self._coefficients))
    
    def dot(self, other: 'FunctionVector') -> float:
        """
        Dot product | 点积
        
        Args:
            other: Another function vector | 另一个函数向量
            
        Returns:
            float: Dot product value | 点积值
        """
        min_len = min(len(self._coefficients), len(other._coefficients))
        return float(np.dot(self._coefficients[:min_len], other._coefficients[:min_len]))
    
    def similarity(self, other: 'FunctionVector') -> float:
        """
        Cosine similarity | 余弦相似度
        
        r = (a·b) / (|a|·|b|)
        
        Args:
            other: Another function vector | 另一个函数向量
            
        Returns:
            float: Similarity in range [-1, 1] | 相似度，范围 [-1, 1]
        """
        norm_product = self.norm * other.norm
        if norm_product < self._config.eps:
            return 0.0
        return self.dot(other) / norm_product
    
    def add(self, other: 'FunctionVector') -> 'FunctionVector':
        """Vector addition | 向量加法"""
        return FunctionVector(self._coefficients + other._coefficients, self._config)
    
    def subtract(self, other: 'FunctionVector') -> 'FunctionVector':
        """Vector subtraction | 向量减法"""
        return FunctionVector(self._coefficients - other._coefficients, self._config)
    
    def scale(self, alpha: float) -> 'FunctionVector':
        """Scalar multiplication | 标量乘法"""
        return FunctionVector(alpha * self._coefficients, self._config)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization | 转换为字典用于序列化"""
        return {
            'coefficients': self._coefficients.tolist(),
            'config': self._config.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FunctionVector':
        """Create from dictionary | 从字典创建"""
        config = VectorSpaceConfig.from_dict(data.get('config', {}))
        return cls(np.array(data['coefficients']), config)
    
    def __len__(self) -> int:
        return len(self._coefficients)
    
    def __repr__(self) -> str:
        return f"FunctionVector(dim={len(self)}, norm={self.norm:.4f})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, FunctionVector):
            return False
        return np.allclose(self._coefficients, other._coefficients, atol=self._config.eps)


class FunctionSpace:
    """
    Function Vector Space | 函数向量空间
    
    Manages the mapping between functions and vectors.
    管理函数与向量之间的映射。
    
    Operations | 操作:
        - Symbolic expression → Vector | 符号表达式 → 向量
        - Numerical samples → Vector | 数值采样 → 向量
        - Vector → Callable function | 向量 → 可调用函数
        - Linear combinations | 线性组合
    """
    
    def __init__(self, config: Optional[VectorSpaceConfig] = None):
        """
        Initialize function space | 初始化函数空间
        
        Args:
            config: Vector space configuration | 向量空间配置
        """
        self.config = config or VectorSpaceConfig()
        self._logger = logging.getLogger("metathin_sci.FunctionSpace")
    
    def from_symbolic(self, expr, x_symbol=None) -> FunctionVector:
        """
        Create vector from symbolic expression | 从符号表达式创建向量
        
        Uses SymPy to compute Laurent series coefficients analytically.
        使用 SymPy 解析计算洛朗级数系数。
        
        Args:
            expr: SymPy expression | SymPy 表达式
            x_symbol: Symbol for variable (default: 'x') | 变量符号
            
        Returns:
            FunctionVector: Vector representation | 向量表示
        """
        from sympy import symbols, limit, factorial
        
        if x_symbol is None:
            x_symbol = symbols('x')
        
        coeffs = []
        center = self.config.center
        
        # Negative power terms | 负幂项
        for n in range(1, self.config.n_negative + 1):
            try:
                term = (x_symbol - center) ** n * expr
                val = limit(term, x_symbol, center)
                coeffs.append(float(val) if hasattr(val, 'is_number') and val.is_number else 0.0)
            except Exception:
                coeffs.append(0.0)
        
        # Constant term | 常数项
        try:
            val = limit(expr, x_symbol, center)
            coeffs.append(float(val) if hasattr(val, 'is_number') and val.is_number else 0.0)
        except Exception:
            coeffs.append(0.0)
        
        # Positive power terms | 正幂项
        for n in range(1, self.config.n_positive + 1):
            try:
                deriv = expr.diff(x_symbol, n)
                val = limit(deriv, x_symbol, center) / factorial(n)
                coeffs.append(float(val) if hasattr(val, 'is_number') and val.is_number else 0.0)
            except Exception:
                coeffs.append(0.0)
        
        return FunctionVector(np.array(coeffs), self.config)
    
    def from_samples(self, x_vals: np.ndarray, y_vals: np.ndarray) -> FunctionVector:
        """
        Create vector from numerical samples | 从数值采样创建向量
        
        Uses least squares to fit Laurent coefficients.
        使用最小二乘法拟合洛朗系数。
        
        Args:
            x_vals: X coordinates | X 坐标数组
            y_vals: Y values | Y 值数组
            
        Returns:
            FunctionVector: Vector representation | 向量表示
        """
        n = len(x_vals)
        dim = self.config.dimension
        
        # Build design matrix | 构建设计矩阵
        A = np.zeros((n, dim))
        for i, x in enumerate(x_vals):
            dx = x - self.config.center
            for j in range(dim):
                power = j - self.config.n_negative
                A[i, j] = dx ** power
        
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, y_vals, rcond=None)
            return FunctionVector(coeffs, self.config)
        except Exception as e:
            self._logger.warning(f"Least squares failed: {e}")
            return FunctionVector(np.zeros(dim), self.config)
    
    def to_function(self, vector: FunctionVector) -> Callable[[float], float]:
        """
        Reconstruct callable function from vector | 从向量重建可调用函数
        
        Args:
            vector: Function vector | 函数向量
            
        Returns:
            Callable: Function f(x) | 可调用函数
        """
        coeffs = vector.coefficients
        center = self.config.center
        n_neg = self.config.n_negative
        
        def f(x: float) -> float:
            dx = x - center
            result = 0.0
            for j, coeff in enumerate(coeffs):
                power = j - n_neg
                result += coeff * (dx ** power)
            return result
        
        return f
    
    def linear_combination(self, 
                          vectors: List[FunctionVector], 
                          weights: List[float]) -> FunctionVector:
        """
        Linear combination of vectors | 向量线性组合
        
        Result = Σ w_i * v_i
        
        Args:
            vectors: List of vectors | 向量列表
            weights: Corresponding weights | 对应权重
            
        Returns:
            FunctionVector: Combined vector | 组合向量
        """
        if not vectors:
            return FunctionVector(np.zeros(self.config.dimension), self.config)
        
        result = np.zeros(self.config.dimension)
        for vec, w in zip(vectors, weights):
            result += w * vec.coefficients
        
        return FunctionVector(result, self.config)
    
    def similarity_matrix(self, vectors: List[FunctionVector]) -> np.ndarray:
        """
        Compute pairwise similarity matrix | 计算两两相似度矩阵
        
        Args:
            vectors: List of vectors | 向量列表
            
        Returns:
            np.ndarray: Similarity matrix (n x n) | 相似度矩阵
        """
        n = len(vectors)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = vectors[i].similarity(vectors[j])
        return matrix
    
    def distance_matrix(self, vectors: List[FunctionVector]) -> np.ndarray:
        """
        Compute pairwise distance matrix | 计算两两距离矩阵
        
        distance = 1 - similarity
        
        Args:
            vectors: List of vectors | 向量列表
            
        Returns:
            np.ndarray: Distance matrix (n x n) | 距离矩阵
        """
        return 1.0 - self.similarity_matrix(vectors)