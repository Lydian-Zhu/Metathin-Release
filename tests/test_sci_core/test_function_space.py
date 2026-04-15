# tests/test_sci_core/test_function_space.py
"""
Test Function Vector Space Module | 测试函数向量空间模块

Tests for FunctionVector, FunctionSpace, VectorSpaceConfig.
测试 FunctionVector、FunctionSpace、VectorSpaceConfig。
"""

import pytest
import numpy as np
from metathin_plus.sci.core.function_space import (
    VectorSpaceConfig,
    FunctionVector,
    FunctionSpace
)


class TestVectorSpaceConfig:
    """Test vector space configuration | 测试向量空间配置"""
    
    def test_default_config(self):
        """Test default configuration | 测试默认配置"""
        config = VectorSpaceConfig()
        assert config.n_negative == 20
        assert config.n_positive == 20
        assert config.center == 0.0
        assert config.dimension == 41  # 20 + 20 + 1
    
    def test_custom_config(self):
        """Test custom configuration | 测试自定义配置"""
        config = VectorSpaceConfig(n_negative=10, n_positive=15, center=2.0)
        assert config.n_negative == 10
        assert config.n_positive == 15
        assert config.center == 2.0
        assert config.dimension == 26  # 10 + 15 + 1
    
    def test_serialization(self):
        """Test serialization to/from dict | 测试序列化与反序列化"""
        config = VectorSpaceConfig(n_negative=5, n_positive=5, center=1.0)
        data = config.to_dict()
        
        restored = VectorSpaceConfig.from_dict(data)
        assert restored.n_negative == 5
        assert restored.n_positive == 5
        assert restored.center == 1.0


class TestFunctionVector:
    """Test function vector operations | 测试函数向量操作"""
    
    def test_creation(self):
        """Test vector creation | 测试向量创建"""
        config = VectorSpaceConfig(n_negative=2, n_positive=2)  # dimension 5
        coeffs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        vec = FunctionVector(coeffs, config)
        
        assert len(vec) == 5
        assert vec.norm > 0
        np.testing.assert_array_equal(vec.coefficients, coeffs)
    
    def test_normalization_padding(self):
        """Test automatic zero padding for short coefficient arrays | 测试短系数数组自动补零"""
        config = VectorSpaceConfig(n_negative=2, n_positive=2)  # dimension 5
        coeffs = np.array([0.1, 0.2])  # insufficient length | 长度不足
        vec = FunctionVector(coeffs, config)
        
        assert len(vec) == 5
        assert vec.coefficients[0] == 0.1
        assert vec.coefficients[1] == 0.2
        assert vec.coefficients[2] == 0.0
        assert vec.coefficients[3] == 0.0
        assert vec.coefficients[4] == 0.0
    
    def test_normalization_truncation(self):
        """Test automatic truncation for long coefficient arrays | 测试长系数数组自动截断"""
        config = VectorSpaceConfig(n_negative=1, n_positive=1)  # dimension 3
        coeffs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # excess length | 长度超出
        vec = FunctionVector(coeffs, config)
        
        assert len(vec) == 3
        assert vec.coefficients[0] == 0.1
        assert vec.coefficients[1] == 0.2
        assert vec.coefficients[2] == 0.3
    
    def test_dot_product(self):
        """Test dot product calculation | 测试点积计算"""
        config = VectorSpaceConfig(n_negative=1, n_positive=1)
        vec1 = FunctionVector(np.array([1, 2, 3]), config)
        vec2 = FunctionVector(np.array([4, 5, 6]), config)
        
        dot = vec1.dot(vec2)
        assert dot == 1*4 + 2*5 + 3*6  # = 32
    
    def test_similarity(self):
        """Test cosine similarity calculation | 测试余弦相似度计算"""
        config = VectorSpaceConfig(n_negative=1, n_positive=1)
        
        # Identical vectors | 相同向量
        vec1 = FunctionVector(np.array([1, 0, 0]), config)
        vec2 = FunctionVector(np.array([1, 0, 0]), config)
        assert vec1.similarity(vec2) == 1.0
        
        # Orthogonal vectors | 正交向量
        vec3 = FunctionVector(np.array([1, 0, 0]), config)
        vec4 = FunctionVector(np.array([0, 1, 0]), config)
        assert abs(vec3.similarity(vec4)) < 1e-10
        
        # Opposite vectors | 相反向量
        vec5 = FunctionVector(np.array([1, 2, 3]), config)
        vec6 = FunctionVector(np.array([-1, -2, -3]), config)
        assert vec5.similarity(vec6) == -1.0
    
    def test_vector_operations(self):
        """Test vector arithmetic operations | 测试向量算术运算"""
        config = VectorSpaceConfig(n_negative=1, n_positive=1)
        vec1 = FunctionVector(np.array([1, 2, 3]), config)
        vec2 = FunctionVector(np.array([4, 5, 6]), config)
        
        # Addition | 加法
        vec_sum = vec1.add(vec2)
        np.testing.assert_array_equal(vec_sum.coefficients, [5, 7, 9])
        
        # Subtraction | 减法
        vec_diff = vec1.subtract(vec2)
        np.testing.assert_array_equal(vec_diff.coefficients, [-3, -3, -3])
        
        # Scalar multiplication | 标量乘法
        vec_scaled = vec1.scale(2)
        np.testing.assert_array_equal(vec_scaled.coefficients, [2, 4, 6])
    
    def test_serialization(self):
        """Test vector serialization to/from dict | 测试向量序列化与反序列化"""
        config = VectorSpaceConfig(n_negative=2, n_positive=2)
        vec = FunctionVector(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), config)
        
        data = vec.to_dict()
        restored = FunctionVector.from_dict(data)
        
        np.testing.assert_array_equal(restored.coefficients, vec.coefficients)
        assert restored.config.dimension == vec.config.dimension
    
    def test_equality(self):
        """Test vector equality comparison | 测试向量相等性比较"""
        config = VectorSpaceConfig(n_negative=1, n_positive=1)
        vec1 = FunctionVector(np.array([1, 2, 3]), config)
        vec2 = FunctionVector(np.array([1, 2, 3]), config)
        vec3 = FunctionVector(np.array([1, 2, 4]), config)
        
        assert vec1 == vec2
        assert vec1 != vec3


class TestFunctionSpace:
    """Test function space operations | 测试函数空间操作"""
    
    def test_creation(self):
        """Test function space creation | 测试函数空间创建"""
        space = FunctionSpace()
        assert space.config.dimension == 41
    
    def test_linear_combination(self):
        """Test linear combination of vectors | 测试向量线性组合"""
        config = VectorSpaceConfig(n_negative=1, n_positive=1)
        space = FunctionSpace(config)
        
        vec1 = FunctionVector(np.array([1, 0, 0]), config)
        vec2 = FunctionVector(np.array([0, 1, 0]), config)
        
        result = space.linear_combination([vec1, vec2], [2, 3])
        np.testing.assert_array_equal(result.coefficients, [2, 3, 0])
    
    def test_similarity_matrix(self):
        """Test pairwise similarity matrix calculation | 测试两两相似度矩阵计算"""
        config = VectorSpaceConfig(n_negative=1, n_positive=1)
        space = FunctionSpace(config)
        
        vec1 = FunctionVector(np.array([1, 0, 0]), config)
        vec2 = FunctionVector(np.array([0, 1, 0]), config)
        vec3 = FunctionVector(np.array([1, 1, 0]), config)
        
        matrix = space.similarity_matrix([vec1, vec2, vec3])
        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 1.0
        assert abs(matrix[0, 1]) < 1e-10
        assert matrix[0, 2] > 0
    
    def test_distance_matrix(self):
        """Test pairwise distance matrix calculation | 测试两两距离矩阵计算"""
        config = VectorSpaceConfig(n_negative=1, n_positive=1)
        space = FunctionSpace(config)
        
        vec1 = FunctionVector(np.array([1, 0, 0]), config)
        vec2 = FunctionVector(np.array([1, 0, 0]), config)
        
        matrix = space.distance_matrix([vec1, vec2])
        assert matrix[0, 1] == 0.0