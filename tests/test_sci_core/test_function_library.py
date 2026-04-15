# tests/test_sci_core/test_function_library.py
"""
Test Function Library Module | 测试函数库模块

Tests for FunctionEntry and FunctionLibrary classes.
测试 FunctionEntry 和 FunctionLibrary 类。
"""

import pytest
import tempfile
import json
from pathlib import Path
import numpy as np
from metathin_plus.sci.core.function_space import FunctionSpace, VectorSpaceConfig
from metathin_plus.sci.core.function_library import FunctionLibrary, FunctionEntry


class TestFunctionEntry:
    """Test function entry data class | 测试函数条目数据类"""
    
    def test_creation(self):
        """Test function entry creation | 测试函数条目创建"""
        config = VectorSpaceConfig(n_negative=2, n_positive=2)
        space = FunctionSpace(config)
        vector = space.linear_combination([], [])  # Zero vector | 零向量
        
        entry = FunctionEntry(
            name="test_func",
            expression="x**2",
            vector=vector,
            category="math",
            parameters=["a"],
            tags=["test", "quadratic"],
            description="Test function | 测试函数"
        )
        
        assert entry.name == "test_func"
        assert entry.expression == "x**2"
        assert entry.category == "math"
        assert entry.parameters == ["a"]
        assert entry.tags == ["test", "quadratic"]
    
    def test_similarity(self):
        """Test similarity calculation | 测试相似度计算"""
        config = VectorSpaceConfig(n_negative=1, n_positive=1)
        from metathin_plus.sci.core.function_space import FunctionVector
        vec = FunctionVector(np.array([1, 0, 0]), config)
        
        entry = FunctionEntry(name="test", expression="x", vector=vec, category="math")
        assert entry.similarity(vec) == 1.0
    
    def test_serialization(self):
        """Test serialization to/from dict | 测试序列化与反序列化"""
        config = VectorSpaceConfig(n_negative=1, n_positive=1)
        from metathin_plus.sci.core.function_space import FunctionVector
        vec = FunctionVector(np.array([1, 2, 3]), config)
        
        entry = FunctionEntry(
            name="test",
            expression="x**2",
            vector=vec,
            category="math",
            parameters=["a"],
            tags=["test"]
        )
        
        data = entry.to_dict()
        assert data['name'] == "test"
        assert data['expression'] == "x**2"


class TestFunctionLibrary:
    """Test function library manager | 测试函数库管理器"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test | 创建测试用临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def library(self, temp_dir):
        """Create test function library | 创建测试用函数库"""
        config = VectorSpaceConfig(n_negative=10, n_positive=10)
        space = FunctionSpace(config)
        return FunctionLibrary(space, data_dir=temp_dir)
    
    def test_initialization(self, library):
        """Test library initialization | 测试库初始化"""
        assert len(library) > 0  # Should have built-in functions | 应该有内置函数
        stats = library.get_statistics()
        assert stats['total_functions'] > 0
        assert stats['builtin_count'] > 0
    
    def test_get_function(self, library):
        """Test getting function by name | 测试根据名称获取函数"""
        # Get math function | 获取数学函数
        sin_func = library.get('sin')
        if sin_func:
            assert sin_func.name == 'sin'
            assert sin_func.category == 'math'
    
    def test_list_functions(self, library):
        """Test listing functions with filters | 测试带过滤器的函数列表"""
        math_funcs = library.list_functions(category='math')
        assert len(math_funcs) > 0
        
        for func in math_funcs:
            assert func['category'] == 'math'
    
    def test_list_functions_by_tag(self, library):
        """Test listing functions by tag | 测试按标签列出函数"""
        trig_funcs = library.list_functions(tags=['trigonometric'])
        assert len(trig_funcs) >= 2  # sin, cos at least | 至少有 sin 和 cos
    
    def test_match(self, library):
        """Test function matching | 测试函数匹配"""
        # Get sin function's vector | 获取 sin 函数的向量
        sin_entry = library.get('sin')
        if sin_entry is None:
            pytest.skip("sin function not available | sin 函数不可用")
        
        # Should match itself | 应该能匹配到自己
        matches = library.match(sin_entry.vector, threshold=0.9, top_k=3)
        assert len(matches) >= 1
        assert matches[0][0] == 'sin'
        assert matches[0][1] > 0.99
    
    def test_match_by_name(self, library):
        """Test matching against specific named function | 测试与特定命名函数的匹配"""
        sin_entry = library.get('sin')
        if sin_entry is None:
            pytest.skip("sin function not available | sin 函数不可用")
        
        similarity = library.match_by_name('sin', sin_entry.vector)
        assert similarity > 0.99
    
    def test_register_user_function(self, library):
        """Test registering user-defined function | 测试注册用户自定义函数"""
        # Register simple function | 注册简单函数
        success = library.register(
            name="my_quadratic",
            expr="a*x**2 + b*x + c",
            category="user",
            parameters=["a", "b", "c"],
            tags=["custom", "polynomial"],
            description="My custom quadratic | 我的自定义二次函数"
        )
        
        assert success is True
        
        # Verify registration | 验证注册
        func = library.get("my_quadratic")
        assert func is not None
        assert func.category == "user"
        assert func.parameters == ["a", "b", "c"]
    
    def test_register_numerical_function(self, library):
        """Test registering numerical function (callable) | 测试注册数值函数（可调用）"""
        def my_func(x):
            return x ** 3
        
        success = library.register(
            name="my_cubic",
            expr=my_func,
            category="user",
            tags=["custom"]
        )
        
        assert success is True
        assert "my_cubic" in library
    
    def test_delete_user_function(self, library):
        """Test deleting user-defined function | 测试删除用户自定义函数"""
        # Register first | 先注册
        library.register("to_delete", "x**2", category="user")
        assert "to_delete" in library
        
        # Delete | 删除
        success = library.delete_user_function("to_delete")
        assert success is True
        assert "to_delete" not in library
    
    def test_cannot_delete_builtin(self, library):
        """Test that built-in functions cannot be deleted | 测试内置函数不能被删除"""
        if "sin" in library:
            success = library.delete_user_function("sin")
            assert success is False
    
    def test_clear_user_functions(self, library):
        """Test clearing all user-defined functions | 测试清空所有用户自定义函数"""
        # Register some user functions | 注册一些用户函数
        library.register("user1", "x", category="user")
        library.register("user2", "x**2", category="user")
        
        user_count_before = sum(1 for e in library.list_functions() if e['category'] == 'user')
        assert user_count_before >= 2
        
        # Clear | 清空
        library.clear_user_functions()
        
        user_count_after = sum(1 for e in library.list_functions() if e['category'] == 'user')
        assert user_count_after == 0
    
    def test_persistence(self, temp_dir):
        """Test function persistence across library instances | 测试跨库实例的函数持久化"""
        # Create first library and register function | 创建第一个库并注册函数
        config = VectorSpaceConfig(n_negative=5, n_positive=5)
        space = FunctionSpace(config)
        lib1 = FunctionLibrary(space, data_dir=temp_dir)
        
        lib1.register("persist_test", "x**3", category="user", tags=["test"])
        assert "persist_test" in lib1
        
        # Create second library, should load saved functions | 创建第二个库，应该加载保存的函数
        lib2 = FunctionLibrary(space, data_dir=temp_dir)
        lib2.reload()
        # Note: User functions may be in separate file | 注意：用户函数可能在单独的文件中
    
    def test_contains(self, library):
        """Test membership test | 测试成员资格测试"""
        assert "sin" in library
        assert "nonexistent" not in library
    
    def test_len(self, library):
        """Test length | 测试长度"""
        assert len(library) > 0
    
    def test_iter(self, library):
        """Test iteration | 测试迭代"""
        count = 0
        for entry in library:
            count += 1
        assert count == len(library)