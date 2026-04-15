"""
Function Library Management | 函数库管理
=========================================

Manages built-in and user-defined functions with persistent storage.
管理内置函数和用户自定义函数，支持持久化存储。

Features | 特性:
    - Built-in mathematical functions | 内置数学函数
    - Physics/engineering functions | 物理工程函数
    - Chemistry functions | 化学函数
    - User function registration | 用户函数注册
    - Persistent JSON storage | 持久化 JSON 存储
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

import numpy as np

from .function_space import FunctionSpace, FunctionVector, VectorSpaceConfig


# ============================================================
# Function Entry Data Class | 函数条目数据类
# ============================================================

@dataclass
class FunctionEntry:
    """
    Function library entry | 函数库条目
    
    Represents a function in the library with its vector representation.
    表示库中的函数及其向量表示。
    
    Attributes:
        name: Function name | 函数名称
        expression: Symbolic expression string | 符号表达式字符串
        vector: Function vector | 函数向量
        category: Function category (math, physics, chemistry, user) | 函数类别
        parameters: List of parameter names | 参数名称列表
        tags: Search tags | 搜索标签
        description: Human-readable description | 人类可读描述
        metadata: Additional metadata | 附加元数据
    """
    name: str
    expression: str
    vector: FunctionVector
    category: str = "user"
    parameters: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def similarity(self, other: FunctionVector) -> float:
        """Compute similarity with another vector | 计算与另一向量的相似度"""
        return self.vector.similarity(other)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization | 转换为字典用于序列化"""
        return {
            'name': self.name,
            'expression': self.expression,
            'vector': self.vector.to_dict(),
            'category': self.category,
            'parameters': self.parameters,
            'tags': self.tags,
            'description': self.description,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict, space: FunctionSpace) -> 'FunctionEntry':
        """Create from dictionary | 从字典创建"""
        vector = FunctionVector.from_dict(data['vector'])
        return cls(
            name=data['name'],
            expression=data['expression'],
            vector=vector,
            category=data.get('category', 'user'),
            parameters=data.get('parameters', []),
            tags=data.get('tags', []),
            description=data.get('description', ''),
            metadata=data.get('metadata', {})
        )
    
    def __repr__(self) -> str:
        return f"FunctionEntry(name='{self.name}', category='{self.category}', dim={len(self.vector)})"


# ============================================================
# Function Library Manager | 函数库管理器
# ============================================================

class FunctionLibrary:
    """
    Function Library Manager | 函数库管理器
    
    Manages built-in and user-defined functions with file persistence.
    管理内置函数和用户自定义函数，支持文件持久化。
    
    File Structure | 文件结构:
        - builtin_functions.json: Pre-defined functions (read-only) | 预定义函数（只读）
        - user_functions.json: User-registered functions (read-write) | 用户注册函数（读写）
    
    Categories | 类别:
        - math: Basic mathematical functions | 基础数学函数
        - physics: Physics/engineering functions | 物理/工程函数
        - chemistry: Chemistry functions | 化学函数
        - user: User-defined functions | 用户自定义函数
    """
    
    # Built-in function definitions | 内置函数定义
    _MATH_FUNCTIONS = [
        # Basic functions | 基础函数
        ('constant', '1', 'math', [], ['constant', 'basic'], 'Constant function f(x) = 1'),
        ('linear', 'x', 'math', [], ['linear', 'polynomial'], 'Linear function f(x) = x'),
        ('quadratic', 'x**2', 'math', [], ['quadratic', 'polynomial'], 'Quadratic function f(x) = x²'),
        ('cubic', 'x**3', 'math', [], ['cubic', 'polynomial'], 'Cubic function f(x) = x³'),
        ('inverse', '1/x', 'math', [], ['inverse', 'rational'], 'Inverse function f(x) = 1/x'),
        ('inverse_square', '1/x**2', 'math', [], ['inverse', 'rational'], 'Inverse square f(x) = 1/x²'),
        
        # Trigonometric functions | 三角函数
        ('sin', 'sin(x)', 'math', [], ['trigonometric', 'periodic'], 'Sine function'),
        ('cos', 'cos(x)', 'math', [], ['trigonometric', 'periodic'], 'Cosine function'),
        ('tan', 'tan(x)', 'math', [], ['trigonometric', 'periodic'], 'Tangent function'),
        ('arcsin', 'asin(x)', 'math', [], ['trigonometric', 'inverse'], 'Arcsine function'),
        ('arccos', 'acos(x)', 'math', [], ['trigonometric', 'inverse'], 'Arccosine function'),
        ('arctan', 'atan(x)', 'math', [], ['trigonometric', 'inverse'], 'Arctangent function'),
        
        # Hyperbolic functions | 双曲函数
        ('sinh', 'sinh(x)', 'math', [], ['hyperbolic'], 'Hyperbolic sine'),
        ('cosh', 'cosh(x)', 'math', [], ['hyperbolic'], 'Hyperbolic cosine'),
        ('tanh', 'tanh(x)', 'math', [], ['hyperbolic'], 'Hyperbolic tangent'),
        
        # Exponential and logarithmic | 指数和对数
        ('exp', 'exp(x)', 'math', [], ['exponential', 'growth'], 'Exponential function'),
        ('exp_decay', 'exp(-x)', 'math', [], ['exponential', 'decay'], 'Exponential decay'),
        ('ln', 'log(x)', 'math', [], ['logarithmic'], 'Natural logarithm'),
        ('log10', 'log(x,10)', 'math', [], ['logarithmic'], 'Base-10 logarithm'),
        ('log2', 'log(x,2)', 'math', [], ['logarithmic'], 'Base-2 logarithm'),
        
        # Power and root | 幂和根
        ('sqrt', 'sqrt(x)', 'math', [], ['power', 'root'], 'Square root'),
        ('cbrt', 'x**(1/3)', 'math', [], ['power', 'root'], 'Cube root'),
        ('power2', 'x**2', 'math', [], ['power'], 'Square'),
        ('power3', 'x**3', 'math', [], ['power'], 'Cube'),
        
        # Special functions | 特殊函数
        ('erf', 'erf(x)', 'math', [], ['special', 'error'], 'Error function'),
        ('erfc', 'erfc(x)', 'math', [], ['special', 'error'], 'Complementary error function'),
        ('gamma', 'gamma(x)', 'math', [], ['special', 'gamma'], 'Gamma function'),
        ('bessel_j0', 'besselj(0, x)', 'math', [], ['bessel', 'special'], 'Bessel function J₀'),
        ('bessel_j1', 'besselj(1, x)', 'math', [], ['bessel', 'special'], 'Bessel function J₁'),
        ('bessel_y0', 'bessely(0, x)', 'math', [], ['bessel', 'special'], 'Bessel function Y₀'),
    ]
    
    _PHYSICS_FUNCTIONS = [
        # Oscillations | 振动
        ('damped_oscillation', 'A*exp(-γ*x)*sin(ω*x + φ)', 'physics', 
         ['A', 'γ', 'ω', 'φ'], ['damped', 'oscillation'], 
         'Damped harmonic oscillation: A·e^{-γx}·sin(ωx + φ)'),
        
        ('forced_oscillation', 'A*sin(ω_d*x + φ) + B*exp(-γ*x)*sin(ω_n*x)', 'physics',
         ['A', 'ω_d', 'φ', 'B', 'γ', 'ω_n'], ['forced', 'oscillation'],
         'Forced oscillation with damping'),
        
        # Quantum mechanics | 量子力学
        ('hydrogen_radial', 'r**2 * exp(-r/a0)', 'physics',
         ['a0'], ['quantum', 'radial'],
         'Hydrogen atom radial wavefunction (1s)'),
        
        ('gaussian_wavepacket', 'exp(-(x-x0)**2/(2σ**2)) * exp(i*k*x)', 'physics',
         ['x0', 'σ', 'k'], ['quantum', 'wavepacket'],
         'Gaussian wave packet'),
        
        ('lorentzian', '1/(π*γ) * 1/(1 + ((x-x0)/γ)**2)', 'physics',
         ['x0', 'γ'], ['spectral', 'lineshape'],
         'Lorentzian lineshape'),
        
        # Thermodynamics | 热力学
        ('boltzmann', 'exp(-E/(k*T))', 'physics',
         ['E', 'k', 'T'], ['statistical', 'probability'],
         'Boltzmann distribution'),
        
        ('fermi_dirac', '1/(exp((E-μ)/(k*T)) + 1)', 'physics',
         ['μ', 'k', 'T'], ['quantum', 'statistical'],
         'Fermi-Dirac distribution'),
        
        ('bose_einstein', '1/(exp((E-μ)/(k*T)) - 1)', 'physics',
         ['μ', 'k', 'T'], ['quantum', 'statistical'],
         'Bose-Einstein distribution'),
        
        # Electromagnetism | 电磁学
        ('coulomb', 'k*q1*q2/x**2', 'physics',
         ['k', 'q1', 'q2'], ['electrostatic', 'inverse_square'],
         'Coulomb\'s law'),
        
        ('exponential_decay_rc', 'V0*exp(-t/(R*C))', 'physics',
         ['V0', 'R', 'C'], ['circuit', 'decay'],
         'RC circuit discharge'),
        
        ('charged_particle', 'q*E*x + q*v0*B*x', 'physics',
         ['q', 'E', 'v0', 'B'], ['electromagnetic'],
         'Charged particle in EM field'),
    ]
    
    _CHEMISTRY_FUNCTIONS = [
        # Kinetics | 动力学
        ('first_order_decay', 'C0*exp(-k*t)', 'chemistry',
         ['C0', 'k'], ['kinetics', 'decay'],
         'First-order reaction kinetics'),
        
        ('second_order_decay', '1/(1/C0 + k*t)', 'chemistry',
         ['C0', 'k'], ['kinetics', 'decay'],
         'Second-order reaction kinetics'),
        
        ('michaelis_menten', 'Vmax*S/(Km + S)', 'chemistry',
         ['Vmax', 'Km'], ['enzyme', 'kinetics'],
         'Michaelis-Menten enzyme kinetics'),
        
        ('arrhenius', 'A*exp(-Ea/(R*T))', 'chemistry',
         ['A', 'Ea', 'R'], ['kinetics', 'temperature'],
         'Arrhenius equation'),
        
        # Adsorption | 吸附
        ('langmuir', 'θ_max*K*C/(1 + K*C)', 'chemistry',
         ['θ_max', 'K'], ['adsorption', 'surface'],
         'Langmuir adsorption isotherm'),
        
        ('freundlich', 'K_f * C**(1/n)', 'chemistry',
         ['K_f', 'n'], ['adsorption', 'surface'],
         'Freundlich adsorption isotherm'),
        
        # Thermodynamics | 热力学
        ('van_der_waals', '(P + a*n**2/V**2)*(V - n*b) = n*R*T', 'chemistry',
         ['a', 'b', 'n', 'R', 'T'], ['equation_of_state'],
         'Van der Waals equation'),
        
        ('clausius_clapeyron', 'ln(P) = -ΔH_vap/(R*T) + C', 'chemistry',
         ['ΔH_vap', 'R', 'C'], ['phase_equilibrium'],
         'Clausius-Clapeyron equation'),
        
        ('gibbs_free', 'ΔG = ΔH - T*ΔS', 'chemistry',
         ['ΔH', 'ΔS'], ['thermodynamics'],
         'Gibbs free energy'),
        
        ('nernst', 'E = E0 - (RT/(nF))*ln(Q)', 'chemistry',
         ['E0', 'R', 'T', 'n', 'F'], ['electrochemistry'],
         'Nernst equation'),
    ]
    
    def __init__(self, 
                 space: FunctionSpace,
                 data_dir: Optional[Union[str, Path]] = None):
        """
        Initialize function library | 初始化函数库
        
        Args:
            space: Function space instance | 函数空间实例
            data_dir: Directory for storing function JSON files | 存储函数 JSON 文件的目录
        """
        self.space = space
        self._entries: Dict[str, FunctionEntry] = {}
        self._logger = logging.getLogger("metathin_sci.FunctionLibrary")
        
        # Set up data directory | 设置数据目录
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._builtin_path = self.data_dir / "builtin_functions.json"
        self._user_path = self.data_dir / "user_functions.json"
        
        # Load functions | 加载函数
        self._init_builtin_functions()
        self._load_user_functions()
        
        self._logger.info(f"Initialized library with {len(self._entries)} functions")
    
    def _init_builtin_functions(self):
        """Initialize built-in functions | 初始化内置函数"""
        from sympy import symbols, sympify
        
        x = symbols('x')
        success_count = 0
        
        # Load from file if exists, otherwise create from definitions
        # 如果文件存在则从文件加载，否则从定义创建
        if self._builtin_path.exists():
            self._load_from_file(self._builtin_path, is_builtin=True)
            return
        
        # Create from definitions | 从定义创建
        all_definitions = (self._MATH_FUNCTIONS + 
                          self._PHYSICS_FUNCTIONS + 
                          self._CHEMISTRY_FUNCTIONS)
        
        for name, expr_str, category, params, tags, desc in all_definitions:
            try:
                expr = sympify(expr_str)
                vector = self.space.from_symbolic(expr, x)
                
                entry = FunctionEntry(
                    name=name,
                    expression=expr_str,
                    vector=vector,
                    category=category,
                    parameters=params,
                    tags=tags,
                    description=desc
                )
                self._entries[name] = entry
                success_count += 1
                
            except Exception as e:
                self._logger.warning(f"Failed to add builtin {name}: {e}")
        
        # Save builtin functions to file | 保存内置函数到文件
        self._save_to_file(self._builtin_path, is_builtin=True)
        self._logger.info(f"Initialized {success_count} builtin functions")
    
    def _load_from_file(self, filepath: Path, is_builtin: bool = False):
        """
        Load functions from JSON file | 从 JSON 文件加载函数
        
        Args:
            filepath: Path to JSON file | JSON 文件路径
            is_builtin: Whether these are builtin functions | 是否为内置函数
        """
        from sympy import sympify
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for name, entry_data in data.items():
                try:
                    # Recreate vector | 重建向量
                    vector_data = entry_data.get('vector', {})
                    coeffs = vector_data.get('coefficients', [])
                    config_data = vector_data.get('config', {})
                    
                    config = VectorSpaceConfig(
                        n_negative=config_data.get('n_negative', 20),
                        n_positive=config_data.get('n_positive', 20),
                        center=config_data.get('center', 0.0)
                    )
                    vector = FunctionVector(np.array(coeffs), config)
                    
                    entry = FunctionEntry(
                        name=name,
                        expression=entry_data.get('expression', ''),
                        vector=vector,
                        category=entry_data.get('category', 'builtin' if is_builtin else 'user'),
                        parameters=entry_data.get('parameters', []),
                        tags=entry_data.get('tags', []),
                        description=entry_data.get('description', ''),
                        metadata=entry_data.get('metadata', {})
                    )
                    self._entries[name] = entry
                    
                except Exception as e:
                    self._logger.warning(f"Failed to load function {name}: {e}")
            
            self._logger.info(f"Loaded {len(data)} functions from {filepath}")
            
        except Exception as e:
            self._logger.warning(f"Failed to load from {filepath}: {e}")
    
    def _save_to_file(self, filepath: Path, is_builtin: bool = False) -> bool:
        """
        Save functions to JSON file | 保存函数到 JSON 文件
        
        Args:
            filepath: Path to JSON file | JSON 文件路径
            is_builtin: Whether these are builtin functions | 是否为内置函数
            
        Returns:
            bool: Success status | 成功状态
        """
        try:
            data = {}
            for name, entry in self._entries.items():
                if is_builtin and entry.category == 'builtin':
                    data[name] = entry.to_dict()
                elif not is_builtin and entry.category != 'builtin':
                    data[name] = entry.to_dict()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self._logger.info(f"Saved {len(data)} functions to {filepath}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to save to {filepath}: {e}")
            return False
    
    def _load_user_functions(self):
        """Load user functions from file | 从文件加载用户函数"""
        if self._user_path.exists():
            self._load_from_file(self._user_path, is_builtin=False)
    
    def _save_user_functions(self) -> bool:
        """Save user functions to file | 保存用户函数到文件"""
        return self._save_to_file(self._user_path, is_builtin=False)
    
    # ============================================================
    # Public API | 公共接口
    # ============================================================
    
    def register(self, 
                 name: str, 
                 expr,
                 category: str = "user",
                 parameters: List[str] = None,
                 tags: List[str] = None,
                 description: str = "",
                 **metadata) -> bool:
        """
        Register a user-defined function | 注册用户自定义函数
        
        Args:
            name: Function name (must be unique) | 函数名称（必须唯一）
            expr: Function expression (SymPy expression or callable) | 函数表达式
            category: Function category | 函数类别
            parameters: List of parameter names | 参数名称列表
            tags: Search tags | 搜索标签
            description: Human-readable description | 描述
            **metadata: Additional metadata | 附加元数据
            
        Returns:
            bool: Success status | 成功状态
        """
        from sympy import sympify
        
        if name in self._entries and self._entries[name].category != 'user':
            self._logger.warning(f"Cannot override builtin function: {name}")
            return False
        
        try:
            if callable(expr):
                # Numerical function | 数值函数
                import numpy as np
                x_vals = np.linspace(-5, 5, 100)
                y_vals = [expr(x) for x in x_vals]
                vector = self.space.from_samples(x_vals, y_vals)
                expr_str = name
            else:
                # Symbolic expression | 符号表达式
                expr_sym = sympify(expr)
                vector = self.space.from_symbolic(expr_sym)
                expr_str = str(expr)
            
            entry = FunctionEntry(
                name=name,
                expression=expr_str,
                vector=vector,
                category=category,
                parameters=parameters or [],
                tags=tags or [],
                description=description,
                metadata=metadata
            )
            
            self._entries[name] = entry
            self._save_user_functions()
            
            self._logger.info(f"Registered user function: {name}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to register {name}: {e}")
            return False
    
    def get(self, name: str) -> Optional[FunctionEntry]:
        """Get function by name | 根据名称获取函数"""
        return self._entries.get(name)
    
    def match(self, 
              vector: FunctionVector, 
              threshold: float = 0.9,
              top_k: int = 5,
              categories: List[str] = None,
              tags: List[str] = None) -> List[Tuple[str, float]]:
        """
        Match vector against library | 匹配向量与库中的函数
        
        Args:
            vector: Function vector to match | 待匹配的函数向量
            threshold: Similarity threshold | 相似度阈值
            top_k: Number of top matches to return | 返回的最佳匹配数量
            categories: Filter by categories | 按类别过滤
            tags: Filter by tags | 按标签过滤
            
        Returns:
            List[Tuple[str, float]]: (function_name, similarity) pairs | (函数名, 相似度) 对
        """
        results = []
        
        for name, entry in self._entries.items():
            # Apply filters | 应用过滤器
            if categories and entry.category not in categories:
                continue
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            
            sim = entry.similarity(vector)
            if sim >= threshold:
                results.append((name, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def match_by_name(self, name: str, vector: FunctionVector) -> float:
        """Compute similarity with a specific named function | 计算与特定命名函数的相似度"""
        entry = self.get(name)
        if entry is None:
            return 0.0
        return entry.similarity(vector)
    
    def list_functions(self, 
                       category: str = None, 
                       tags: List[str] = None) -> List[Dict[str, Any]]:
        """
        List all functions with their information | 列出所有函数及其信息
        
        Args:
            category: Filter by category | 按类别过滤
            tags: Filter by tags | 按标签过滤
            
        Returns:
            List[Dict]: Function information list | 函数信息列表
        """
        result = []
        for name, entry in self._entries.items():
            if category and entry.category != category:
                continue
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            
            result.append({
                'name': entry.name,
                'expression': entry.expression,
                'category': entry.category,
                'parameters': entry.parameters,
                'tags': entry.tags,
                'description': entry.description
            })
        
        return result
    
    def delete_user_function(self, name: str) -> bool:
        """
        Delete a user-defined function | 删除用户自定义函数
        
        Args:
            name: Function name | 函数名称
            
        Returns:
            bool: Success status | 成功状态
        """
        if name not in self._entries:
            return False
        
        if self._entries[name].category == 'builtin':
            self._logger.warning(f"Cannot delete builtin function: {name}")
            return False
        
        del self._entries[name]
        self._save_user_functions()
        self._logger.info(f"Deleted user function: {name}")
        return True
    
    def clear_user_functions(self) -> bool:
        """Clear all user-defined functions | 清空所有用户自定义函数"""
        to_delete = [name for name, entry in self._entries.items() 
                    if entry.category != 'builtin']
        
        for name in to_delete:
            del self._entries[name]
        
        self._save_user_functions()
        self._logger.info(f"Cleared {len(to_delete)} user functions")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics | 获取库统计信息"""
        categories = {}
        for entry in self._entries.values():
            categories[entry.category] = categories.get(entry.category, 0) + 1
        
        return {
            'total_functions': len(self._entries),
            'builtin_count': sum(1 for e in self._entries.values() if e.category == 'builtin'),
            'user_count': sum(1 for e in self._entries.values() if e.category == 'user'),
            'categories': categories,
            'data_directory': str(self.data_dir)
        }
    
    def reload(self) -> bool:
        """Reload functions from files | 从文件重新加载函数"""
        self._entries.clear()
        self._init_builtin_functions()
        self._load_user_functions()
        self._logger.info(f"Reloaded {len(self._entries)} functions")
        return True
    
    def __len__(self) -> int:
        return len(self._entries)
    
    def __contains__(self, name: str) -> bool:
        return name in self._entries
    
    def __iter__(self):
        return iter(self._entries.values())