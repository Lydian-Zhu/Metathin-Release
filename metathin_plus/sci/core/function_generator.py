# metathin_sci/core/function_generator.py
"""
Function Generator - Generate Labeled Function Data | 函数生成器 - 批量生成带标签的函数数据
=============================================================================================

Generates synthetic time series data from various mathematical functions with controllable noise.
Used for training and testing scientific discovery algorithms.

从各种数学函数生成带可控噪声的合成时间序列数据。
用于训练和测试科学发现算法。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
import logging
from enum import Enum
import json
import hashlib


# ============================================================
# Function Type Enum | 函数类型枚举
# ============================================================

class FunctionType(Enum):
    """Function type enumeration | 函数类型枚举"""
    LINEAR = "linear"                    # Linear | 线性
    QUADRATIC = "quadratic"              # Quadratic | 二次
    CUBIC = "cubic"                      # Cubic | 三次
    POLYNOMIAL = "polynomial"            # Polynomial | 多项式
    POWER = "power"                      # Power law | 幂函数
    SIN = "sin"                          # Sine | 正弦
    COS = "cos"                          # Cosine | 余弦
    TAN = "tan"                          # Tangent | 正切
    EXP = "exp"                          # Exponential | 指数
    LOG = "log"                          # Logarithmic | 对数
    SIN_LINEAR = "sin_linear"            # Sine + linear | 正弦+线性
    EXP_SIN = "exp_sin"                  # Exponential decay sine | 指数衰减正弦
    RATIONAL = "rational"                # Rational function | 有理函数
    PIECEWISE = "piecewise"              # Piecewise | 分段函数
    CUSTOM = "custom"                    # Custom | 自定义


@dataclass
class FunctionTemplate:
    """
    Function template definition | 函数模板定义
    
    Defines a family of functions with parameters that can be randomly sampled.
    
    定义一个函数族，其参数可以随机采样。
    
    Attributes | 属性:
        name: Template name | 模板名称
        type: Function type category | 函数类型类别
        expr: Human-readable expression string | 人类可读的表达式字符串
        param_ranges: Parameter ranges for sampling | 参数采样范围
        func: Callable function implementation | 可调用的函数实现
        description: Optional description | 可选描述
    """
    name: str
    type: FunctionType
    expr: str
    param_ranges: Dict[str, Tuple[float, float]]
    func: Callable
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary | 转换为字典"""
        return {
            'name': self.name,
            'type': self.type.value,
            'expr': self.expr,
            'param_ranges': self.param_ranges,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FunctionTemplate':
        """Create from dictionary | 从字典创建"""
        data['type'] = FunctionType(data['type'])
        return cls(
            name=data['name'],
            type=data['type'],
            expr=data['expr'],
            param_ranges=data['param_ranges'],
            func=None,  # Will be set separately | 需要单独设置
            description=data.get('description', '')
        )


@dataclass
class FunctionSample:
    """
    Function sample data | 函数样本数据
    
    Contains the generated data and metadata for a single function instance.
    
    包含单个函数实例的生成数据和元数据。
    
    Attributes | 属性:
        template_name: Name of the template used | 使用的模板名称
        parameters: Parameter values used | 使用的参数值
        x: Input domain points | 输入域点
        y: Output values (with noise) | 输出值（含噪声）
        clean_y: Output values (without noise) | 输出值（无噪声）
        noise_level: Noise level applied | 应用的噪声水平
        metadata: Additional metadata | 额外元数据
    """
    template_name: str
    parameters: Dict[str, float]
    x: np.ndarray
    y: np.ndarray
    clean_y: np.ndarray
    noise_level: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def compute_snr(self) -> float:
        """
        Compute signal-to-noise ratio | 计算信噪比
        
        Returns | 返回:
            float: SNR value (inf if no noise) | 信噪比值（无噪声时为无穷大）
        """
        signal_power = np.var(self.clean_y)
        noise_power = np.var(self.y - self.clean_y)
        if noise_power < 1e-10:
            return float('inf')
        return signal_power / noise_power
    
    def get_hash(self) -> str:
        """Generate unique hash | 生成唯一哈希"""
        content = f"{self.template_name}_{self.parameters}_{self.x[0]}_{self.x[-1]}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


class FunctionGenerator:
    """
    Function Generator - Creates synthetic function data | 函数生成器 - 创建合成函数数据
    
    Generates time series data from various mathematical functions with controllable noise.
    Supports built-in templates and custom template registration.
    
    从各种数学函数生成带可控噪声的时间序列数据。
    支持内置模板和自定义模板注册。
    
    Example | 示例:
        >>> generator = FunctionGenerator(seed=42)
        >>> # Generate a single sample | 生成单个样本
        >>> sample = generator.generate_one(template_name="sin")
        >>> # Generate a batch | 批量生成
        >>> X, y, labels = generator.generate_batch(n_samples=100)
    """
    
    def __init__(self,
                 x_range: Tuple[float, float] = (-10, 10),
                 n_points: int = 100,
                 noise_level: float = 0.01,
                 seed: Optional[int] = None):
        """
        Initialize function generator | 初始化函数生成器
        
        Args | 参数:
            x_range: Input domain range (min, max) | 输入域范围
            n_points: Number of points per sample | 每个样本的点数
            noise_level: Default noise level (std of signal) | 默认噪声水平（信号标准差比例）
            seed: Random seed for reproducibility | 随机种子，用于可复现性
        """
        self.x_range = x_range
        self.n_points = n_points
        self.default_noise = noise_level
        self.rng = np.random.RandomState(seed)
        self.logger = logging.getLogger("metathin_sci.core.FunctionGenerator")
        
        # Template registry | 模板注册表
        self.templates: Dict[str, FunctionTemplate] = {}
        self._init_default_templates()
        
        self.logger.info(f"FunctionGenerator initialized: {len(self.templates)} templates | 初始化完成，{len(self.templates)} 个模板")
    
    def _init_default_templates(self):
        """Initialize default function templates | 初始化默认函数模板"""
        
        # Linear | 线性函数
        self.add_template(FunctionTemplate(
            name="linear",
            type=FunctionType.LINEAR,
            expr="a*x + b",
            param_ranges={'a': (0.1, 5.0), 'b': (-5.0, 5.0)},
            func=lambda x, a, b: a * x + b,
            description="Linear function y = a*x + b | 线性函数"
        ))
        
        # Quadratic | 二次函数
        self.add_template(FunctionTemplate(
            name="quadratic",
            type=FunctionType.QUADRATIC,
            expr="a*x^2 + b*x + c",
            param_ranges={'a': (0.1, 3.0), 'b': (-3.0, 3.0), 'c': (-3.0, 3.0)},
            func=lambda x, a, b, c: a * x**2 + b * x + c,
            description="Quadratic function y = a*x² + b*x + c | 二次函数"
        ))
        
        # Cubic | 三次函数
        self.add_template(FunctionTemplate(
            name="cubic",
            type=FunctionType.CUBIC,
            expr="a*x^3 + b*x^2 + c*x + d",
            param_ranges={'a': (0.1, 2.0), 'b': (-2.0, 2.0), 'c': (-2.0, 2.0), 'd': (-2.0, 2.0)},
            func=lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
            description="Cubic function y = a*x³ + b*x² + c*x + d | 三次函数"
        ))
        
        # Power | 幂函数
        self.add_template(FunctionTemplate(
            name="power",
            type=FunctionType.POWER,
            expr="a * x^b",
            param_ranges={'a': (0.1, 5.0), 'b': (0.1, 3.0)},
            func=lambda x, a, b: a * np.power(np.abs(x) + 1e-8, b),
            description="Power law y = a * x^b | 幂函数"
        ))
        
        # Sine | 正弦函数
        self.add_template(FunctionTemplate(
            name="sin",
            type=FunctionType.SIN,
            expr="A*sin(ω*x + φ)",
            param_ranges={'A': (0.1, 5.0), 'ω': (0.1, 5.0), 'φ': (0, 2*np.pi)},
            func=lambda x, A, ω, φ: A * np.sin(ω * x + φ),
            description="Sine function y = A·sin(ωx + φ) | 正弦函数"
        ))
        
        # Cosine | 余弦函数
        self.add_template(FunctionTemplate(
            name="cos",
            type=FunctionType.COS,
            expr="A*cos(ω*x + φ)",
            param_ranges={'A': (0.1, 5.0), 'ω': (0.1, 5.0), 'φ': (0, 2*np.pi)},
            func=lambda x, A, ω, φ: A * np.cos(ω * x + φ),
            description="Cosine function y = A·cos(ωx + φ) | 余弦函数"
        ))
        
        # Exponential | 指数函数
        self.add_template(FunctionTemplate(
            name="exp",
            type=FunctionType.EXP,
            expr="A*exp(α*x)",
            param_ranges={'A': (0.1, 5.0), 'α': (-2.0, 2.0)},
            func=lambda x, A, α: A * np.exp(α * x),
            description="Exponential function y = A·e^(αx) | 指数函数"
        ))
        
        # Logarithmic | 对数函数
        self.add_template(FunctionTemplate(
            name="log",
            type=FunctionType.LOG,
            expr="A*log(x + B)",
            param_ranges={'A': (0.1, 5.0), 'B': (0.1, 5.0)},
            func=lambda x, A, B: A * np.log(np.abs(x) + B + 1e-8),
            description="Logarithmic function y = A·ln(x + B) | 对数函数"
        ))
        
        # Sine + Linear | 正弦+线性
        self.add_template(FunctionTemplate(
            name="sin_linear",
            type=FunctionType.SIN_LINEAR,
            expr="A*sin(ω*x) + a*x + b",
            param_ranges={'A': (0.1, 3.0), 'ω': (0.1, 3.0), 'a': (-1.0, 1.0), 'b': (-1.0, 1.0)},
            func=lambda x, A, ω, a, b: A * np.sin(ω * x) + a * x + b,
            description="Sine with linear trend | 正弦加线性趋势"
        ))
        
        # Exponential decay sine | 指数衰减正弦
        self.add_template(FunctionTemplate(
            name="exp_sin",
            type=FunctionType.EXP_SIN,
            expr="A*exp(α*x)*sin(ω*x + φ)",
            param_ranges={'A': (0.1, 3.0), 'α': (-1.0, 1.0), 'ω': (0.5, 4.0), 'φ': (0, 2*np.pi)},
            func=lambda x, A, α, ω, φ: A * np.exp(α * x) * np.sin(ω * x + φ),
            description="Exponentially damped sine | 指数衰减正弦"
        ))
        
        # Rational | 有理函数
        self.add_template(FunctionTemplate(
            name="rational",
            type=FunctionType.RATIONAL,
            expr="(a*x + b)/(c*x + d)",
            param_ranges={'a': (-5.0, 5.0), 'b': (-5.0, 5.0), 'c': (0.1, 3.0), 'd': (0.1, 3.0)},
            func=lambda x, a, b, c, d: (a * x + b) / (c * x + d + 1e-8),
            description="Rational function | 有理函数"
        ))
        
        self.logger.debug(f"Added {len(self.templates)} default templates | 添加了 {len(self.templates)} 个默认模板")
    
    def add_template(self, template: FunctionTemplate) -> None:
        """
        Add a custom function template | 添加自定义函数模板
        
        Args | 参数:
            template: Function template to add | 要添加的函数模板
        """
        self.templates[template.name] = template
        self.logger.debug(f"Added template: {template.name} | 添加模板")
    
    def remove_template(self, name: str) -> bool:
        """
        Remove a template by name | 按名称移除模板
        
        Args | 参数:
            name: Template name | 模板名称
            
        Returns | 返回:
            bool: True if removed, False if not found | 成功返回True，未找到返回False
        """
        if name in self.templates:
            del self.templates[name]
            return True
        return False
    
    def get_template_names(self) -> List[str]:
        """Get all template names | 获取所有模板名称"""
        return list(self.templates.keys())
    
    def _sample_parameters(self, template: FunctionTemplate) -> Dict[str, float]:
        """
        Sample parameters from ranges | 从参数范围内随机采样参数
        
        Args | 参数:
            template: Function template | 函数模板
            
        Returns | 返回:
            Dict: Parameter values | 参数值字典
        """
        params = {}
        for param_name, (min_val, max_val) in template.param_ranges.items():
            # Use log-uniform sampling for wide ranges | 对宽范围使用对数均匀采样
            if min_val <= 0:
                params[param_name] = self.rng.uniform(min_val, max_val)
            elif max_val / min_val > 100:
                log_min = np.log(min_val)
                log_max = np.log(max_val)
                params[param_name] = np.exp(self.rng.uniform(log_min, log_max))
            else:
                params[param_name] = self.rng.uniform(min_val, max_val)
        return params
    
    def generate_one(self,
                     template_name: Optional[str] = None,
                     x_range: Optional[Tuple[float, float]] = None,
                     n_points: Optional[int] = None,
                     noise_level: Optional[float] = None,
                     fixed_params: Optional[Dict[str, float]] = None) -> FunctionSample:
        """
        Generate a single function sample | 生成一个函数样本
        
        Args | 参数:
            template_name: Template name (random if None) | 模板名称
            x_range: Domain range (uses default if None) | 域范围
            n_points: Number of points (uses default if None) | 点数
            noise_level: Noise level (uses default if None) | 噪声水平
            fixed_params: Fixed parameters (random sampling if None) | 固定参数
            
        Returns | 返回:
            FunctionSample: Generated sample | 生成的样本
        """
        # Select template | 选择模板
        if template_name is None:
            template_name = self.rng.choice(list(self.templates.keys()))
        template = self.templates[template_name]
        
        # Sample parameters | 采样参数
        if fixed_params:
            params = fixed_params
        else:
            params = self._sample_parameters(template)
        
        # Generate domain and values | 生成域和值
        x_min, x_max = x_range or self.x_range
        x = np.linspace(x_min, x_max, n_points or self.n_points)
        clean_y = template.func(x, **params)
        
        # Add noise | 添加噪声
        noise_std = (noise_level or self.default_noise) * np.std(clean_y)
        noise = self.rng.normal(0, noise_std, len(x))
        y = clean_y + noise
        
        return FunctionSample(
            template_name=template_name,
            parameters=params,
            x=x,
            y=y,
            clean_y=clean_y,
            noise_level=noise_level or self.default_noise,
            metadata={
                'template': template.to_dict(),
                'x_range': (x_min, x_max),
                'n_points': len(x)
            }
        )
    
    def generate_batch(self,
                       n_samples: int,
                       template_weights: Optional[Dict[str, float]] = None,
                       x_range: Optional[Tuple[float, float]] = None,
                       n_points: Optional[int] = None,
                       noise_level: Optional[float] = None,
                       return_type: str = 'all') -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """
        Generate a batch of function samples | 批量生成函数样本
        
        Args | 参数:
            n_samples: Number of samples to generate | 生成的样本数量
            template_weights: Template sampling weights | 模板采样权重
            x_range: Domain range | 域范围
            n_points: Points per sample | 每个样本的点数
            noise_level: Noise level | 噪声水平
            return_type: 'all', 'samples', or 'arrays' | 返回类型
            
        Returns | 返回:
            Tuple of (x_list, y_list, labels_list) or samples list | 元组或样本列表
        """
        # Build sampling distribution | 构建采样分布
        if template_weights is None:
            template_names = list(self.templates.keys())
            probs = [1.0 / len(template_names)] * len(template_names)
        else:
            template_names = list(template_weights.keys())
            probs = [template_weights[name] for name in template_names]
            probs = np.array(probs) / np.sum(probs)
        
        X_list = []
        y_list = []
        labels_list = []
        samples = []
        
        for i in range(n_samples):
            template_name = self.rng.choice(template_names, p=probs)
            
            sample = self.generate_one(
                template_name=template_name,
                x_range=x_range,
                n_points=n_points,
                noise_level=noise_level
            )
            
            X_list.append(sample.x)
            y_list.append(sample.y)
            labels_list.append({
                'name': sample.template_name,
                'expr': self.templates[sample.template_name].expr,
                'params': sample.parameters,
                'snr': sample.compute_snr(),
                'hash': sample.get_hash()
            })
            samples.append(sample)
            
            if (i + 1) % 1000 == 0:
                self.logger.info(f"Generated {i + 1}/{n_samples} samples | 已生成")
        
        self.logger.info(f"Batch generation complete: {n_samples} samples | 批量生成完成")
        
        if return_type == 'samples':
            return samples
        return X_list, y_list, labels_list
    
    def save_templates(self, filename: str) -> bool:
        """
        Save templates to file | 保存模板到文件
        
        Args | 参数:
            filename: Output file path | 输出文件路径
            
        Returns | 返回:
            bool: True if successful | 成功返回True
        """
        try:
            data = {name: template.to_dict() for name, template in self.templates.items()}
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Templates saved to {filename} | 模板已保存")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save templates: {e} | 保存失败")
            return False
    
    def load_templates(self, filename: str) -> bool:
        """
        Load templates from file | 从文件加载模板
        
        Args | 参数:
            filename: Input file path | 输入文件路径
            
        Returns | 返回:
            bool: True if successful | 成功返回True
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for name, template_data in data.items():
                # Note: func must be redefined after loading | 注意：加载后需要重新定义函数
                template = FunctionTemplate.from_dict(template_data)
                self.templates[name] = template
            
            self.logger.info(f"Templates loaded from {filename} | 模板已加载")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load templates: {e} | 加载失败")
            return False