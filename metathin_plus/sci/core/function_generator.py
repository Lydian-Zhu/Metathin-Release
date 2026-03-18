"""
Function Generator - Batch Generation of Labeled Function Data
======================================================================

Generates synthetic datasets from various mathematical function families with controlled
parameters and noise levels. Essential for training and testing scientific discovery algorithms.

Design Philosophy:
    - Comprehensive coverage: Supports multiple function families
    - Controlled generation: Precise control over parameters and noise
    - Reproducible: Random seed control ensures consistent generation
    - Scalable: Efficient batch generation for large datasets
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
import logging
from enum import Enum
import json
import hashlib


# ============================================================
# Function Type Enumeration
# ============================================================

class FunctionType(Enum):
    """Function type enumeration."""
    LINEAR = "linear"                # Linear: a*x + b
    QUADRATIC = "quadratic"          # Quadratic: a*x² + b*x + c
    CUBIC = "cubic"                  # Cubic: a*x³ + b*x² + c*x + d
    POLYNOMIAL = "polynomial"        # General polynomial
    POWER = "power"                  # Power law: a * x^b
    SIN = "sin"                      # Sine: A*sin(ωx + φ)
    COS = "cos"                      # Cosine: A*cos(ωx + φ)
    TAN = "tan"                      # Tangent: A*tan(ωx + φ)
    EXP = "exp"                      # Exponential: A*e^(αx)
    LOG = "log"                      # Logarithm: A*ln(x + B)
    SIN_LINEAR = "sin_linear"        # Sine + linear: A*sin(ωx) + a*x + b
    EXP_SIN = "exp_sin"              # Exponential sine: A*e^(αx)*sin(ωx + φ)
    RATIONAL = "rational"            # Rational: (a*x + b)/(c*x + d)
    PIECEWISE = "piecewise"          # Piecewise function
    CUSTOM = "custom"                 # Custom user-defined function


@dataclass
class FunctionTemplate:
    """
    Function template data class.
    
    Defines a family of functions with parameter ranges and evaluation function.
    
    Attributes:
        name: Template name
        type: Function type
        expr: Mathematical expression string
        param_ranges: Dictionary mapping parameter names to (min, max) ranges
        func: Callable that evaluates the function
        description: Human-readable description
    """
    name: str
    type: FunctionType
    expr: str
    param_ranges: Dict[str, Tuple[float, float]]
    func: Callable
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Convert template to dictionary (for serialization)."""
        return {
            'name': self.name,
            'type': self.type.value,
            'expr': self.expr,
            'param_ranges': self.param_ranges,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FunctionTemplate':
        """Create template from dictionary (for deserialization)."""
        data['type'] = FunctionType(data['type'])
        return cls(
            name=data['name'],
            type=data['type'],
            expr=data['expr'],
            param_ranges=data['param_ranges'],
            func=None,  # Function must be redefined after loading
            description=data.get('description', '')
        )


@dataclass
class FunctionSample:
    """
    Function sample data class.
    
    Contains a single generated function sample with all metadata.
    
    Attributes:
        template_name: Name of the template used
        parameters: Dictionary of actual parameters used
        x: Input domain values
        y: Noisy function values
        clean_y: Clean (noise-free) function values
        noise_level: Amount of noise added
        metadata: Additional metadata (x_range, n_points, etc.)
    """
    template_name: str
    parameters: Dict[str, float]
    x: np.ndarray
    y: np.ndarray
    clean_y: np.ndarray
    noise_level: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def compute_snr(self) -> float:
        """Compute signal-to-noise ratio."""
        signal_power = np.var(self.clean_y)
        noise_power = np.var(self.y - self.clean_y)
        if noise_power < 1e-10:
            return float('inf')
        return signal_power / noise_power
    
    def get_hash(self) -> str:
        """Generate unique hash for this sample."""
        content = f"{self.template_name}_{self.parameters}_{self.x[0]}_{self.x[-1]}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


class FunctionGenerator:
    """
    Function Generator - Creates synthetic function datasets.
    
    Generates labeled data from various function families with controlled parameters
    and noise levels. Supports both single sample generation and batch generation.
    
    Parameters:
        x_range: Default domain range (min, max)
        n_points: Default number of points per sample
        noise_level: Default noise level (fraction of signal std)
        seed: Random seed for reproducibility
    
    Example:
        >>> generator = FunctionGenerator(x_range=(-5, 5), n_points=100, seed=42)
        >>> 
        >>> # Generate single sample
        >>> sample = generator.generate_one("sin")
        >>> 
        >>> # Generate batch of 1000 samples
        >>> X_list, y_list, labels = generator.generate_batch(1000)
        >>> 
        >>> # Generate with custom weights
        >>> weights = {'sin': 0.5, 'linear': 0.3, 'exp': 0.2}
        >>> X_list, y_list, labels = generator.generate_batch(1000, template_weights=weights)
    """
    
    def __init__(self,
                 x_range: Tuple[float, float] = (-10, 10),
                 n_points: int = 100,
                 noise_level: float = 0.01,
                 seed: Optional[int] = None):
        """
        Initialize function generator.
        
        Args:
            x_range: Default domain range (min, max)
            n_points: Default number of points per sample
            noise_level: Default noise level (fraction of signal std)
            seed: Random seed for reproducibility
        """
        self.x_range = x_range
        self.n_points = n_points
        self.default_noise = noise_level
        self.rng = np.random.RandomState(seed)
        self.logger = logging.getLogger("metathin_sci.core.FunctionGenerator")
        
        # Initialize templates
        self.templates: Dict[str, FunctionTemplate] = {}
        self._init_default_templates()
        
        self.logger.info(f"Function generator initialized with {len(self.templates)} templates")
    
    def _init_default_templates(self):
        """Initialize default function templates."""
        
        # Linear function
        self.add_template(FunctionTemplate(
            name="linear",
            type=FunctionType.LINEAR,
            expr="a*x + b",
            param_ranges={'a': (0.1, 5.0), 'b': (-5.0, 5.0)},
            func=lambda x, a, b: a * x + b,
            description="Linear function y = a*x + b"
        ))
        
        # Quadratic function
        self.add_template(FunctionTemplate(
            name="quadratic",
            type=FunctionType.QUADRATIC,
            expr="a*x^2 + b*x + c",
            param_ranges={
                'a': (0.1, 3.0),
                'b': (-3.0, 3.0),
                'c': (-3.0, 3.0)
            },
            func=lambda x, a, b, c: a * x**2 + b * x + c,
            description="Quadratic function y = a*x² + b*x + c"
        ))
        
        # Cubic function
        self.add_template(FunctionTemplate(
            name="cubic",
            type=FunctionType.CUBIC,
            expr="a*x^3 + b*x^2 + c*x + d",
            param_ranges={
                'a': (0.1, 2.0),
                'b': (-2.0, 2.0),
                'c': (-2.0, 2.0),
                'd': (-2.0, 2.0)
            },
            func=lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
            description="Cubic function y = a*x³ + b*x² + c*x + d"
        ))
        
        # Power function
        self.add_template(FunctionTemplate(
            name="power",
            type=FunctionType.POWER,
            expr="a * x^b",
            param_ranges={'a': (0.1, 5.0), 'b': (0.1, 3.0)},
            func=lambda x, a, b: a * np.power(np.abs(x) + 1e-8, b),
            description="Power function y = a * x^b"
        ))
        
        # Sine function
        self.add_template(FunctionTemplate(
            name="sin",
            type=FunctionType.SIN,
            expr="A*sin(ω*x + φ)",
            param_ranges={
                'A': (0.1, 5.0),
                'ω': (0.1, 5.0),
                'φ': (0, 2*np.pi)
            },
            func=lambda x, A, ω, φ: A * np.sin(ω * x + φ),
            description="Sine function y = A·sin(ωx + φ)"
        ))
        
        # Cosine function
        self.add_template(FunctionTemplate(
            name="cos",
            type=FunctionType.COS,
            expr="A*cos(ω*x + φ)",
            param_ranges={
                'A': (0.1, 5.0),
                'ω': (0.1, 5.0),
                'φ': (0, 2*np.pi)
            },
            func=lambda x, A, ω, φ: A * np.cos(ω * x + φ),
            description="Cosine function y = A·cos(ωx + φ)"
        ))
        
        # Exponential function
        self.add_template(FunctionTemplate(
            name="exp",
            type=FunctionType.EXP,
            expr="A*exp(α*x)",
            param_ranges={'A': (0.1, 5.0), 'α': (-2.0, 2.0)},
            func=lambda x, A, α: A * np.exp(α * x),
            description="Exponential function y = A·e^(αx)"
        ))
        
        # Logarithmic function
        self.add_template(FunctionTemplate(
            name="log",
            type=FunctionType.LOG,
            expr="A*log(x + B)",
            param_ranges={'A': (0.1, 5.0), 'B': (0.1, 5.0)},
            func=lambda x, A, B: A * np.log(np.abs(x) + B + 1e-8),
            description="Logarithmic function y = A·ln(x + B)"
        ))
        
        # Sine + linear
        self.add_template(FunctionTemplate(
            name="sin_linear",
            type=FunctionType.SIN_LINEAR,
            expr="A*sin(ω*x) + a*x + b",
            param_ranges={
                'A': (0.1, 3.0),
                'ω': (0.1, 3.0),
                'a': (-1.0, 1.0),
                'b': (-1.0, 1.0)
            },
            func=lambda x, A, ω, a, b: A * np.sin(ω * x) + a * x + b,
            description="Sine + linear y = A·sin(ωx) + a·x + b"
        ))
        
        # Exponential * sine
        self.add_template(FunctionTemplate(
            name="exp_sin",
            type=FunctionType.EXP_SIN,
            expr="A*exp(α*x)*sin(ω*x + φ)",
            param_ranges={
                'A': (0.1, 3.0),
                'α': (-1.0, 1.0),
                'ω': (0.5, 4.0),
                'φ': (0, 2*np.pi)
            },
            func=lambda x, A, α, ω, φ: A * np.exp(α * x) * np.sin(ω * x + φ),
            description="Exponentially damped sine y = A·e^(αx)·sin(ωx + φ)"
        ))
        
        # Rational function
        self.add_template(FunctionTemplate(
            name="rational",
            type=FunctionType.RATIONAL,
            expr="(a*x + b)/(c*x + d)",
            param_ranges={
                'a': (-5.0, 5.0),
                'b': (-5.0, 5.0),
                'c': (0.1, 3.0),
                'd': (0.1, 3.0)
            },
            func=lambda x, a, b, c, d: (a * x + b) / (c * x + d + 1e-8),
            description="Rational function y = (a·x + b)/(c·x + d)"
        ))
        
        self.logger.debug(f"Added {len(self.templates)} default templates")
    
    def add_template(self, template: FunctionTemplate) -> None:
        """Add a function template."""
        self.templates[template.name] = template
        self.logger.debug(f"Added template: {template.name}")
    
    def remove_template(self, name: str) -> bool:
        """Remove a function template by name."""
        if name in self.templates:
            del self.templates[name]
            return True
        return False
    
    def get_template_names(self) -> List[str]:
        """Get list of available template names."""
        return list(self.templates.keys())
    
    def _sample_parameters(self, template: FunctionTemplate) -> Dict[str, float]:
        """
        Sample parameters uniformly from ranges.
        
        Uses log-uniform sampling for parameters with large ranges to ensure
        better coverage of the parameter space.
        
        Args:
            template: Function template
            
        Returns:
            Dict[str, float]: Sampled parameters
        """
        params = {}
        for param_name, (min_val, max_val) in template.param_ranges.items():
            # Fix division by zero error
            if min_val <= 0:
                # If min <= 0, use uniform distribution
                params[param_name] = self.rng.uniform(min_val, max_val)
            elif max_val / min_val > 100:
                # Log-uniform sampling for large ranges
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
        Generate a single function sample.

        Args:
            template_name: Name of template to use, None for random
            x_range: Domain range, uses default if None
            n_points: Number of points, uses default if None
            noise_level: Noise level, uses default if None
            fixed_params: Fixed parameters (no randomization)

        Returns:
            FunctionSample: Generated sample
        """
        if template_name is None:
            template_name = self.rng.choice(list(self.templates.keys()))
        template = self.templates[template_name]
        
        if fixed_params:
            params = fixed_params
        else:
            params = self._sample_parameters(template)
        
        x_min, x_max = x_range or self.x_range
        x = np.linspace(x_min, x_max, n_points or self.n_points)
        
        clean_y = template.func(x, **params)
        
        noise_std = (noise_level or self.default_noise) * np.std(clean_y)
        noise = self.rng.normal(0, noise_std, len(x))
        y = clean_y + noise
        
        sample = FunctionSample(
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
        
        return sample
    
    def generate_batch(self,
                       n_samples: int,
                       template_weights: Optional[Dict[str, float]] = None,
                       x_range: Optional[Tuple[float, float]] = None,
                       n_points: Optional[int] = None,
                       noise_level: Optional[float] = None,
                       return_type: str = 'all') -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """
        Generate a batch of function samples.

        Args:
            n_samples: Number of samples to generate
            template_weights: Weight for each template, None for uniform
            x_range: Domain range, uses default if None
            n_points: Number of points, uses default if None
            noise_level: Noise level, uses default if None
            return_type: 'all' for (X, y, labels), 'samples' for FunctionSample list

        Returns:
            If return_type == 'all':
                Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]: (X_list, y_list, labels)
            If return_type == 'samples':
                List[FunctionSample]: List of generated samples
        """
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
                self.logger.info(f"Generated {i + 1}/{n_samples} samples")
        
        self.logger.info(f"Batch generation complete: {n_samples} samples")
        
        if return_type == 'samples':
            return samples
        return X_list, y_list, labels_list
    
    def generate_with_variations(self,
                                 base_template: str,
                                 n_variations: int,
                                 param_variations: Dict[str, List[float]],
                                 **kwargs) -> List[FunctionSample]:
        """
        Generate a family of functions with parameter variations.

        Args:
            base_template: Base template name
            n_variations: Number of variations
            param_variations: Dictionary mapping parameter names to value lists
            **kwargs: Additional arguments passed to generate_one

        Returns:
            List[FunctionSample]: Generated samples
        """
        samples = []
        
        for i in range(n_variations):
            params = {}
            for param_name, values in param_variations.items():
                if len(values) == n_variations:
                    params[param_name] = values[i]
                else:
                    params[param_name] = self.rng.choice(values)
            
            sample = self.generate_one(
                template_name=base_template,
                fixed_params=params,
                **kwargs
            )
            samples.append(sample)
        
        return samples
    
    def save_templates(self, filename: str) -> bool:
        """
        Save templates to JSON file.

        Args:
            filename: Output filename

        Returns:
            bool: Success status
        """
        try:
            data = {
                name: template.to_dict()
                for name, template in self.templates.items()
            }
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save templates: {e}")
            return False
    
    def load_templates(self, filename: str) -> bool:
        """
        Load templates from JSON file.

        Note: Loaded templates will have func=None and must be redefined.

        Args:
            filename: Input filename

        Returns:
            bool: Success status
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for name, template_data in data.items():
                template = FunctionTemplate.from_dict(template_data)
                self.templates[name] = template
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to load templates: {e}")
            return False