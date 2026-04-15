# metathin_sci/discovery/adaptive_extrapolator.py
"""
Adaptive Symbolic Extrapolator - Core Discovery Algorithm | 自适应符号外推器 - 核心发现算法
===========================================================================================

Real-time learning of symbolic patterns from data streams. Automatically restarts
extrapolation when error exceeds threshold. This is the core engine of the
scientific discovery module.

实时从数据流中学习符号形式的规律，当误差超标时自动重启外推。
这是整个科学发现模块的核心引擎。

Design Philosophy | 设计理念:
    - Adaptive: Automatically adjusts to data complexity | 自适应：根据数据复杂度自动调整
    - Real-time: Online learning, no batch training needed | 实时性：在线学习，无需批量训练
    - Interpretable: Outputs human-readable symbolic forms | 可解释：输出人类可读的符号形式
    - Robust: Insensitive to noise and outliers | 鲁棒性：对噪声和异常值不敏感

Algorithm | 核心算法:
    1. Fit symbolic form using recent N data points | 用最近N步数据拟合符号形式
    2. Set error threshold delta | 设定误差阈值delta
    3. Restart extrapolation when prediction error > delta/2 | 当预测误差 > delta/2 时重启外推
    4. Select better form in overlapping regions | 重叠区域选择误差最小的形式
    5. Record discovered patterns from each phase | 记录每个阶段发现的规律
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import time
import logging
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SymbolicForm:
    """
    Symbolic form data class | 符号形式数据类
    
    Represents a discovered mathematical law with expression, parameters,
    and valid range.
    
    表示一个发现的数学规律，包含表达式、参数和有效范围。
    
    Attributes | 属性:
        expression: Expression string | 表达式字符串
        func: Callable function object | 可调用的函数对象
        params: Parameter dictionary | 参数字典
        error: Fitting error | 拟合误差
        valid_range: Valid range (x_min, x_max) | 有效范围
        created_at: Creation timestamp | 创建时间
        used_count: Usage count | 使用次数
    """
    expression: str
    func: Callable
    params: Dict[str, float]
    error: float
    valid_range: Tuple[float, float]
    created_at: float = field(default_factory=time.time)
    used_count: int = 0
    
    def __post_init__(self):
        """Post-initialization validation | 初始化后验证"""
        if self.error < 0:
            self.error = abs(self.error)
        if self.valid_range[0] > self.valid_range[1]:
            self.valid_range = (self.valid_range[1], self.valid_range[0])
    
    def predict(self, x: float) -> float:
        """
        Predict using this form | 用这个形式预测
        
        Args | 参数:
            x: Input value | 输入值
            
        Returns | 返回:
            float: Prediction value | 预测值
        """
        self.used_count += 1
        try:
            result = self.func(x, **self.params)
            if np.isnan(result) or np.isinf(result):
                return 0.0
            return float(result)
        except Exception as e:
            return 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary | 转换为字典"""
        return {
            'expression': self.expression,
            'params': self.params.copy(),
            'error': self.error,
            'range': list(self.valid_range),
            'created_at': self.created_at,
            'used_count': self.used_count
        }
    
    def __repr__(self) -> str:
        return f"SymbolicForm(expr='{self.expression}', err={self.error:.4f}, range={self.valid_range})"


class SymbolicLibrary:
    """
    Symbolic fragment library - Predefined function forms | 符号片段库 - 预定义的基本函数形式
    
    Contains various common function forms for fitting data.
    
    包含各种常见的函数形式，用于拟合数据。
    """
    
    def __init__(self):
        self.forms = self._init_forms()
        self._logger = logging.getLogger("metathin_sci.discovery.SymbolicLibrary")
    
    def _init_forms(self) -> List[Dict]:
        """Initialize function form library | 初始化函数形式库"""
        return [
            {
                'name': 'linear',
                'expr': 'a*x + b',
                'func': lambda x, a, b: a * x + b,
                'params': ['a', 'b'],
                'p0': [1.0, 0.0]
            },
            {
                'name': 'quadratic',
                'expr': 'a*x**2 + b*x + c',
                'func': lambda x, a, b, c: a*x**2 + b*x + c,
                'params': ['a', 'b', 'c'],
                'p0': [1.0, 0.0, 0.0]
            },
            {
                'name': 'cubic',
                'expr': 'a*x**3 + b*x**2 + c*x + d',
                'func': lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d,
                'params': ['a', 'b', 'c', 'd'],
                'p0': [1.0, 0.0, 0.0, 0.0]
            },
            {
                'name': 'sin',
                'expr': 'A*sin(ω*x + φ)',
                'func': lambda x, A, ω, φ: A * np.sin(ω * x + φ),
                'params': ['A', 'ω', 'φ'],
                'p0': [1.0, 1.0, 0.0]
            },
            {
                'name': 'cos',
                'expr': 'A*cos(ω*x + φ)',
                'func': lambda x, A, ω, φ: A * np.cos(ω * x + φ),
                'params': ['A', 'ω', 'φ'],
                'p0': [1.0, 1.0, 0.0]
            },
            {
                'name': 'exp',
                'expr': 'A*exp(α*x)',
                'func': lambda x, A, α: A * np.exp(α * x),
                'params': ['A', 'α'],
                'p0': [1.0, 0.1]
            },
            {
                'name': 'log',
                'expr': 'A*log(x + B)',
                'func': lambda x, A, B: A * np.log(np.abs(x) + B + 1e-10),
                'params': ['A', 'B'],
                'p0': [1.0, 1.0]
            },
            {
                'name': 'power',
                'expr': 'A*x**b',
                'func': lambda x, A, b: A * np.power(np.abs(x) + 1e-10, b),
                'params': ['A', 'b'],
                'p0': [1.0, 1.0]
            },
            {
                'name': 'sin_linear',
                'expr': 'A*sin(ω*x) + a*x + b',
                'func': lambda x, A, ω, a, b: A * np.sin(ω * x) + a * x + b,
                'params': ['A', 'ω', 'a', 'b'],
                'p0': [1.0, 1.0, 0.0, 0.0]
            },
            {
                'name': 'exp_sin',
                'expr': 'A*exp(α*x)*sin(ω*x + φ)',
                'func': lambda x, A, α, ω, φ: A * np.exp(α * x) * np.sin(ω * x + φ),
                'params': ['A', 'α', 'ω', 'φ'],
                'p0': [1.0, 0.0, 1.0, 0.0]
            },
            {
                'name': 'rational',
                'expr': '(a*x + b)/(c*x + d)',
                'func': lambda x, a, b, c, d: (a*x + b) / (c*x + d + 1e-10),
                'params': ['a', 'b', 'c', 'd'],
                'p0': [1.0, 0.0, 1.0, 1.0]
            },
        ]
    
    def get_all_forms(self) -> List[Dict]:
        """Get all function forms | 获取所有函数形式"""
        return self.forms
    
    def get_form_names(self) -> List[str]:
        """Get all form names | 获取所有形式名称"""
        return [f['name'] for f in self.forms]
    
    def get_form_by_name(self, name: str) -> Optional[Dict]:
        """
        Get function form by name | 根据名称获取函数形式
        
        Args | 参数:
            name: Form name | 形式名称
            
        Returns | 返回:
            Optional[Dict]: Form definition or None | 形式定义或None
        """
        for f in self.forms:
            if f['name'] == name:
                return f
        return None


class AdaptiveExtrapolator:
    """
    Adaptive Symbolic Extrapolator - Core discovery algorithm | 自适应符号外推器 - 核心发现算法
    
    Main features | 主要功能:
        1. Real-time learning of symbolic forms from data stream | 实时从数据流中学习符号形式
        2. Automatic restart when error exceeds threshold | 误差超标时自动重启外推
        3. Optimal selection in overlapping regions | 重叠区域择优保留
        4. Recording of discovered patterns | 记录发现的规律
    
    Parameters | 参数:
        N: Number of steps for extrapolation | 外推用的步数
        delta: Error threshold | 误差阈值
        library: Symbolic fragment library | 符号片段库
        min_samples: Minimum samples for fitting | 最小样本数
        max_forms: Maximum number of forms to keep | 最大保留的形式数
    
    Example | 示例:
        >>> extrapolator = AdaptiveExtrapolator(N=30, delta=0.1)
        >>> 
        >>> # Online prediction | 在线预测
        >>> for x, y_true in data_stream:
        ...     y_pred = extrapolator.predict(x)
        ...     extrapolator.update(x, y_true)
        ... 
        >>> # Get discovered forms | 获取发现的形式
        >>> for form in extrapolator.get_history():
        ...     print(f"Discovered: {form.expression}, error={form.error:.4f}")
    """
    
    def __init__(self,
                 N: int = 50,
                 delta: float = 0.1,
                 library: Optional[SymbolicLibrary] = None,
                 min_samples: int = 10,
                 max_forms: int = 100):
        """
        Initialize adaptive extrapolator | 初始化自适应外推器
        
        Args | 参数:
            N: Number of steps for extrapolation | 外推用的步数
            delta: Error threshold | 误差阈值
            library: Symbolic fragment library | 符号片段库
            min_samples: Minimum samples for fitting | 最小样本数
            max_forms: Maximum number of forms to keep | 最大保留的形式数
        """
        self.N = N
        self.delta = delta
        self.min_samples = min_samples
        self.max_forms = max_forms
        
        # Symbolic library | 符号库
        self.library = library or SymbolicLibrary()
        
        # Logger | 日志
        self._logger = logging.getLogger("metathin_sci.discovery.AdaptiveExtrapolator")
        
        # Data buffers | 数据缓冲区
        self.x_buffer = deque(maxlen=N*2)
        self.y_buffer = deque(maxlen=N*2)
        
        # Current and backup forms | 当前使用的符号形式和备份
        self.current_form: Optional[SymbolicForm] = None
        self.backup_form: Optional[SymbolicForm] = None
        
        # History | 历史记录
        self.forms_history: List[SymbolicForm] = []
        self.predictions: List[Tuple[float, float, float]] = []  # (x, predicted, actual) | (x, 预测值, 实际值)
        self.errors: List[float] = []
        
        # Statistics | 统计信息
        self.restart_count = 0
        self.total_predictions = 0
        self.start_time = time.time()
        
        self._logger.info(f"AdaptiveExtrapolator initialized: N={N}, delta={delta} | 初始化完成")
    
    def _safe_get_last_y(self, default: float = 0.0) -> float:
        """
        Safely get the last y value | 安全地获取最后一个y值
        
        Args | 参数:
            default: Default value | 默认值
            
        Returns | 返回:
            float: Last y value or default | 最后一个y值或默认值
        """
        return self.y_buffer[-1] if self.y_buffer else default
    
    def _safe_get_last_x(self, default: float = 0.0) -> float:
        """
        Safely get the last x value | 安全地获取最后一个x值
        
        Args | 参数:
            default: Default value | 默认值
            
        Returns | 返回:
            float: Last x value or default | 最后一个x值或默认值
        """
        return self.x_buffer[-1] if self.x_buffer else default
    
    def _fit_form(self, x_data: np.ndarray, y_data: np.ndarray) -> Optional[SymbolicForm]:
        """
        Fit data to find the best symbolic form | 拟合数据的最佳符号形式
        
        Args | 参数:
            x_data: X values | x值数组
            y_data: Y values | y值数组
            
        Returns | 返回:
            Optional[SymbolicForm]: Best fitting form or None | 最佳符号形式或None
        """
        if len(x_data) < self.min_samples:
            return None
        
        # Validate data | 验证数据
        if np.any(np.isnan(x_data)) or np.any(np.isnan(y_data)):
            self._logger.debug("Data contains NaN, skipping fit | 数据包含NaN，跳过拟合")
            return None
        
        if np.any(np.isinf(x_data)) or np.any(np.isinf(y_data)):
            self._logger.debug("Data contains Inf, skipping fit | 数据包含Inf，跳过拟合")
            return None
        
        best_form = None
        best_error = float('inf')
        
        for template in self.library.get_all_forms():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, pcov = curve_fit(
                        template['func'],
                        x_data, y_data,
                        p0=template['p0'],
                        maxfev=5000
                    )
                
                # Calculate fitting error | 计算拟合误差
                y_pred = template['func'](x_data, *popt)
                error = np.mean((y_pred - y_data) ** 2)
                
                # Validate error | 验证误差有效性
                if np.isnan(error) or np.isinf(error):
                    continue
                
                # Convert parameters to dict | 参数转字典
                params = {}
                for i, name in enumerate(template['params']):
                    params[name] = float(popt[i])
                
                # Create symbolic form | 创建符号形式
                form = SymbolicForm(
                    expression=template['expr'],
                    func=template['func'],
                    params=params,
                    error=error,
                    valid_range=(float(x_data[0]), float(x_data[-1]))
                )
                
                if error < best_error:
                    best_error = error
                    best_form = form
                    
            except Exception as e:
                continue
        
        return best_form
    
    def add_data_point(self, x: float, y: float):
        """
        Add a data point to buffers | 添加数据点到缓冲区
        
        Args | 参数:
            x: X value | x值
            y: Y value | y值
        """
        self.x_buffer.append(x)
        self.y_buffer.append(y)
    
    def _should_restart(self, predicted: float, actual: float) -> bool:
        """
        Check if extrapolation should restart | 判断是否需要重启外推
        
        Args | 参数:
            predicted: Predicted value | 预测值
            actual: Actual value | 实际值
            
        Returns | 返回:
            bool: True if restart needed | 是否需要重启
        """
        error = abs(predicted - actual)
        return error > self.delta / 2
    
    def _extrapolate_new_form(self) -> Optional[SymbolicForm]:
        """
        Extrapolate new form using recent N steps | 用最近N步数据外推新形式
        
        Returns | 返回:
            Optional[SymbolicForm]: Newly discovered form or None | 新发现的形式或None
        """
        if len(self.x_buffer) < self.N:
            return None
        
        # Take recent N steps | 取最近N步
        x_recent = np.array(list(self.x_buffer)[-self.N:])
        y_recent = np.array(list(self.y_buffer)[-self.N:])
        
        # Fit best form | 拟合最佳形式
        new_form = self._fit_form(x_recent, y_recent)
        
        if new_form:
            self.forms_history.append(new_form)
            # Limit history size | 限制历史大小
            if len(self.forms_history) > self.max_forms:
                self.forms_history = self.forms_history[-self.max_forms:]
            self._logger.debug(f"Discovered new form: {new_form.expression}, error={new_form.error:.6f} | 发现新形式")
        
        return new_form
    
    def _has_overlap(self, form1: SymbolicForm, form2: SymbolicForm) -> bool:
        """
        Check if two forms have overlapping valid ranges | 判断两个形式是否有重叠的有效范围
        
        Args | 参数:
            form1: First form | 第一个形式
            form2: Second form | 第二个形式
            
        Returns | 返回:
            bool: True if overlapping | 是否重叠
        """
        overlap_min = max(form1.valid_range[0], form2.valid_range[0])
        overlap_max = min(form1.valid_range[1], form2.valid_range[1])
        return overlap_max > overlap_min
    
    def _select_better_form(self, form1: SymbolicForm, form2: SymbolicForm) -> SymbolicForm:
        """
        Select the better form in overlapping region | 在重叠区域选择误差小的形式
        
        Args | 参数:
            form1: First form | 第一个形式
            form2: Second form | 第二个形式
            
        Returns | 返回:
            SymbolicForm: Better form | 更好的形式
        """
        if form1.error < form2.error:
            self._logger.debug(f"New form better: {form1.error:.6f} < {form2.error:.6f} | 新形式更好")
            return form1
        else:
            self._logger.debug(f"Old form better: {form2.error:.6f} < {form1.error:.6f} | 旧形式更好")
            return form2
    
    def predict(self, x_next: float) -> float:
        """
        Predict the next value | 预测下一个值
        
        Args | 参数:
            x_next: Next X value | 下一个x值
            
        Returns | 返回:
            float: Predicted value | 预测值
        """
        self.total_predictions += 1
        
        # Default prediction (last value) | 默认预测值（最后一个值）
        default_prediction = self._safe_get_last_y()
        
        # Use current form if available | 如果有当前形式，先用它预测
        if self.current_form:
            try:
                # Check if x is within valid range | 检查x是否在有效范围内
                if (x_next >= self.current_form.valid_range[0] and 
                    x_next <= self.current_form.valid_range[1]):
                    prediction = self.current_form.predict(x_next)
                else:
                    prediction = default_prediction
            except Exception as e:
                self._logger.debug(f"Current form prediction failed: {e} | 当前形式预测失败")
                prediction = default_prediction
        else:
            # No current form, try extrapolation | 没有当前形式，尝试外推
            self.current_form = self._extrapolate_new_form()
            if self.current_form:
                try:
                    prediction = self.current_form.predict(x_next)
                except Exception:
                    prediction = default_prediction
            else:
                prediction = default_prediction
        
        # Validate prediction | 检查预测值有效性
        if np.isnan(prediction) or np.isinf(prediction):
            prediction = default_prediction
        
        self.predictions.append((x_next, prediction, 0.0))  # Actual value updated later | 实际值后面更新
        return prediction
    
    def update(self, x_actual: float, y_actual: float):
        """
        Update with actual value | 用实际值更新
        
        Args | 参数:
            x_actual: Actual X value | 实际x值
            y_actual: Actual Y value | 实际y值
        """
        # Add new data point | 添加新数据点
        self.add_data_point(x_actual, y_actual)
        
        # Update last prediction with actual value | 更新最后一次预测的实际值
        if self.predictions and abs(self.predictions[-1][0] - x_actual) < 1e-10:
            last_x, last_pred, _ = self.predictions[-1]
            self.predictions[-1] = (last_x, last_pred, y_actual)
            
            # Check if restart needed | 检查是否需要重启
            error = abs(last_pred - y_actual)
            if not np.isnan(error) and not np.isinf(error):
                self.errors.append(error)
                # Limit error history | 限制错误历史大小
                if len(self.errors) > 1000:
                    self.errors = self.errors[-1000:]
                
                if self._should_restart(last_pred, y_actual):
                    self._logger.info(f"🔄 Error exceeded threshold, restarting extrapolation (error={error:.4f} > {self.delta/2}) | 误差超标，重启外推")
                    self.restart_count += 1
                    
                    # Backup current form | 备份当前形式
                    self.backup_form = self.current_form
                    
                    # Extrapolate new form | 用最近N步外推新形式
                    new_form = self._extrapolate_new_form()
                    
                    if new_form:
                        # Compare with backup if overlapping | 如果有重叠区域，比较新旧形式
                        if self.backup_form and self._has_overlap(new_form, self.backup_form):
                            self.current_form = self._select_better_form(
                                new_form, self.backup_form
                            )
                        else:
                            self.current_form = new_form
                    else:
                        # Extrapolation failed, continue with old form | 外推失败，继续用旧的
                        self.current_form = self.backup_form
    
    def get_current_form(self) -> Optional[SymbolicForm]:
        """Get current symbolic form | 获取当前使用的符号形式"""
        return self.current_form
    
    def get_history(self) -> List[SymbolicForm]:
        """Get all discovered forms | 获取所有发现的历史形式"""
        return self.forms_history.copy()
    
    def get_recent_errors(self, window: int = 100) -> float:
        """
        Get recent average error | 获取近期平均误差
        
        Args | 参数:
            window: Window size | 窗口大小
            
        Returns | 返回:
            float: Average error | 平均误差
        """
        if not self.errors:
            return 0.0
        
        if window > 0 and len(self.errors) > window:
            recent_errors = self.errors[-window:]
            valid_errors = [e for e in recent_errors if not np.isnan(e) and not np.isinf(e)]
            if valid_errors:
                return float(np.mean(valid_errors))
            return 0.0
        else:
            valid_errors = [e for e in self.errors if not np.isnan(e) and not np.isinf(e)]
            if valid_errors:
                return float(np.mean(valid_errors))
            return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics | 获取统计信息
        
        Returns | 返回:
            Dict: Statistics dictionary | 统计字典
        """
        current_error = self.get_recent_errors(10)
        avg_error = self.get_recent_errors(100)
        
        return {
            'total_predictions': self.total_predictions,
            'restart_count': self.restart_count,
            'forms_discovered': len(self.forms_history),
            'current_form': self.current_form.expression if self.current_form else None,
            'current_error': current_error,
            'avg_error': avg_error,
            'buffer_size': len(self.x_buffer),
            'uptime': time.time() - self.start_time
        }
    
    def reset(self):
        """Reset extrapolator | 重置外推器"""
        self.x_buffer.clear()
        self.y_buffer.clear()
        self.current_form = None
        self.backup_form = None
        self.forms_history.clear()
        self.predictions.clear()
        self.errors.clear()
        self.restart_count = 0
        self.total_predictions = 0
        self.start_time = time.time()
        self._logger.info("Extrapolator reset | 外推器已重置")
    
    def save_state(self, filename: str) -> bool:
        """
        Save state to file | 保存状态到文件
        
        Args | 参数:
            filename: Output filename | 文件名
            
        Returns | 返回:
            bool: True if successful | 成功返回True
        """
        try:
            import pickle
            state = {
                'N': self.N,
                'delta': self.delta,
                'min_samples': self.min_samples,
                'forms_history': [f.to_dict() for f in self.forms_history],
                'restart_count': self.restart_count,
                'total_predictions': self.total_predictions
            }
            with open(filename, 'wb') as f:
                pickle.dump(state, f)
            self._logger.info(f"State saved to {filename} | 状态已保存")
            return True
        except Exception as e:
            self._logger.error(f"Failed to save state: {e} | 保存状态失败")
            return False
    
    def load_state(self, filename: str) -> bool:
        """
        Load state from file | 从文件加载状态
        
        Args | 参数:
            filename: Input filename | 文件名
            
        Returns | 返回:
            bool: True if successful | 成功返回True
        """
        try:
            import pickle
            with open(filename, 'rb') as f:
                state = pickle.load(f)
            
            self.N = state['N']
            self.delta = state['delta']
            self.min_samples = state['min_samples']
            self.restart_count = state['restart_count']
            self.total_predictions = state['total_predictions']
            
            # Recreate forms history | 重新创建forms_history
            self.forms_history = []
            for f_data in state['forms_history']:
                # Get function template by expression | 根据表达式获取函数模板
                expr_name = f_data['expression'].split('(')[0]
                template = self.library.get_form_by_name(expr_name)
                if template:
                    form = SymbolicForm(
                        expression=f_data['expression'],
                        func=template['func'],
                        params=f_data['params'],
                        error=f_data['error'],
                        valid_range=tuple(f_data['range'])
                    )
                    self.forms_history.append(form)
            
            self._logger.info(f"State loaded from {filename} | 状态已加载")
            return True
        except Exception as e:
            self._logger.error(f"Failed to load state: {e} | 加载状态失败")
            return False