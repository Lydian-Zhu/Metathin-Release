"""
Adaptive Symbolic Extrapolator - Core Discovery Algorithm
======================================================================

Learns symbolic patterns from data streams in real-time and automatically restarts 
extrapolation when error exceeds threshold. This is the core implementation of the 
discovery engine for the scientific discovery module.

Design Philosophy:
    - Adaptive: Automatically adjusts based on data complexity
    - Real-time: Online learning without batch training
    - Interpretable: Outputs human-readable symbolic forms
    - Robust: Insensitive to noise and outliers

Core Algorithm:
    1. Fit symbolic forms using last N data points
    2. Set error threshold delta
    3. Restart extrapolation when prediction error > delta/2
    4. Select the form with minimum error in overlapping regions
    5. Record discovered patterns for each phase
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
    Symbolic form data class.
    
    Represents a discovered mathematical pattern, containing the expression,
    parameters, and valid range.
    
    Attributes:
        expression: String representation of the expression
        func: Callable function object
        params: Dictionary of parameter values
        error: Fitting error
        valid_range: Valid range (x_min, x_max)
        created_at: Creation timestamp
        used_count: Number of times used for prediction
    """
    expression: str
    func: Callable
    params: Dict[str, float]
    error: float
    valid_range: Tuple[float, float]
    created_at: float = field(default_factory=time.time)
    used_count: int = 0
    
    def __post_init__(self):
        """Post-initialization validation."""
        # Ensure error is non-negative
        if self.error < 0:
            self.error = abs(self.error)
        
        # Ensure range is properly formatted
        if self.valid_range[0] > self.valid_range[1]:
            self.valid_range = (self.valid_range[1], self.valid_range[0])
    
    def predict(self, x: float) -> float:
        """
        Predict using this form.
        
        Args:
            x: Input value
            
        Returns:
            float: Predicted value
        """
        self.used_count += 1
        try:
            result = self.func(x, **self.params)
            # Check result validity
            if np.isnan(result) or np.isinf(result):
                return 0.0
            return float(result)
        except Exception as e:
            return 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
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
    Symbolic pattern library - Predefined basic function forms.
    
    Contains various common function forms for fitting data.
    """
    
    def __init__(self):
        """Initialize the symbolic library with default forms."""
        self.forms = self._init_forms()
    
    def _init_forms(self) -> List[Dict]:
        """Initialize the function form library."""
        return [
            {
                'name': 'linear',
                'expr': 'a*x + b',
                'func': lambda x, a, b: a * x + b,
                'params': ['a', 'b'],
                'p0': [1.0, 0.0]  # Initial guess
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
        """Get all function forms."""
        return self.forms
    
    def get_form_names(self) -> List[str]:
        """Get names of all forms."""
        return [f['name'] for f in self.forms]
    
    def get_form_by_name(self, name: str) -> Optional[Dict]:
        """Get function form by name."""
        for f in self.forms:
            if f['name'] == name:
                return f
        return None


class AdaptiveExtrapolator:
    """
    Adaptive Symbolic Extrapolator - Core Discovery Algorithm.
    
    Main Features:
        1. Real-time learning of symbolic patterns from data streams
        2. Automatic restart when error exceeds threshold
        3. Optimal selection in overlapping regions
        4. Recording of discovered patterns
    
    Parameters:
        N: Number of steps used for extrapolation
        delta: Error threshold
        library: Symbolic pattern library
        min_samples: Minimum number of samples required for fitting
        max_forms: Maximum number of forms to keep in history
    """
    
    def __init__(self,
                 N: int = 50,
                 delta: float = 0.1,
                 library: Optional[SymbolicLibrary] = None,
                 min_samples: int = 10,
                 max_forms: int = 100):
        """
        Initialize adaptive extrapolator.
        
        Args:
            N: Number of steps used for extrapolation
            delta: Error threshold
            library: Symbolic pattern library
            min_samples: Minimum samples required for fitting
            max_forms: Maximum forms to keep in history
        """
        self.N = N
        self.delta = delta
        self.min_samples = min_samples
        self.max_forms = max_forms
        
        # Symbolic library
        self.library = library or SymbolicLibrary()
        
        # Logger
        self.logger = logging.getLogger("metathin_sci.discovery.AdaptiveExtrapolator")
        
        # Data buffers
        self.x_buffer = deque(maxlen=N*2)
        self.y_buffer = deque(maxlen=N*2)
        
        # Current and backup symbolic forms
        self.current_form: Optional[SymbolicForm] = None
        self.backup_form: Optional[SymbolicForm] = None
        
        # History tracking
        self.forms_history: List[SymbolicForm] = []
        self.predictions: List[Tuple[float, float, float]] = []  # (x, predicted, actual)
        self.errors: List[float] = []
        
        # Statistics
        self.restart_count = 0
        self.total_predictions = 0
        self.start_time = time.time()
        
        self.logger.info(f"Adaptive extrapolator initialized: N={N}, delta={delta}")
    
    def _safe_get_last_y(self, default: float = 0.0) -> float:
        """
        Safely get the last y value.
        
        Args:
            default: Default value
            
        Returns:
            float: Last y value, or default if buffer is empty
        """
        return self.y_buffer[-1] if self.y_buffer else default
    
    def _safe_get_last_x(self, default: float = 0.0) -> float:
        """
        Safely get the last x value.
        
        Args:
            default: Default value
            
        Returns:
            float: Last x value, or default if buffer is empty
        """
        return self.x_buffer[-1] if self.x_buffer else default
    
    def _fit_form(self, x_data: np.ndarray, y_data: np.ndarray) -> Optional[SymbolicForm]:
        """
        Fit the best symbolic form to the data.
        
        Args:
            x_data: Array of x values
            y_data: Array of y values
            
        Returns:
            Optional[SymbolicForm]: Best fitting symbolic form, or None if fitting fails
        """
        if len(x_data) < self.min_samples:
            return None
        
        # Check data validity
        if np.any(np.isnan(x_data)) or np.any(np.isnan(y_data)):
            self.logger.debug("Data contains NaN, skipping fit")
            return None
        
        if np.any(np.isinf(x_data)) or np.any(np.isinf(y_data)):
            self.logger.debug("Data contains Inf, skipping fit")
            return None
        
        best_form = None
        best_error = float('inf')
        
        for template in self.library.get_all_forms():
            try:
                # Attempt fitting
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, pcov = curve_fit(
                        template['func'],
                        x_data, y_data,
                        p0=template['p0'],
                        maxfev=5000
                    )
                
                # Calculate fitting error
                y_pred = template['func'](x_data, *popt)
                error = np.mean((y_pred - y_data) ** 2)
                
                # Check error validity
                if np.isnan(error) or np.isinf(error):
                    continue
                
                # Convert parameters to dictionary
                params = {}
                for i, name in enumerate(template['params']):
                    params[name] = float(popt[i])
                
                # Create symbolic form
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
        """Add a data point to the buffer."""
        self.x_buffer.append(x)
        self.y_buffer.append(y)
    
    def _should_restart(self, predicted: float, actual: float) -> bool:
        """Determine if extrapolation should restart."""
        error = abs(predicted - actual)
        return error > self.delta / 2
    
    def _extrapolate_new_form(self) -> Optional[SymbolicForm]:
        """Extrapolate a new form using the most recent N steps."""
        if len(self.x_buffer) < self.N:
            return None
        
        # Take most recent N steps
        x_recent = np.array(list(self.x_buffer)[-self.N:])
        y_recent = np.array(list(self.y_buffer)[-self.N:])
        
        # Fit best form
        new_form = self._fit_form(x_recent, y_recent)
        
        if new_form:
            self.forms_history.append(new_form)
            # Limit history size
            if len(self.forms_history) > self.max_forms:
                self.forms_history = self.forms_history[-self.max_forms:]
            self.logger.debug(f"Discovered new form: {new_form.expression}, error={new_form.error:.6f}")
        
        return new_form
    
    def _has_overlap(self, form1: SymbolicForm, form2: SymbolicForm) -> bool:
        """Check if two forms have overlapping valid ranges."""
        overlap_min = max(form1.valid_range[0], form2.valid_range[0])
        overlap_max = min(form1.valid_range[1], form2.valid_range[1])
        return overlap_max > overlap_min
    
    def _select_better_form(self, form1: SymbolicForm, form2: SymbolicForm) -> SymbolicForm:
        """Select the form with smaller error in overlapping region."""
        if form1.error < form2.error:
            self.logger.debug(f"New form better: {form1.error:.6f} < {form2.error:.6f}")
            return form1
        else:
            self.logger.debug(f"Old form better: {form2.error:.6f} < {form1.error:.6f}")
            return form2
    
    def predict(self, x_next: float) -> float:
        """
        Predict the next value.
        
        Args:
            x_next: Next x value
            
        Returns:
            float: Predicted value
        """
        self.total_predictions += 1
        
        # Default prediction
        default_prediction = self._safe_get_last_y()
        
        # If current form exists, use it for prediction
        if self.current_form:
            try:
                # Check if x is within valid range
                if (x_next >= self.current_form.valid_range[0] and 
                    x_next <= self.current_form.valid_range[1]):
                    prediction = self.current_form.predict(x_next)
                else:
                    # Out of range, use last value
                    prediction = default_prediction
            except Exception as e:
                self.logger.debug(f"Current form prediction failed: {e}")
                prediction = default_prediction
        else:
            # No current form, try to extrapolate
            self.current_form = self._extrapolate_new_form()
            if self.current_form:
                try:
                    prediction = self.current_form.predict(x_next)
                except Exception:
                    prediction = default_prediction
            else:
                prediction = default_prediction
        
        # Check prediction validity
        if np.isnan(prediction) or np.isinf(prediction):
            prediction = default_prediction
        
        self.predictions.append((x_next, prediction, 0.0))  # Actual value updated later
        return prediction
    
    def update(self, x_actual: float, y_actual: float):
        """
        Update with actual value.
        
        Args:
            x_actual: Actual x value
            y_actual: Actual y value
        """
        # Add new data point
        self.add_data_point(x_actual, y_actual)
        
        # Update last prediction with actual value
        if self.predictions and abs(self.predictions[-1][0] - x_actual) < 1e-10:
            last_x, last_pred, _ = self.predictions[-1]
            self.predictions[-1] = (last_x, last_pred, y_actual)
            
            # Check if restart needed
            error = abs(last_pred - y_actual)
            if not np.isnan(error) and not np.isinf(error):
                self.errors.append(error)
                # Limit error history size
                if len(self.errors) > 1000:
                    self.errors = self.errors[-1000:]
                
                if self._should_restart(last_pred, y_actual):
                    self.logger.info(f"🔄 Error exceeded threshold, restarting extrapolation (error={error:.4f} > {self.delta/2})")
                    self.restart_count += 1
                    
                    # Backup current form
                    self.backup_form = self.current_form
                    
                    # Extrapolate new form using recent N steps
                    new_form = self._extrapolate_new_form()
                    
                    if new_form:
                        # If overlap exists, compare old and new forms
                        if self.backup_form and self._has_overlap(new_form, self.backup_form):
                            self.current_form = self._select_better_form(
                                new_form, self.backup_form
                            )
                        else:
                            self.current_form = new_form
                    else:
                        # Extrapolation failed, continue using old form
                        self.current_form = self.backup_form
    
    def get_current_form(self) -> Optional[SymbolicForm]:
        """Get the currently used symbolic form."""
        return self.current_form
    
    def get_history(self) -> List[SymbolicForm]:
        """Get all historically discovered forms."""
        return self.forms_history.copy()
    
    def get_recent_errors(self, window: int = 100) -> float:
        """Get recent average error."""
        if not self.errors:
            return 0.0
        
        if window > 0 and len(self.errors) > window:
            recent_errors = self.errors[-window:]
            # Filter NaN and Inf
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
        """Get statistics."""
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
        """Reset the extrapolator."""
        self.x_buffer.clear()
        self.y_buffer.clear()
        self.current_form = None
        self.backup_form = None
        self.forms_history.clear()
        self.predictions.clear()
        self.errors.clear()
        self.restart_count = 0
        self.total_predictions = 0
        self.logger.info("Extrapolator reset")
    
    def save_state(self, filename: str) -> bool:
        """
        Save state to file.
        
        Args:
            filename: Output filename
            
        Returns:
            bool: Success status
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
            return True
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            return False
    
    def load_state(self, filename: str) -> bool:
        """
        Load state from file.
        
        Args:
            filename: Input filename
            
        Returns:
            bool: Success status
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
            
            # Recreate forms_history (requires recreating function objects)
            self.forms_history = []
            for f_data in state['forms_history']:
                # Get function template based on expression
                template = self.library.get_form_by_name(f_data['expression'].split('(')[0])
                if template:
                    form = SymbolicForm(
                        expression=f_data['expression'],
                        func=template['func'],
                        params=f_data['params'],
                        error=f_data['error'],
                        valid_range=tuple(f_data['range'])
                    )
                    self.forms_history.append(form)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return False


# ============================================================
# Usage Examples
# ============================================================
"""
>>> from metathin_sci.discovery import AdaptiveExtrapolator
>>> import numpy as np
>>> 
>>> # 1. Create extrapolator
>>> extrapolator = AdaptiveExtrapolator(N=30, delta=0.1)
>>> 
>>> # 2. Generate test data (piecewise function)
>>> x_vals = np.linspace(0, 15, 300)
>>> y_vals = []
>>> for x in x_vals:
...     if x < 5:
...         y = 2 * np.sin(0.5 * x)
...     elif x < 10:
...         y = 0.5 * x
...     else:
...         y = 0.1 * x**2
...     y_vals.append(y)
>>> 
>>> # 3. Real-time prediction
>>> for x, y_true in zip(x_vals, y_vals):
...     y_pred = extrapolator.predict(x)
...     extrapolator.update(x, y_true)
... 
>>> # 4. View discovered patterns
>>> for i, form in enumerate(extrapolator.get_history()):
...     print(f"Phase {i+1}: {form.expression}, range={form.valid_range}")
... 
>>> # 5. Get statistics
>>> stats = extrapolator.get_stats()
>>> print(f"Restarts: {stats['restart_count']}")
>>> print(f"Forms discovered: {stats['forms_discovered']}")
"""