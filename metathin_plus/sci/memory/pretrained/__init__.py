"""
Pretrained Function Library - Ready-to-Use Function Knowledge
==========================================================================

Provides pre-trained function libraries for common mathematical patterns across
different domains. These libraries serve as initial knowledge for the scientific
discovery agent, enabling it to recognize and build upon known function families.

Available Libraries:
    - basic: Fundamental mathematical functions (linear, polynomial, trigonometric, etc.)
    - physics: Physical system models (oscillators, chaos, mechanics)
    - chemistry: Chemical reaction kinetics
    - chaos: Chaotic system characteristics (Lorenz, Rossler, Duffing, etc.)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

from ..function_memory import FunctionMemory, FunctionMemoryBank

# Get current file directory
PRETRAINED_DIR = Path(__file__).parent

PRETRAINED_FILES = {
    'basic': PRETRAINED_DIR / 'basic_functions.json',
    'physics': PRETRAINED_DIR / 'physics_functions.json',
    'chemistry': PRETRAINED_DIR / 'chemistry_functions.json',
    'chaos': PRETRAINED_DIR / 'chaos_functions.json',  # New chaos library
}

PRETRAINED_INFO = {
    'basic': {
        'name': 'Basic Mathematical Functions Library',
        'description': 'Contains linear, polynomial, trigonometric, exponential, logarithmic, and other basic functions',
        'size': 100,
        'version': '1.0',
        'tags': ['mathematics', 'basic', 'trigonometric', 'polynomial']
    },
    'physics': {
        'name': 'Physical Systems Functions Library',
        'description': 'Contains pendulum, spring oscillator, chaotic systems, and other physical functions',
        'size': 50,
        'version': '1.0',
        'tags': ['physics', 'oscillation', 'chaos', 'mechanics']
    },
    'chemistry': {
        'name': 'Chemical Reaction Functions Library',
        'description': 'Contains first-order, second-order, oscillating reactions, and other chemical functions',
        'size': 30,
        'version': '1.0',
        'tags': ['chemistry', 'kinetics', 'reaction']
    },
    'chaos': {  # New chaos library info
        'name': 'Chaotic Systems Functions Library',
        'description': 'Contains characteristics of Lorenz, Rossler, Duffing, and other chaotic systems',
        'size': 100,
        'version': '1.0',
        'tags': ['chaos', 'nonlinear', 'dynamics', 'lorenz', 'rossler']
    }
}

# Setup logging
logger = logging.getLogger("metathin_sci.memory.pretrained")


def _load_pretrained_file(filename: Path) -> List[Dict[str, Any]]:
    """
    Load pretrained JSON file.
    
    Args:
        filename: Path to the JSON file
        
    Returns:
        List[Dict]: List of function data, empty list if failed
    """
    if not filename.exists():
        logger.warning(f"Pretrained file does not exist: {filename}")
        return []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            # If there's metadata, print some information
            if 'metadata' in data:
                logger.info(f"Loading {data['metadata'].get('name', 'unknown library')}: "
                           f"{data['metadata'].get('n_functions', 0)} functions")
            
            # Get function list
            if 'functions' in data:
                function_list = data['functions']
            elif isinstance(data.get('data'), list):
                function_list = data['data']
            else:
                # Try to convert entire dict to list
                function_list = [data]
        elif isinstance(data, list):
            function_list = data
        else:
            logger.warning(f"Unknown data format: {filename}")
            return []
        
        # ===== Fix: Filter out unsupported fields =====
        filtered_list = []
        for item in function_list:
            # Create copy to avoid modifying original data
            item_copy = item.copy() if isinstance(item, dict) else item
            
            if isinstance(item_copy, dict):
                # Remove fields not supported by FunctionMemory
                unsupported_fields = ['complexity', 'id']  # Add other unsupported fields here
                for field in unsupported_fields:
                    if field in item_copy:
                        del item_copy[field]
            
            filtered_list.append(item_copy)
        
        return filtered_list
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed {filename}: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load pretrained file {filename}: {e}")
        return []


def _safe_convert_feature_vector(feature_vector: Any, expected_dim: Optional[int] = None) -> np.ndarray:
    """
    Safely convert feature vector to numpy array.
    
    Args:
        feature_vector: Input feature vector (can be list, array, number, etc.)
        expected_dim: Expected dimension, pads or truncates if specified
        
    Returns:
        np.ndarray: Converted float64 array
    """
    try:
        # Handle None
        if feature_vector is None:
            if expected_dim:
                return np.zeros(expected_dim, dtype=np.float64)
            return np.array([0.0], dtype=np.float64)
        
        # If it's already a numpy array
        if isinstance(feature_vector, np.ndarray):
            arr = feature_vector.astype(np.float64)
        # If it's a list or tuple
        elif isinstance(feature_vector, (list, tuple)):
            # Filter non-numeric elements, convert to float
            float_list = []
            for x in feature_vector:
                try:
                    float_list.append(float(x))
                except (TypeError, ValueError):
                    float_list.append(0.0)
            arr = np.array(float_list, dtype=np.float64)
        # If it's a single number
        elif isinstance(feature_vector, (int, float)):
            arr = np.array([float(feature_vector)], dtype=np.float64)
        # If it's a string
        elif isinstance(feature_vector, str):
            # Try to parse string as number
            try:
                val = float(feature_vector)
                arr = np.array([val], dtype=np.float64)
            except ValueError:
                # If failed, use string length as feature
                arr = np.array([float(len(feature_vector))], dtype=np.float64)
        else:
            # Other types, return 0
            arr = np.array([0.0], dtype=np.float64)
        
        # Handle dimension
        if expected_dim is not None:
            if len(arr) > expected_dim:
                # Truncate
                arr = arr[:expected_dim]
            elif len(arr) < expected_dim:
                # Pad
                padding = expected_dim - len(arr)
                arr = np.pad(arr, (0, padding), mode='constant', constant_values=0)
        
        return arr
        
    except Exception as e:
        logger.error(f"Feature vector conversion failed: {e}")
        if expected_dim:
            return np.zeros(expected_dim, dtype=np.float64)
        return np.array([0.0], dtype=np.float64)


def load_basic_functions(expected_dim: Optional[int] = None) -> List[FunctionMemory]:
    """
    Load basic mathematical functions library.
    
    Args:
        expected_dim: Expected feature dimension, adjusts all vectors if specified
        
    Returns:
        List[FunctionMemory]: List of function memory objects
    """
    data = _load_pretrained_file(PRETRAINED_FILES['basic'])
    functions = []
    
    for item in data:
        try:
            # Remove potential id field (will be auto-generated in __init__)
            item_copy = item.copy()
            item_copy.pop('id', None)
            
            # Safely convert feature vector
            if 'feature_vector' in item_copy:
                item_copy['feature_vector'] = _safe_convert_feature_vector(
                    item_copy['feature_vector'], 
                    expected_dim
                )
            else:
                logger.warning(f"Function {item_copy.get('expression', 'unknown')} missing feature_vector field")
                item_copy['feature_vector'] = _safe_convert_feature_vector([], expected_dim)
            
            # Validate required fields
            if 'expression' not in item_copy:
                logger.warning("Function missing expression field, skipping")
                continue
                
            if 'parameters' not in item_copy:
                item_copy['parameters'] = {}
            
            # Create FunctionMemory object
            func = FunctionMemory.from_dict(item_copy)
            functions.append(func)
            
        except Exception as e:
            logger.error(f"Failed to create function memory object: {e}, data: {item.get('expression', 'unknown')}")
            continue
    
    logger.info(f"Loaded basic function library: {len(functions)} functions")
    return functions


def load_physics_functions(expected_dim: Optional[int] = None) -> List[FunctionMemory]:
    """
    Load physical systems functions library.
    
    Args:
        expected_dim: Expected feature dimension
        
    Returns:
        List[FunctionMemory]: List of function memory objects
    """
    data = _load_pretrained_file(PRETRAINED_FILES['physics'])
    functions = []
    
    for item in data:
        try:
            item_copy = item.copy()
            item_copy.pop('id', None)
            
            if 'feature_vector' in item_copy:
                item_copy['feature_vector'] = _safe_convert_feature_vector(
                    item_copy['feature_vector'], 
                    expected_dim
                )
            else:
                item_copy['feature_vector'] = _safe_convert_feature_vector([], expected_dim)
            
            if 'expression' not in item_copy:
                continue
                
            if 'parameters' not in item_copy:
                item_copy['parameters'] = {}
            
            func = FunctionMemory.from_dict(item_copy)
            functions.append(func)
            
        except Exception as e:
            logger.error(f"Failed to create physics function: {e}")
            continue
    
    logger.info(f"Loaded physics function library: {len(functions)} functions")
    return functions


def load_chemistry_functions(expected_dim: Optional[int] = None) -> List[FunctionMemory]:
    """
    Load chemical reactions functions library.
    
    Args:
        expected_dim: Expected feature dimension
        
    Returns:
        List[FunctionMemory]: List of function memory objects
    """
    data = _load_pretrained_file(PRETRAINED_FILES['chemistry'])
    functions = []
    
    for item in data:
        try:
            item_copy = item.copy()
            item_copy.pop('id', None)
            
            if 'feature_vector' in item_copy:
                item_copy['feature_vector'] = _safe_convert_feature_vector(
                    item_copy['feature_vector'], 
                    expected_dim
                )
            else:
                item_copy['feature_vector'] = _safe_convert_feature_vector([], expected_dim)
            
            if 'expression' not in item_copy:
                continue
                
            if 'parameters' not in item_copy:
                item_copy['parameters'] = {}
            
            func = FunctionMemory.from_dict(item_copy)
            functions.append(func)
            
        except Exception as e:
            logger.error(f"Failed to create chemistry function: {e}")
            continue
    
    logger.info(f"Loaded chemistry function library: {len(functions)} functions")
    return functions


def load_chaos_functions(expected_dim: Optional[int] = None) -> List[FunctionMemory]:
    """
    Load chaotic systems functions library.
    
    Args:
        expected_dim: Expected feature dimension
        
    Returns:
        List[FunctionMemory]: List of function memory objects
    """
    data = _load_pretrained_file(PRETRAINED_FILES['chaos'])
    functions = []
    
    for item in data:
        try:
            item_copy = item.copy()
            item_copy.pop('id', None)
            
            if 'feature_vector' in item_copy:
                item_copy['feature_vector'] = _safe_convert_feature_vector(
                    item_copy['feature_vector'], 
                    expected_dim
                )
            else:
                item_copy['feature_vector'] = _safe_convert_feature_vector([], expected_dim)
            
            if 'expression' not in item_copy:
                continue
                
            if 'parameters' not in item_copy:
                item_copy['parameters'] = {}
            
            func = FunctionMemory.from_dict(item_copy)
            functions.append(func)
            
        except Exception as e:
            logger.error(f"Failed to create chaos function: {e}")
            continue
    
    logger.info(f"Loaded chaos function library: {len(functions)} functions")
    return functions


def load_all_pretrained(limits: Optional[Dict[str, int]] = None,
                       expected_dim: Optional[int] = None) -> Dict[str, List[FunctionMemory]]:
    """
    Load all pretrained function libraries.
    
    Args:
        limits: Limits per library, e.g., {'basic': 100, 'physics': 50}
        expected_dim: Expected feature dimension
        
    Returns:
        Dict: Mapping from library name to list of functions
    """
    if limits is None:
        limits = {}
    
    result = {}
    
    # Load basic library
    basic_funcs = load_basic_functions(expected_dim)
    if 'basic' in limits and limits['basic'] > 0:
        basic_funcs = basic_funcs[:min(limits['basic'], len(basic_funcs))]
    result['basic'] = basic_funcs
    
    # Load physics library
    physics_funcs = load_physics_functions(expected_dim)
    if 'physics' in limits and limits['physics'] > 0:
        physics_funcs = physics_funcs[:min(limits['physics'], len(physics_funcs))]
    result['physics'] = physics_funcs
    
    # Load chemistry library
    chemistry_funcs = load_chemistry_functions(expected_dim)
    if 'chemistry' in limits and limits['chemistry'] > 0:
        chemistry_funcs = chemistry_funcs[:min(limits['chemistry'], len(chemistry_funcs))]
    result['chemistry'] = chemistry_funcs
    
    # Load chaos library
    chaos_funcs = load_chaos_functions(expected_dim)
    if 'chaos' in limits and limits['chaos'] > 0:
        chaos_funcs = chaos_funcs[:min(limits['chaos'], len(chaos_funcs))]
    result['chaos'] = chaos_funcs
    
    total = sum(len(funcs) for funcs in result.values())
    logger.info(f"Loaded all pretrained libraries, total {total} functions")
    
    return result


def list_available_pretrained() -> List[Dict[str, Any]]:
    """
    List all available pretrained libraries with their information.
    
    Returns:
        List[Dict]: Information about each library
    """
    info_list = []
    
    for key, info in PRETRAINED_INFO.items():
        file_path = PRETRAINED_FILES.get(key)
        file_exists = file_path.exists() if file_path else False
        
        # If file exists, try to read actual size
        actual_size = info['size']
        if file_exists:
            try:
                data = _load_pretrained_file(file_path)
                actual_size = len(data)
            except:
                pass
        
        info_list.append({
            'key': key,
            'name': info['name'],
            'description': info['description'],
            'size': actual_size,
            'version': info['version'],
            'tags': info['tags'],
            'file_exists': file_exists,
            'file_path': str(file_path) if file_path else None
        })
    
    return info_list


def get_pretrained_info(library_key: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific pretrained library.
    
    Args:
        library_key: Library key ('basic', 'physics', 'chemistry', 'chaos')
        
    Returns:
        Optional[Dict]: Library information, None if not found
    """
    if library_key not in PRETRAINED_INFO:
        return None
    
    info = PRETRAINED_INFO[library_key].copy()
    info['key'] = library_key
    info['file_exists'] = PRETRAINED_FILES[library_key].exists()
    info['file_path'] = str(PRETRAINED_FILES[library_key])
    
    # If file exists, read actual size
    if info['file_exists']:
        try:
            data = _load_pretrained_file(PRETRAINED_FILES[library_key])
            info['size'] = len(data)
        except:
            pass
    
    return info


def create_pretrained_bank(library_keys: Optional[List[str]] = None,
                           limits: Optional[Dict[str, int]] = None,
                           expected_dim: Optional[int] = None) -> FunctionMemoryBank:
    """
    Create a memory bank with pretrained functions.
    
    Args:
        library_keys: List of library keys to load, None for all
        limits: Limits per library
        expected_dim: Expected feature dimension
        
    Returns:
        FunctionMemoryBank: Memory bank containing pretrained functions
    """
    if library_keys is None:
        library_keys = ['basic', 'physics', 'chemistry', 'chaos']
    
    # Create in-memory memory bank (non-persistent)
    bank = FunctionMemoryBank(memory_backend='memory')
    
    # Load all functions
    all_functions = load_all_pretrained(limits, expected_dim)
    
    added_count = 0
    for key in library_keys:
        if key in all_functions:
            funcs = all_functions[key]
            for func in funcs:
                if bank.add(func):
                    added_count += 1
            logger.info(f"Added {len(funcs)} {key} functions to memory bank")
    
    logger.info(f"Pretrained memory bank created with {added_count} functions")
    
    return bank


__all__ = [
    'load_basic_functions',
    'load_physics_functions',
    'load_chemistry_functions',
    'load_chaos_functions',  # New
    'load_all_pretrained',
    'list_available_pretrained',
    'get_pretrained_info',
    'create_pretrained_bank',
    'PRETRAINED_INFO',
    'PRETRAINED_FILES',
]