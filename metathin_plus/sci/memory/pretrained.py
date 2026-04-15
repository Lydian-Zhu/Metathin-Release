# metathin_sci/memory/pretrained.py
"""
Pretrained Function Library | 预训练函数库
===========================================

Provides ready-to-use function knowledge for scientific discovery.
Includes basic math, physics, chemistry, and chaos function libraries.

提供开箱即用的函数知识，用于科学发现。
包含基础数学、物理、化学和混沌函数库。
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .function_memory import FunctionMemory, FunctionMemoryBank

# Get current directory | 获取当前文件所在目录
PRETRAINED_DIR = Path(__file__).parent

PRETRAINED_FILES = {
    'basic': PRETRAINED_DIR / 'basic_functions.json',
    'physics': PRETRAINED_DIR / 'physics_functions.json',
    'chemistry': PRETRAINED_DIR / 'chemistry_functions.json',
    'chaos': PRETRAINED_DIR / 'chaos_functions.json',
}

PRETRAINED_INFO = {
    'basic': {
        'name': 'Basic Math Functions | 基础数学函数库',
        'description': 'Linear, polynomial, trigonometric, exponential, logarithmic functions | 线性、多项式、三角函数、指数、对数等基础函数',
        'size': 450,
        'version': '1.0',
        'tags': ['mathematics', 'basic', 'trigonometric', 'polynomial']
    },
    'physics': {
        'name': 'Physics Functions | 物理系统函数库',
        'description': 'Pendulum, spring oscillator, damped oscillation | 单摆、弹簧振子、阻尼振荡等物理函数',
        'size': 100,
        'version': '1.0',
        'tags': ['physics', 'oscillation', 'mechanics', 'damped']
    },
    'chemistry': {
        'name': 'Chemistry Functions | 化学反应函数库',
        'description': 'First-order, second-order, oscillating reactions | 一级反应、二级反应、振荡反应等化学函数',
        'size': 45,
        'version': '1.0',
        'tags': ['chemistry', 'kinetics', 'reaction']
    },
    'chaos': {
        'name': 'Chaos System Functions | 混沌系统函数库',
        'description': 'Lorenz, Rossler, Duffing, Van der Pol systems | 洛伦兹、罗斯勒、杜芬、范德波尔等混沌系统',
        'size': 120,
        'version': '1.0',
        'tags': ['chaos', 'nonlinear', 'dynamics', 'attractor']
    }
}

logger = logging.getLogger("metathin_sci.memory.pretrained")


def _load_pretrained_file(filename: Path) -> List[Dict[str, Any]]:
    """
    Load pretrained file | 加载预训练文件
    
    Args | 参数:
        filename: File path | 文件路径
        
    Returns | 返回:
        List[Dict]: Function data list | 函数数据列表
    """
    if not filename.exists():
        logger.warning(f"Pretrained file not found: {filename} | 预训练文件不存在")
        return []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures | 处理不同的JSON结构
        if isinstance(data, dict):
            if 'metadata' in data:
                logger.info(f"Loading {data['metadata'].get('name', 'unknown')}: "
                           f"{data['metadata'].get('n_functions', 0)} functions | 加载")
            
            if 'functions' in data:
                function_list = data['functions']
            elif isinstance(data.get('data'), list):
                function_list = data['data']
            else:
                function_list = [data]
        elif isinstance(data, list):
            function_list = data
        else:
            logger.warning(f"Unknown data format: {filename} | 未知的数据格式")
            return []
        
        # Filter out unsupported fields | 过滤掉不受支持的字段
        filtered_list = []
        for item in function_list:
            item_copy = item.copy() if isinstance(item, dict) else item
            
            if isinstance(item_copy, dict):
                unsupported_fields = ['complexity', 'id']
                for field in unsupported_fields:
                    if field in item_copy:
                        del item_copy[field]
            
            filtered_list.append(item_copy)
        
        return filtered_list
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed {filename}: {e} | JSON解析失败")
        return []
    except Exception as e:
        logger.error(f"Failed to load pretrained file {filename}: {e} | 加载预训练文件失败")
        return []


def _safe_convert_feature_vector(feature_vector: Any, expected_dim: Optional[int] = None) -> np.ndarray:
    """
    Safely convert feature vector to numpy array | 安全地转换特征向量为numpy数组
    
    Args | 参数:
        feature_vector: Input feature vector | 输入的特征向量
        expected_dim: Expected dimension | 期望的维度
        
    Returns | 返回:
        np.ndarray: Converted float64 array | 转换后的float64数组
    """
    try:
        if feature_vector is None:
            if expected_dim:
                return np.zeros(expected_dim, dtype=np.float64)
            return np.array([0.0], dtype=np.float64)
        
        if isinstance(feature_vector, np.ndarray):
            arr = feature_vector.astype(np.float64)
        elif isinstance(feature_vector, (list, tuple)):
            float_list = []
            for x in feature_vector:
                try:
                    float_list.append(float(x))
                except (TypeError, ValueError):
                    float_list.append(0.0)
            arr = np.array(float_list, dtype=np.float64)
        elif isinstance(feature_vector, (int, float)):
            arr = np.array([float(feature_vector)], dtype=np.float64)
        elif isinstance(feature_vector, str):
            try:
                val = float(feature_vector)
                arr = np.array([val], dtype=np.float64)
            except ValueError:
                arr = np.array([float(len(feature_vector))], dtype=np.float64)
        else:
            arr = np.array([0.0], dtype=np.float64)
        
        # Handle dimension | 处理维度
        if expected_dim is not None:
            if len(arr) > expected_dim:
                arr = arr[:expected_dim]
            elif len(arr) < expected_dim:
                padding = expected_dim - len(arr)
                arr = np.pad(arr, (0, padding), mode='constant', constant_values=0)
        
        return arr
        
    except Exception as e:
        logger.error(f"Feature vector conversion failed: {e} | 特征向量转换失败")
        if expected_dim:
            return np.zeros(expected_dim, dtype=np.float64)
        return np.array([0.0], dtype=np.float64)


def load_basic_functions(expected_dim: Optional[int] = None) -> List[FunctionMemory]:
    """
    Load basic math function library | 加载基础数学函数库
    
    Args | 参数:
        expected_dim: Expected feature dimension | 期望的特征维度
        
    Returns | 返回:
        List[FunctionMemory]: Function memory list | 函数记忆对象列表
    """
    data = _load_pretrained_file(PRETRAINED_FILES['basic'])
    functions = []
    
    for item in data:
        try:
            item_copy = item.copy()
            item_copy.pop('id', None)
            
            if 'feature_vector' in item_copy:
                item_copy['feature_vector'] = _safe_convert_feature_vector(
                    item_copy['feature_vector'], expected_dim
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
            logger.error(f"Failed to create function: {e} | 创建函数失败")
            continue
    
    logger.info(f"Loaded basic functions: {len(functions)} | 加载基础函数库")
    return functions


def load_physics_functions(expected_dim: Optional[int] = None) -> List[FunctionMemory]:
    """Load physics function library | 加载物理系统函数库"""
    data = _load_pretrained_file(PRETRAINED_FILES['physics'])
    functions = []
    
    for item in data:
        try:
            item_copy = item.copy()
            item_copy.pop('id', None)
            
            if 'feature_vector' in item_copy:
                item_copy['feature_vector'] = _safe_convert_feature_vector(
                    item_copy['feature_vector'], expected_dim
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
            logger.error(f"Failed to create physics function: {e} | 创建物理函数失败")
            continue
    
    logger.info(f"Loaded physics functions: {len(functions)} | 加载物理函数库")
    return functions


def load_chemistry_functions(expected_dim: Optional[int] = None) -> List[FunctionMemory]:
    """Load chemistry function library | 加载化学反应函数库"""
    data = _load_pretrained_file(PRETRAINED_FILES['chemistry'])
    functions = []
    
    for item in data:
        try:
            item_copy = item.copy()
            item_copy.pop('id', None)
            
            if 'feature_vector' in item_copy:
                item_copy['feature_vector'] = _safe_convert_feature_vector(
                    item_copy['feature_vector'], expected_dim
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
            logger.error(f"Failed to create chemistry function: {e} | 创建化学函数失败")
            continue
    
    logger.info(f"Loaded chemistry functions: {len(functions)} | 加载化学函数库")
    return functions


def load_chaos_functions(expected_dim: Optional[int] = None) -> List[FunctionMemory]:
    """Load chaos system function library | 加载混沌系统函数库"""
    data = _load_pretrained_file(PRETRAINED_FILES['chaos'])
    functions = []
    
    for item in data:
        try:
            item_copy = item.copy()
            item_copy.pop('id', None)
            
            if 'feature_vector' in item_copy:
                item_copy['feature_vector'] = _safe_convert_feature_vector(
                    item_copy['feature_vector'], expected_dim
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
            logger.error(f"Failed to create chaos function: {e} | 创建混沌函数失败")
            continue
    
    logger.info(f"Loaded chaos functions: {len(functions)} | 加载混沌函数库")
    return functions


def load_all_pretrained(limits: Optional[Dict[str, int]] = None,
                       expected_dim: Optional[int] = None) -> Dict[str, List[FunctionMemory]]:
    """
    Load all pretrained function libraries | 加载所有预训练函数库
    
    Args | 参数:
        limits: Limits per library, e.g., {'basic': 100} | 各库的限制数量
        expected_dim: Expected feature dimension | 期望的特征维度
        
    Returns | 返回:
        Dict: Library name to function list mapping | 库名到函数列表的映射
    """
    if limits is None:
        limits = {}
    
    result = {}
    
    basic_funcs = load_basic_functions(expected_dim)
    if 'basic' in limits and limits['basic'] > 0:
        basic_funcs = basic_funcs[:min(limits['basic'], len(basic_funcs))]
    result['basic'] = basic_funcs
    
    physics_funcs = load_physics_functions(expected_dim)
    if 'physics' in limits and limits['physics'] > 0:
        physics_funcs = physics_funcs[:min(limits['physics'], len(physics_funcs))]
    result['physics'] = physics_funcs
    
    chemistry_funcs = load_chemistry_functions(expected_dim)
    if 'chemistry' in limits and limits['chemistry'] > 0:
        chemistry_funcs = chemistry_funcs[:min(limits['chemistry'], len(chemistry_funcs))]
    result['chemistry'] = chemistry_funcs
    
    chaos_funcs = load_chaos_functions(expected_dim)
    if 'chaos' in limits and limits['chaos'] > 0:
        chaos_funcs = chaos_funcs[:min(limits['chaos'], len(chaos_funcs))]
    result['chaos'] = chaos_funcs
    
    total = sum(len(funcs) for funcs in result.values())
    logger.info(f"Loaded all pretrained libraries: {total} total functions | 加载所有预训练库完成")
    
    return result


def list_available_pretrained() -> List[Dict[str, Any]]:
    """
    List all available pretrained libraries | 列出所有可用的预训练库信息
    
    Returns | 返回:
        List[Dict]: Library information list | 每个库的信息
    """
    info_list = []
    
    for key, info in PRETRAINED_INFO.items():
        file_path = PRETRAINED_FILES.get(key)
        file_exists = file_path.exists() if file_path else False
        
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
    Get detailed information for a pretrained library | 获取指定预训练库的详细信息
    
    Args | 参数:
        library_key: Library key ('basic', 'physics', 'chemistry', 'chaos') | 库键名
        
    Returns | 返回:
        Optional[Dict]: Library information | 库信息
    """
    if library_key not in PRETRAINED_INFO:
        return None
    
    info = PRETRAINED_INFO[library_key].copy()
    info['key'] = library_key
    info['file_exists'] = PRETRAINED_FILES[library_key].exists()
    info['file_path'] = str(PRETRAINED_FILES[library_key])
    
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
    Create a pretrained memory bank | 创建预训练记忆库
    
    Args | 参数:
        library_keys: Library keys to load (None for all) | 要加载的库键名列表
        limits: Limits per library | 各库的限制数量
        expected_dim: Expected feature dimension | 期望的特征维度
        
    Returns | 返回:
        FunctionMemoryBank: Memory bank with pretrained functions | 包含预训练函数的记忆库
    """
    if library_keys is None:
        library_keys = ['basic', 'physics', 'chemistry', 'chaos']
    
    bank = FunctionMemoryBank(memory_backend='memory')
    all_functions = load_all_pretrained(limits, expected_dim)
    
    added_count = 0
    for key in library_keys:
        if key in all_functions:
            funcs = all_functions[key]
            for func in funcs:
                if bank.add(func):
                    added_count += 1
            logger.info(f"Added {len(funcs)} {key} functions to memory bank | 向记忆库添加")
    
    logger.info(f"Pretrained memory bank created: {added_count} functions | 预训练记忆库创建完成")
    
    return bank


__all__ = [
    'load_basic_functions',
    'load_physics_functions',
    'load_chemistry_functions',
    'load_chaos_functions',
    'load_all_pretrained',
    'list_available_pretrained',
    'get_pretrained_info',
    'create_pretrained_bank',
    'PRETRAINED_INFO',
    'PRETRAINED_FILES',
]