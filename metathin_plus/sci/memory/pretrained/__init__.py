"""
预训练函数库 - 开箱即用的函数知识
====================================
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

from ..function_memory import FunctionMemory, FunctionMemoryBank

# 获取当前文件所在目录
PRETRAINED_DIR = Path(__file__).parent

PRETRAINED_FILES = {
    'basic': PRETRAINED_DIR / 'basic_functions.json',
    'physics': PRETRAINED_DIR / 'physics_functions.json',
    'chemistry': PRETRAINED_DIR / 'chemistry_functions.json',
    'chaos': PRETRAINED_DIR / 'chaos_functions.json',  # 新增混沌库
}

PRETRAINED_INFO = {
    'basic': {
        'name': '基础数学函数库',
        'description': '包含线性、多项式、三角函数、指数、对数等基础函数',
        'size': 100,
        'version': '1.0',
        'tags': ['mathematics', 'basic', 'trigonometric', 'polynomial']
    },
    'physics': {
        'name': '物理系统函数库',
        'description': '包含单摆、弹簧振子、混沌系统等物理函数',
        'size': 50,
        'version': '1.0',
        'tags': ['physics', 'oscillation', 'chaos', 'mechanics']
    },
    'chemistry': {
        'name': '化学反应函数库',
        'description': '包含一级反应、二级反应、振荡反应等化学函数',
        'size': 30,
        'version': '1.0',
        'tags': ['chemistry', 'kinetics', 'reaction']
    },
    'chaos': {  # 新增混沌库信息
        'name': '混沌系统函数库',
        'description': '包含洛伦兹、罗斯勒、杜芬等混沌系统的特征',
        'size': 100,
        'version': '1.0',
        'tags': ['chaos', 'nonlinear', 'dynamics', 'lorenz', 'rossler']
    }
}

# 设置日志
logger = logging.getLogger("metathin_sci.memory.pretrained")


def _load_pretrained_file(filename: Path) -> List[Dict[str, Any]]:
    """
    加载预训练文件
    
    Args:
        filename: 文件路径
        
    Returns:
        List[Dict]: 函数数据列表，失败返回空列表
    """
    if not filename.exists():
        logger.warning(f"预训练文件不存在: {filename}")
        return []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理不同的JSON结构
        if isinstance(data, dict):
            # 如果有metadata字段，打印一些信息
            if 'metadata' in data:
                logger.info(f"加载 {data['metadata'].get('name', '未知库')}: "
                           f"{data['metadata'].get('n_functions', 0)} 个函数")
            
            # 获取函数列表
            if 'functions' in data:
                function_list = data['functions']
            elif isinstance(data.get('data'), list):
                function_list = data['data']
            else:
                # 尝试将整个字典转换为列表
                function_list = [data]
        elif isinstance(data, list):
            function_list = data
        else:
            logger.warning(f"未知的数据格式: {filename}")
            return []
        
        # ===== 修复点：过滤掉不受支持的字段 =====
        filtered_list = []
        for item in function_list:
            # 创建副本，避免修改原数据
            item_copy = item.copy() if isinstance(item, dict) else item
            
            if isinstance(item_copy, dict):
                # 删除 FunctionMemory 不支持的字段
                unsupported_fields = ['complexity', 'id']  # 可以添加其他不支持的字段
                for field in unsupported_fields:
                    if field in item_copy:
                        del item_copy[field]
            
            filtered_list.append(item_copy)
        
        return filtered_list
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败 {filename}: {e}")
        return []
    except Exception as e:
        logger.error(f"加载预训练文件失败 {filename}: {e}")
        return []
def _safe_convert_feature_vector(feature_vector: Any, expected_dim: Optional[int] = None) -> np.ndarray:
    """
    安全地转换特征向量为numpy数组
    
    Args:
        feature_vector: 输入的特征向量（可以是列表、数组、数字等）
        expected_dim: 期望的维度，如果指定则进行填充或截断
        
    Returns:
        np.ndarray: 转换后的float64数组
    """
    try:
        # 处理None
        if feature_vector is None:
            if expected_dim:
                return np.zeros(expected_dim, dtype=np.float64)
            return np.array([0.0], dtype=np.float64)
        
        # 如果是numpy数组
        if isinstance(feature_vector, np.ndarray):
            arr = feature_vector.astype(np.float64)
        # 如果是列表或元组
        elif isinstance(feature_vector, (list, tuple)):
            # 过滤非数字元素，转换为float
            float_list = []
            for x in feature_vector:
                try:
                    float_list.append(float(x))
                except (TypeError, ValueError):
                    float_list.append(0.0)
            arr = np.array(float_list, dtype=np.float64)
        # 如果是单个数字
        elif isinstance(feature_vector, (int, float)):
            arr = np.array([float(feature_vector)], dtype=np.float64)
        # 如果是字符串
        elif isinstance(feature_vector, str):
            # 尝试解析字符串为数字
            try:
                val = float(feature_vector)
                arr = np.array([val], dtype=np.float64)
            except ValueError:
                # 如果失败，返回特征长度作为特征
                arr = np.array([float(len(feature_vector))], dtype=np.float64)
        else:
            # 其他类型，返回0
            arr = np.array([0.0], dtype=np.float64)
        
        # 处理维度
        if expected_dim is not None:
            if len(arr) > expected_dim:
                # 截断
                arr = arr[:expected_dim]
            elif len(arr) < expected_dim:
                # 填充
                padding = expected_dim - len(arr)
                arr = np.pad(arr, (0, padding), mode='constant', constant_values=0)
        
        return arr
        
    except Exception as e:
        logger.error(f"特征向量转换失败: {e}")
        if expected_dim:
            return np.zeros(expected_dim, dtype=np.float64)
        return np.array([0.0], dtype=np.float64)


def load_basic_functions(expected_dim: Optional[int] = None) -> List[FunctionMemory]:
    """
    加载基础数学函数库
    
    Args:
        expected_dim: 期望的特征维度，如果指定则调整所有特征向量到此维度
        
    Returns:
        List[FunctionMemory]: 函数记忆对象列表
    """
    data = _load_pretrained_file(PRETRAINED_FILES['basic'])
    functions = []
    
    for item in data:
        try:
            # 移除可能存在的id字段（因为它会在__init__时自动生成）
            item_copy = item.copy()
            item_copy.pop('id', None)
            
            # 安全转换特征向量
            if 'feature_vector' in item_copy:
                item_copy['feature_vector'] = _safe_convert_feature_vector(
                    item_copy['feature_vector'], 
                    expected_dim
                )
            else:
                logger.warning(f"函数 {item_copy.get('expression', 'unknown')} 缺少feature_vector字段")
                item_copy['feature_vector'] = _safe_convert_feature_vector([], expected_dim)
            
            # 验证必要字段
            if 'expression' not in item_copy:
                logger.warning("函数缺少expression字段，跳过")
                continue
                
            if 'parameters' not in item_copy:
                item_copy['parameters'] = {}
            
            # 创建FunctionMemory对象
            func = FunctionMemory.from_dict(item_copy)
            functions.append(func)
            
        except Exception as e:
            logger.error(f"创建函数记忆对象失败: {e}, 数据: {item.get('expression', 'unknown')}")
            continue
    
    logger.info(f"加载基础函数库: {len(functions)} 个函数")
    return functions


def load_physics_functions(expected_dim: Optional[int] = None) -> List[FunctionMemory]:
    """
    加载物理系统函数库
    
    Args:
        expected_dim: 期望的特征维度
        
    Returns:
        List[FunctionMemory]: 函数记忆对象列表
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
            logger.error(f"创建物理函数失败: {e}")
            continue
    
    logger.info(f"加载物理函数库: {len(functions)} 个函数")
    return functions


def load_chemistry_functions(expected_dim: Optional[int] = None) -> List[FunctionMemory]:
    """
    加载化学反应函数库
    
    Args:
        expected_dim: 期望的特征维度
        
    Returns:
        List[FunctionMemory]: 函数记忆对象列表
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
            logger.error(f"创建化学函数失败: {e}")
            continue
    
    logger.info(f"加载化学函数库: {len(functions)} 个函数")
    return functions


def load_chaos_functions(expected_dim: Optional[int] = None) -> List[FunctionMemory]:
    """
    加载混沌系统函数库
    
    Args:
        expected_dim: 期望的特征维度
        
    Returns:
        List[FunctionMemory]: 函数记忆对象列表
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
            logger.error(f"创建混沌函数失败: {e}")
            continue
    
    logger.info(f"加载混沌函数库: {len(functions)} 个函数")
    return functions


def load_all_pretrained(limits: Optional[Dict[str, int]] = None,
                       expected_dim: Optional[int] = None) -> Dict[str, List[FunctionMemory]]:
    """
    加载所有预训练函数库
    
    Args:
        limits: 各库的限制数量，如 {'basic': 100, 'physics': 50}
        expected_dim: 期望的特征维度
        
    Returns:
        Dict: 库名到函数列表的映射
    """
    if limits is None:
        limits = {}
    
    result = {}
    
    # 加载基础函数库
    basic_funcs = load_basic_functions(expected_dim)
    if 'basic' in limits and limits['basic'] > 0:
        basic_funcs = basic_funcs[:min(limits['basic'], len(basic_funcs))]
    result['basic'] = basic_funcs
    
    # 加载物理函数库
    physics_funcs = load_physics_functions(expected_dim)
    if 'physics' in limits and limits['physics'] > 0:
        physics_funcs = physics_funcs[:min(limits['physics'], len(physics_funcs))]
    result['physics'] = physics_funcs
    
    # 加载化学函数库
    chemistry_funcs = load_chemistry_functions(expected_dim)
    if 'chemistry' in limits and limits['chemistry'] > 0:
        chemistry_funcs = chemistry_funcs[:min(limits['chemistry'], len(chemistry_funcs))]
    result['chemistry'] = chemistry_funcs
    
    # 加载混沌函数库
    chaos_funcs = load_chaos_functions(expected_dim)
    if 'chaos' in limits and limits['chaos'] > 0:
        chaos_funcs = chaos_funcs[:min(limits['chaos'], len(chaos_funcs))]
    result['chaos'] = chaos_funcs
    
    total = sum(len(funcs) for funcs in result.values())
    logger.info(f"加载所有预训练库完成，总计 {total} 个函数")
    
    return result


def list_available_pretrained() -> List[Dict[str, Any]]:
    """
    列出所有可用的预训练库信息
    
    Returns:
        List[Dict]: 每个库的信息
    """
    info_list = []
    
    for key, info in PRETRAINED_INFO.items():
        file_path = PRETRAINED_FILES.get(key)
        file_exists = file_path.exists() if file_path else False
        
        # 如果文件存在，尝试读取实际数量
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
    获取指定预训练库的详细信息
    
    Args:
        library_key: 库键名 ('basic', 'physics', 'chemistry', 'chaos')
        
    Returns:
        Optional[Dict]: 库信息，不存在返回None
    """
    if library_key not in PRETRAINED_INFO:
        return None
    
    info = PRETRAINED_INFO[library_key].copy()
    info['key'] = library_key
    info['file_exists'] = PRETRAINED_FILES[library_key].exists()
    info['file_path'] = str(PRETRAINED_FILES[library_key])
    
    # 如果文件存在，读取实际数量
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
    创建预训练记忆库
    
    Args:
        library_keys: 要加载的库键名列表，None表示全部
        limits: 各库的限制数量
        expected_dim: 期望的特征维度
        
    Returns:
        FunctionMemoryBank: 包含预训练函数的记忆库
    """
    if library_keys is None:
        library_keys = ['basic', 'physics', 'chemistry', 'chaos']
    
    # 创建内存模式的记忆库（不持久化）
    bank = FunctionMemoryBank(memory_backend='memory')
    
    # 加载所有函数
    all_functions = load_all_pretrained(limits, expected_dim)
    
    added_count = 0
    for key in library_keys:
        if key in all_functions:
            funcs = all_functions[key]
            for func in funcs:
                if bank.add(func):
                    added_count += 1
            logger.info(f"向记忆库添加 {len(funcs)} 个 {key} 函数")
    
    logger.info(f"预训练记忆库创建完成，共添加 {added_count} 个函数")
    
    return bank


__all__ = [
    'load_basic_functions',
    'load_physics_functions',
    'load_chemistry_functions',
    'load_chaos_functions',  # 新增
    'load_all_pretrained',
    'list_available_pretrained',
    'get_pretrained_info',
    'create_pretrained_bank',
    'PRETRAINED_INFO',
    'PRETRAINED_FILES',
]
