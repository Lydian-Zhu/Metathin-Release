# metathin_sci/memory/function_memory.py
"""
Function Memory - Store and Manage Function Knowledge | 函数记忆库 - 存储和管理函数知识
=======================================================================================

Provides memory storage for discovered functions with two-tier caching.
Supports persistent storage via JSON or SQLite backends.

提供已发现函数的记忆存储，支持二级缓存。
通过 JSON 或 SQLite 后端支持持久化存储。
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
import logging
from pathlib import Path


# ============================================================
# Function Memory Data Class | 函数记忆数据类
# ============================================================

@dataclass
class FunctionMemory:
    """
    Function memory data class | 函数记忆数据类
    
    Stores a discovered function with its features and metadata.
    
    存储一个已发现的函数及其特征和元数据。
    
    Attributes | 属性:
        expression: Function expression string | 函数表达式字符串
        parameters: Parameter dictionary | 参数字典
        feature_vector: Feature vector for similarity matching | 特征向量
        accuracy: Recognition accuracy (0-1) | 识别准确率
        usage_count: Number of times used | 使用次数
        domain: Valid domain (min, max) | 有效定义域
        tags: Tag list for categorization | 标签列表
        source: Source of the function | 来源
        created_at: Creation timestamp | 创建时间戳
        last_used: Last used timestamp | 最后使用时间
        metadata: Additional metadata | 额外元数据
    """
    
    # Required fields | 必需字段
    expression: str
    parameters: Dict[str, float]
    feature_vector: np.ndarray
    
    # Optional fields | 可选字段
    accuracy: float = 1.0
    usage_count: int = 0
    domain: Tuple[float, float] = (-10.0, 10.0)
    tags: List[str] = field(default_factory=list)
    source: str = "generated"
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    last_used: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def id(self) -> str:
        """Generate unique ID (computed property) | 生成唯一ID（计算属性）"""
        content = f"{self.expression}_{self.parameters}_{self.created_at}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def __post_init__(self):
        """Post-initialization processing | 初始化后处理"""
        # Handle feature vector | 处理特征向量
        if self.feature_vector is None:
            self.feature_vector = np.array([0.0], dtype=np.float64)
        elif not isinstance(self.feature_vector, np.ndarray):
            try:
                self.feature_vector = np.array(self.feature_vector, dtype=np.float64)
            except:
                self.feature_vector = np.array([0.0], dtype=np.float64)
        
        # Ensure 1D | 确保一维
        if self.feature_vector.ndim > 1:
            self.feature_vector = self.feature_vector.flatten()
        
        # Ensure not empty | 确保不为空
        if len(self.feature_vector) == 0:
            self.feature_vector = np.array([0.0], dtype=np.float64)
        
        # Handle parameters | 处理参数
        if self.parameters is None:
            self.parameters = {}
        
        # Handle tags | 处理标签
        if self.tags is None:
            self.tags = []
        
        # Handle metadata | 处理元数据
        if self.metadata is None:
            self.metadata = {}
        
        # Clamp accuracy | 限制准确率范围
        self.accuracy = max(0.0, min(1.0, self.accuracy))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization | 转换为字典（用于序列化）"""
        return {
            'id': self.id,
            'expression': self.expression,
            'parameters': self.parameters.copy(),
            'feature_vector': self.feature_vector.tolist(),
            'accuracy': float(self.accuracy),
            'usage_count': int(self.usage_count),
            'domain': (float(self.domain[0]), float(self.domain[1])),
            'tags': self.tags.copy(),
            'source': self.source,
            'created_at': float(self.created_at),
            'last_used': float(self.last_used),
            'metadata': self.metadata.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FunctionMemory':
        """Create from dictionary | 从字典创建"""
        data_copy = data.copy()
        data_copy.pop('id', None)  # Remove id, it's computed | 移除id，它是计算属性
        
        # Handle feature vector | 处理特征向量
        if 'feature_vector' in data_copy:
            if isinstance(data_copy['feature_vector'], list):
                data_copy['feature_vector'] = np.array(data_copy['feature_vector'], dtype=np.float64)
            elif not isinstance(data_copy['feature_vector'], np.ndarray):
                data_copy['feature_vector'] = np.array([0.0], dtype=np.float64)
        
        # Handle domain | 处理域
        if 'domain' in data_copy and isinstance(data_copy['domain'], list):
            data_copy['domain'] = tuple(data_copy['domain'])
        
        return cls(**data_copy)
    
    def update_usage(self):
        """Update usage statistics | 更新使用统计"""
        self.usage_count += 1
        self.last_used = datetime.now().timestamp()
    
    def similarity(self, other: 'FunctionMemory') -> float:
        """
        Calculate similarity with another function | 计算与另一个函数的相似度
        
        Args | 参数:
            other: Other function memory | 另一个函数记忆
            
        Returns | 返回:
            float: Similarity score (0-1) | 相似度分数
        """
        if other is None or other.feature_vector is None:
            return 0.0
        
        v1 = self.feature_vector
        v2 = other.feature_vector
        
        if v1 is None or v2 is None or len(v1) == 0 or len(v2) == 0:
            return 0.0
        
        # Handle dimension mismatch | 处理维度不匹配
        if len(v1) != len(v2):
            min_len = min(len(v1), len(v2))
            v1 = v1[:min_len]
            v2 = v2[:min_len]
        
        # Calculate cosine similarity | 计算余弦相似度
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cos_sim = np.dot(v1, v2) / (norm1 * norm2)
        # Map [-1,1] to [0,1] | 将[-1,1]映射到[0,1]
        return float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))
    
    def __repr__(self) -> str:
        return f"FunctionMemory(expr='{self.expression}', acc={self.accuracy:.2f}, used={self.usage_count})"


# ============================================================
# Simple Memory Backend (Fallback) | 简化版记忆后端（降级方案）
# ============================================================

class SimpleMemoryBackend:
    """
    Simple memory backend (in-memory only) | 简化版记忆后端（仅内存）
    
    Provides compatibility with Metathin memory system interface.
    
    提供与Metathin记忆系统兼容的接口。
    """
    
    def __init__(self):
        self._memory: Dict[str, Any] = {}
        self._logger = logging.getLogger("metathin_sci.memory.SimpleMemoryBackend")
    
    def save(self, key: str, value: Any) -> bool:
        """Save memory | 保存记忆"""
        try:
            self._memory[key] = value
            self._logger.debug(f"Saved: {key} | 保存")
            return True
        except Exception as e:
            self._logger.error(f"Failed to save {key}: {e} | 保存失败")
            return False
    
    def load(self, key: str) -> Optional[Any]:
        """Load memory | 加载记忆"""
        value = self._memory.get(key)
        self._logger.debug(f"Loaded: {key} -> {'found' if value is not None else 'not found'} | 加载")
        return value
    
    def delete(self, key: str) -> bool:
        """Delete memory | 删除记忆"""
        if key in self._memory:
            del self._memory[key]
            self._logger.debug(f"Deleted: {key} | 删除")
            return True
        return False
    
    def list_keys(self) -> List[str]:
        """List all keys | 列出所有键"""
        return list(self._memory.keys())
    
    def clear(self) -> bool:
        """Clear all memory | 清空记忆"""
        self._memory.clear()
        self._logger.info("Cleared all memory | 清空所有记忆")
        return True
    
    def get_size(self) -> int:
        """Get number of records | 获取记录数量"""
        return len(self._memory)


class SimpleJSONMemoryBackend:
    """
    Simple JSON memory backend | 简化版JSON记忆后端
    
    Provides JSON file-based persistent storage.
    
    提供基于JSON文件的持久化存储。
    """
    
    def __init__(self, filepath: Union[str, Path] = "function_memory.json"):
        self.filepath = Path(filepath)
        self._memory: Dict[str, Any] = {}
        self._logger = logging.getLogger("metathin_sci.memory.SimpleJSONMemoryBackend")
        self._load()
    
    def _load(self):
        """Load from file | 从文件加载"""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    self._memory = json.load(f)
                self._logger.info(f"Loaded {len(self._memory)} memories from {self.filepath} | 加载")
            except Exception as e:
                self._logger.error(f"Failed to load: {e} | 加载失败")
                self._memory = {}
    
    def _save(self) -> bool:
        """Save to file | 保存到文件"""
        try:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self._memory, f, indent=2, ensure_ascii=False, default=str)
            return True
        except Exception as e:
            self._logger.error(f"Failed to save: {e} | 保存失败")
            return False
    
    def save(self, key: str, value: Any) -> bool:
        """Save memory | 保存记忆"""
        try:
            self._memory[key] = value
            self._save()
            self._logger.debug(f"Saved: {key} | 保存")
            return True
        except Exception as e:
            self._logger.error(f"Failed to save {key}: {e} | 保存失败")
            return False
    
    def load(self, key: str) -> Optional[Any]:
        """Load memory | 加载记忆"""
        return self._memory.get(key)
    
    def delete(self, key: str) -> bool:
        """Delete memory | 删除记忆"""
        if key in self._memory:
            del self._memory[key]
            self._save()
            self._logger.debug(f"Deleted: {key} | 删除")
            return True
        return False
    
    def list_keys(self) -> List[str]:
        """List all keys | 列出所有键"""
        return list(self._memory.keys())
    
    def clear(self) -> bool:
        """Clear all memory | 清空记忆"""
        self._memory.clear()
        self._save()
        self._logger.info("Cleared all memory | 清空所有记忆")
        return True


# ============================================================
# Function Memory Bank Manager | 函数记忆库管理器
# ============================================================

class FunctionMemoryBank:
    """
    Function Memory Bank Manager | 函数记忆库管理器
    
    Manages storage, retrieval, and learning of function memories.
    Provides persistent storage and fast retrieval with similarity matching.
    
    管理函数记忆的存储、检索和学习。
    提供持久化存储和快速检索。
    
    Parameters | 参数:
        memory_backend: Backend type ('json', 'memory') | 后端类型
        memory_path: Storage path | 存储路径
        auto_save: Whether to auto-save | 是否自动保存
    """
    
    def __init__(self,
                 memory_backend: str = 'json',
                 memory_path: Optional[str] = None,
                 auto_save: bool = True):
        """
        Initialize function memory bank | 初始化函数记忆库
        
        Args | 参数:
            memory_backend: Backend type ('json', 'memory') | 后端类型
            memory_path: Storage path | 存储路径
            auto_save: Whether to auto-save | 是否自动保存
        """
        self.auto_save = auto_save
        self._logger = logging.getLogger("metathin_sci.memory.FunctionMemoryBank")
        
        # Initialize memory backend | 初始化记忆后端
        if memory_backend == 'json':
            path = memory_path or "function_memory.json"
            self.memory = SimpleJSONMemoryBackend(path)
            self._logger.info(f"Using JSON memory backend: {path} | 使用JSON记忆后端")
        else:
            self.memory = SimpleMemoryBackend()
            self._logger.info("Using in-memory backend | 使用内存记忆后端")
        
        # Index for fast similarity matching | 索引（用于快速相似度匹配）
        self.functions: List[FunctionMemory] = []
        self.feature_matrix: Optional[np.ndarray] = None
        self._need_rebuild_index = False
        
        # Load existing memories | 加载现有记忆
        self._load_all()
        
        self._logger.info(f"FunctionMemoryBank initialized: {len(self.functions)} functions | 初始化完成")
    
    def _load_all(self):
        """Load all memories | 加载所有记忆"""
        try:
            keys = self.memory.list_keys()
            for key in keys:
                try:
                    data = self.memory.load(key)
                    if data:
                        if isinstance(data, dict):
                            func = FunctionMemory.from_dict(data)
                        else:
                            func = data
                        if func is not None and isinstance(func, FunctionMemory):
                            self.functions.append(func)
                except Exception as e:
                    self._logger.debug(f"Failed to load memory {key}: {e} | 加载单个记忆失败")
                    continue
            
            if self.functions:
                self._rebuild_index()
                self._logger.info(f"Loaded {len(self.functions)} function memories | 加载了 {len(self.functions)} 个函数记忆")
                
        except Exception as e:
            self._logger.error(f"Failed to load memories: {e} | 加载记忆失败")
    
    def _rebuild_index(self):
        """Rebuild feature index | 重建特征索引"""
        if not self.functions:
            self.feature_matrix = None
            return
        
        # Build feature matrix | 构建特征矩阵
        features = []
        valid_indices = []
        
        for i, func in enumerate(self.functions):
            if func is not None and func.feature_vector is not None:
                fv = func.feature_vector
                if len(fv) > 0 and not np.any(np.isnan(fv)) and not np.any(np.isinf(fv)):
                    features.append(fv)
                    valid_indices.append(i)
        
        if not features:
            self.feature_matrix = None
            return
        
        # Align feature dimensions | 对齐特征维度
        max_dim = max(len(f) for f in features)
        aligned_features = []
        
        for f in features:
            if len(f) < max_dim:
                aligned_f = np.pad(f, (0, max_dim - len(f)), mode='constant', constant_values=0)
            else:
                aligned_f = f[:max_dim]
            aligned_features.append(aligned_f)
        
        self.feature_matrix = np.array(aligned_features)
        self._need_rebuild_index = False
        
        self._logger.debug(f"Index rebuilt: feature matrix shape {self.feature_matrix.shape} | 索引重建完成")
    
    def add(self, func: FunctionMemory) -> bool:
        """
        Add a function memory | 添加函数记忆
        
        Args | 参数:
            func: Function memory object | 函数记忆对象
            
        Returns | 返回:
            bool: True if successful | 是否成功
        """
        if func is None:
            self._logger.warning("Attempted to add None function | 尝试添加None函数")
            return False
        
        if not isinstance(func, FunctionMemory):
            self._logger.warning(f"Attempted to add non-FunctionMemory object: {type(func)} | 尝试添加非FunctionMemory对象")
            return False
        
        try:
            key = f"func_{func.id}"
            success = self.memory.save(key, func.to_dict())
            
            if success:
                self.functions.append(func)
                self._need_rebuild_index = True
                self._logger.debug(f"Added function: {func.expression} | 添加函数")
                
                if self.auto_save:
                    self.save()
                
                return True
            return False
            
        except Exception as e:
            self._logger.error(f"Failed to add function: {e} | 添加函数失败")
            return False
    
    def add_batch(self, funcs: List[FunctionMemory]) -> int:
        """
        Batch add function memories | 批量添加函数记忆
        
        Args | 参数:
            funcs: List of function memories | 函数记忆列表
            
        Returns | 返回:
            int: Number of successfully added | 成功添加的数量
        """
        success_count = 0
        for func in funcs:
            if self.add(func):
                success_count += 1
        
        self._logger.info(f"Batch add complete: {success_count}/{len(funcs)} | 批量添加完成")
        return success_count
    
    def remove(self, func_id: str) -> bool:
        """
        Remove a function memory | 移除函数记忆
        
        Args | 参数:
            func_id: Function ID | 函数ID
            
        Returns | 返回:
            bool: True if successful | 是否成功
        """
        try:
            remove_idx = None
            for i, func in enumerate(self.functions):
                if func is not None and func.id == func_id:
                    remove_idx = i
                    break
            
            if remove_idx is None:
                self._logger.warning(f"Function not found: {func_id} | 未找到要移除的函数")
                return False
            
            self.functions.pop(remove_idx)
            
            key = f"func_{func_id}"
            success = self.memory.delete(key)
            
            if success:
                self._need_rebuild_index = True
                self._logger.debug(f"Removed function: {func_id} | 移除函数")
                
                if self.auto_save:
                    self.save()
            
            return success
            
        except Exception as e:
            self._logger.error(f"Failed to remove function: {e} | 移除函数失败")
            return False
    
    def get(self, func_id: str) -> Optional[FunctionMemory]:
        """
        Get a function memory by ID | 根据ID获取函数记忆
        
        Args | 参数:
            func_id: Function ID | 函数ID
            
        Returns | 返回:
            Optional[FunctionMemory]: Function memory or None | 函数记忆对象或None
        """
        for func in self.functions:
            if func is not None and func.id == func_id:
                func.update_usage()
                return func
        return None
    
    def find_similar(self,
                    query_features: np.ndarray,
                    k: int = 5,
                    threshold: float = 0.7,
                    min_similarity: float = 0.3) -> List[FunctionMemory]:
        """
        Find similar functions | 查找相似函数
        
        Args | 参数:
            query_features: Query feature vector | 查询特征向量
            k: Maximum number to return | 返回的最大数量
            threshold: Similarity threshold | 相似度阈值
            min_similarity: Minimum similarity (fallback) | 最小相似度（降级使用）
            
        Returns | 返回:
            List[FunctionMemory]: Similar functions list | 相似函数列表
        """
        # Validate input | 验证输入
        if query_features is None or not self.functions:
            return []
        
        # Ensure 1D | 确保一维
        if isinstance(query_features, np.ndarray):
            if query_features.ndim > 1:
                query_features = query_features.flatten()
        else:
            try:
                query_features = np.array(query_features, dtype=np.float64)
                if query_features.ndim > 1:
                    query_features = query_features.flatten()
            except Exception as e:
                self._logger.error(f"Feature conversion failed: {e} | 特征转换失败")
                return []
        
        # Validate feature | 验证特征有效性
        if len(query_features) == 0 or np.any(np.isnan(query_features)) or np.any(np.isinf(query_features)):
            return []
        
        # Rebuild index if needed | 重建索引如果需要
        if self._need_rebuild_index:
            self._rebuild_index()
        
        if self.feature_matrix is None:
            return []
        
        # Align dimensions | 对齐维度
        if len(query_features) != self.feature_matrix.shape[1]:
            if len(query_features) > self.feature_matrix.shape[1]:
                query_features = query_features[:self.feature_matrix.shape[1]]
            else:
                query_features = np.pad(query_features, 
                                      (0, self.feature_matrix.shape[1] - len(query_features)),
                                      mode='constant', constant_values=0)
        
        # Calculate cosine similarity | 计算余弦相似度
        try:
            query_norm = np.linalg.norm(query_features)
            if query_norm < 1e-10:
                return []
            
            norms = np.linalg.norm(self.feature_matrix, axis=1)
            norms = np.maximum(norms, 1e-10)
            
            dot_products = np.dot(self.feature_matrix, query_features)
            similarities = dot_products / (norms * query_norm)
            similarities = (similarities + 1.0) / 2.0  # Map to [0,1] | 映射到[0,1]
            similarities = np.clip(similarities, 0.0, 1.0)
            
            # Find above threshold | 查找高于阈值的
            above_threshold = np.where(similarities >= threshold)[0]
            
            results = []
            if len(above_threshold) > 0:
                sorted_indices = above_threshold[np.argsort(similarities[above_threshold])[::-1]]
                result_indices = sorted_indices[:k]
                
                for idx in result_indices:
                    if idx < len(self.functions) and self.functions[idx] is not None:
                        func = self.functions[idx]
                        func.update_usage()
                        results.append(func)
                
                return results
            
            # Fallback to lower threshold | 降级到更低阈值
            above_min = np.where(similarities >= min_similarity)[0]
            
            if len(above_min) > 0:
                sorted_indices = above_min[np.argsort(similarities[above_min])[::-1]]
                result_indices = sorted_indices[:k]
                
                for idx in result_indices:
                    if idx < len(self.functions) and self.functions[idx] is not None:
                        func = self.functions[idx]
                        func.update_usage()
                        results.append(func)
                
                return results
            
            # Return most similar if any | 返回最相似的
            if len(self.functions) > 0:
                max_idx = np.argmax(similarities)
                if max_idx < len(self.functions) and self.functions[max_idx] is not None:
                    max_sim = similarities[max_idx]
                    if max_sim > 0.1:
                        func = self.functions[max_idx]
                        func.update_usage()
                        return [func]
            
        except Exception as e:
            self._logger.error(f"Similarity calculation failed: {e} | 相似度计算失败")
        
        return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics | 获取统计信息
        
        Returns | 返回:
            Dict: Statistics dictionary | 统计信息
        """
        valid_functions = [f for f in self.functions if f is not None]
        
        if not valid_functions:
            return {
                'total': 0,
                'feature_dim': 0,
                'tag_distribution': {},
                'source_distribution': {},
                'accuracy': {'mean': 0, 'min': 0, 'max': 0},
                'usage_stats': {'total_uses': 0, 'avg_uses': 0, 'max_uses': 0}
            }
        
        # Tag distribution | 标签分布
        tag_count = {}
        for func in valid_functions:
            for tag in func.tags:
                tag_count[tag] = tag_count.get(tag, 0) + 1
        
        # Source distribution | 来源分布
        source_count = {}
        for func in valid_functions:
            source_count[func.source] = source_count.get(func.source, 0) + 1
        
        # Accuracy statistics | 准确率统计
        accuracies = [func.accuracy for func in valid_functions if func.accuracy is not None]
        
        # Usage statistics | 使用统计
        usage_counts = [func.usage_count for func in valid_functions if func.usage_count is not None]
        
        return {
            'total': len(valid_functions),
            'feature_dim': self.feature_matrix.shape[1] if self.feature_matrix is not None else 0,
            'tag_distribution': tag_count,
            'source_distribution': source_count,
            'accuracy': {
                'mean': float(np.mean(accuracies)) if accuracies else 0,
                'min': float(np.min(accuracies)) if accuracies else 0,
                'max': float(np.max(accuracies)) if accuracies else 0
            },
            'usage_stats': {
                'total_uses': sum(usage_counts),
                'avg_uses': float(np.mean(usage_counts)) if usage_counts else 0,
                'max_uses': float(np.max(usage_counts)) if usage_counts else 0
            }
        }
    
    def save(self, filename: Optional[str] = None) -> bool:
        """
        Save memory bank to file | 保存记忆库到文件
        
        Args | 参数:
            filename: Output filename (optional) | 输出文件名（可选）
            
        Returns | 返回:
            bool: True if successful | 是否成功
        """
        try:
            if filename is None:
                filename = "function_memory_backup.json"
            
            valid_functions = [func for func in self.functions if func is not None]
            data = [func.to_dict() for func in valid_functions]
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self._logger.info(f"Memory bank saved to {filename} | 记忆库已保存")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to save: {e} | 保存失败")
            return False
    
    def load(self, filename: str) -> bool:
        """
        Load memory bank from file | 从文件加载记忆库
        
        Args | 参数:
            filename: Input filename | 文件名
            
        Returns | 返回:
            bool: True if successful | 是否成功
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.clear()
            
            success_count = 0
            for item in data:
                try:
                    func = FunctionMemory.from_dict(item)
                    if self.add(func):
                        success_count += 1
                except Exception as e:
                    self._logger.error(f"Failed to load function: {e} | 加载单个函数失败")
                    continue
            
            self._logger.info(f"Loaded {success_count} functions from {filename} | 加载")
            return success_count > 0
            
        except Exception as e:
            self._logger.error(f"Failed to load: {e} | 加载失败")
            return False
    
    def clear(self) -> None:
        """Clear memory bank | 清空记忆库"""
        self.functions.clear()
        self.feature_matrix = None
        self._need_rebuild_index = False
        
        try:
            self.memory.clear()
        except:
            pass
        
        self._logger.info("Memory bank cleared | 记忆库已清空")
    
    def __len__(self) -> int:
        """Return number of valid functions | 返回有效函数数量"""
        return len([f for f in self.functions if f is not None])
    
    def __iter__(self):
        """Iterate over valid functions | 迭代器，跳过None值"""
        return (f for f in self.functions if f is not None)
    
    def __getitem__(self, idx: int) -> FunctionMemory:
        """Get function by index | 获取指定索引的函数"""
        if idx < 0:
            idx = len(self.functions) + idx
        if 0 <= idx < len(self.functions) and self.functions[idx] is not None:
            return self.functions[idx]
        raise IndexError(f"Index {idx} out of range or function is None | 索引超出范围或函数为None")