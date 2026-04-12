"""
Memory Backend Interface and Implementations | 记忆后端接口与实现
===================================================================

Defines the MemoryBackend interface and concrete implementations for persistent key-value storage.
This is an auxiliary component that supports the agent's memory system.

定义记忆后端接口和具体实现，用于持久化键值存储。
这是一个辅助组件，支持代理的记忆系统。

Backend Types | 后端类型:
    - InMemoryBackend: Fastest, non-persistent (for testing) | 最快，非持久化（用于测试）
    - JSONMemoryBackend: Human-readable, file-based | 人类可读，基于文件
    - SQLiteMemoryBackend: Production-ready, ACID compliant | 生产就绪，支持 ACID

Design Philosophy | 设计理念:
    - Unified interface: Same operations for different storage backends
      统一接口：不同存储后端使用相同操作
    - Pluggable: Easy to add new backends | 可插拔：易于添加新后端
    - Observable: Provides storage statistics | 可观测：提供存储统计信息
"""

import json
import time
import pickle
import hashlib
import threading
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging


# ============================================================
# Memory Backend Interface | 记忆后端接口
# ============================================================

class MemoryBackend(ABC):
    """
    Memory backend abstract interface.
    
    记忆后端抽象接口。
    
    All concrete storage backends must implement these methods.
    This defines a unified "database interface" - regardless of whether the underlying
    storage is a JSON file, SQLite database, or in-memory dictionary.
    
    所有具体的存储后端都必须实现这些方法。
    这定义了一个统一的"数据库接口"——无论底层存储是 JSON 文件、
    SQLite 数据库还是内存字典。
    
    Implementation Requirements | 实现要求:
        - save(), load(), delete(), list_keys(), clear() MUST be implemented
          必须实现 save()、load()、delete()、list_keys()、clear()
        - contains(), get_size(), get_stats() have default implementations
          contains()、get_size()、get_stats() 有默认实现
    """
    
    @abstractmethod
    def save(self, key: str, value: Any) -> bool:
        """
        Save a key-value pair.
        
        保存键值对。
        
        Args | 参数:
            key: Memory key for later retrieval | 记忆键，用于后续检索
            value: Memory value, can be any Python object | 记忆值，可以是任意 Python 对象
            
        Returns | 返回:
            bool: True if save was successful, False otherwise
                 保存成功返回 True，否则返回 False
        """
        pass
    
    @abstractmethod
    def load(self, key: str) -> Optional[Any]:
        """
        Load a value by key.
        
        根据键加载值。
        
        Args | 参数:
            key: Memory key | 记忆键
            
        Returns | 返回:
            Optional[Any]: Stored value, None if key doesn't exist
                           存储的值，键不存在时返回 None
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete a key-value pair.
        
        删除键值对。
        
        Args | 参数:
            key: Memory key to delete | 要删除的记忆键
            
        Returns | 返回:
            bool: True if deletion was successful (key existed), False otherwise
                 删除成功（键存在）返回 True，否则返回 False
        """
        pass
    
    @abstractmethod
    def list_keys(self) -> List[str]:
        """
        List all keys.
        
        列出所有键。
        
        Returns | 返回:
            List[str]: List of all keys | 所有键的列表
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """
        Clear all data.
        
        清空所有数据。
        
        Returns | 返回:
            bool: True if clear was successful, False otherwise
                 清空成功返回 True，否则返回 False
        """
        pass
    
    def contains(self, key: str) -> bool:
        """
        Check if a key exists.
        
        检查键是否存在。
        
        Args | 参数:
            key: Memory key to check | 要检查的记忆键
            
        Returns | 返回:
            bool: True if key exists, False otherwise | 键存在返回 True，否则返回 False
        """
        return key in self.list_keys()
    
    def get_size(self) -> int:
        """
        Get number of stored records.
        
        获取存储记录数。
        
        Returns | 返回:
            int: Number of stored records | 存储的记录数
        """
        return len(self.list_keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        获取存储统计信息。
        
        Returns | 返回:
            Dict: Statistics about the backend storage | 后端存储的统计信息
        """
        return {
            'type': self.__class__.__name__,
            'size': self.get_size(),
        }


# ============================================================
# 1. In-Memory Backend | 内存后端
# ============================================================

class InMemoryBackend(MemoryBackend):
    """
    In-memory backend: Stores data only in RAM.
    
    内存后端：仅将数据存储在 RAM 中。
    
    Fastest but non-persistent - data is lost when the program exits.
    Suitable for temporary caching, session-level memory, and testing.
    
    速度最快但非持久化——程序退出时数据丢失。
    适用于临时缓存、会话级记忆和测试。
    
    Characteristics | 特性:
        - Extremely fast: Pure memory operations, no I/O overhead | 极快：纯内存操作，无 I/O 开销
        - Thread-safe: Protected by reentrant lock | 线程安全：由可重入锁保护
        - Volatile: Data lost on program exit | 易失：程序退出时数据丢失
    
    Example | 示例:
        >>> backend = InMemoryBackend()
        >>> backend.save("name", "Metathin")
        >>> value = backend.load("name")  # Returns "Metathin"
        >>> backend.save("config", {"theme": "dark", "lang": "zh"})
        >>> backend.list_keys()  # Returns ['name', 'config']
    """
    
    def __init__(self):
        """Initialize in-memory backend | 初始化内存后端"""
        self._memory: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._logger = logging.getLogger("metathin.core.memory.InMemoryBackend")
    
    def save(self, key: str, value: Any) -> bool:
        """Save to memory | 保存到内存"""
        with self._lock:
            try:
                self._memory[key] = value
                self._logger.debug(f"Saved: {key}")
                return True
            except Exception as e:
                self._logger.error(f"Failed to save {key}: {e}")
                return False
    
    def load(self, key: str) -> Optional[Any]:
        """Load from memory | 从内存加载"""
        with self._lock:
            value = self._memory.get(key)
            self._logger.debug(f"Loaded: {key} -> {'found' if value is not None else 'not found'}")
            return value
    
    def delete(self, key: str) -> bool:
        """Delete from memory | 从内存删除"""
        with self._lock:
            if key in self._memory:
                del self._memory[key]
                self._logger.debug(f"Deleted: {key}")
                return True
            self._logger.debug(f"Delete failed: {key} does not exist")
            return False
    
    def list_keys(self) -> List[str]:
        """List all keys | 列出所有键"""
        with self._lock:
            return list(self._memory.keys())
    
    def clear(self) -> bool:
        """Clear all memory | 清空所有内存"""
        with self._lock:
            self._memory.clear()
            self._logger.info("Cleared all memory")
            return True
    
    def contains(self, key: str) -> bool:
        """Check if key exists | 检查键是否存在"""
        with self._lock:
            return key in self._memory
    
    def get_size(self) -> int:
        """Get number of records | 获取记录数"""
        with self._lock:
            return len(self._memory)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics | 获取统计信息"""
        with self._lock:
            return {
                'type': self.__class__.__name__,
                'size': len(self._memory),
                'keys': list(self._memory.keys())
            }


# ============================================================
# 2. JSON File Backend | JSON 文件后端
# ============================================================

class JSONMemoryBackend(MemoryBackend):
    """
    JSON file-based memory backend.
    
    基于 JSON 文件的记忆后端。
    
    Saves memory as a JSON file. Human-readable, suitable for small amounts of data.
    
    将记忆保存为 JSON 文件。人类可读，适用于少量数据。
    
    Characteristics | 特性:
        - Readability: JSON format, can be viewed and edited directly | 可读性：JSON 格式，可直接查看和编辑
        - Persistent: Data stored in the file system | 持久化：数据存储在文件系统中
        - Atomic operations: Uses temporary file + rename to ensure data integrity
          原子操作：使用临时文件 + 重命名确保数据完整性
        - Automatic serialization: Supports Python basic types, complex objects converted to strings
          自动序列化：支持 Python 基本类型，复杂对象转换为字符串
    
    Suitable Scenarios | 适用场景:
        - Configuration storage | 配置存储
        - Small database (< 1000 records) | 小数据库（< 1000 条记录）
        - Data that needs manual viewing or editing | 需要手动查看或编辑的数据
    
    Attributes | 属性:
        filepath: JSON file path | JSON 文件路径
        auto_save: Whether to automatically save after each modification | 是否在每次修改后自动保存
    
    Example | 示例:
        >>> backend = JSONMemoryBackend("my_memory.json")
        >>> backend.save("user", {"name": "Alice", "age": 30})
        >>> backend.save("preferences", {"theme": "dark"})
        >>> backend.save("list_data", [1, 2, 3, 4, 5])
        >>> # All data will be automatically saved to my_memory.json
        >>> # 所有数据将自动保存到 my_memory.json
    """
    
    def __init__(self, 
                 filepath: Union[str, Path] = "metathin_memory.json",
                 auto_save: bool = True,
                 encoding: str = 'utf-8'):
        """
        Initialize JSON memory backend.
        
        初始化 JSON 记忆后端。
        
        Args | 参数:
            filepath: JSON file path | JSON 文件路径
            auto_save: Whether to automatically save after each modification
                       是否在每次修改后自动保存
                      True: Immediately write to file after each save/delete/clear
                            每次保存/删除/清空后立即写入文件
                      False: Need to manually call flush() to save
                            需要手动调用 flush() 保存
            encoding: File encoding, default utf-8 | 文件编码，默认 utf-8
        """
        super().__init__()
        
        self.filepath = Path(filepath)
        self.auto_save = auto_save
        self.encoding = encoding
        self._lock = threading.RLock()
        self._logger = logging.getLogger("metathin.core.memory.JSONMemoryBackend")
        
        # In-memory cache | 内存缓存
        self._memory: Dict[str, Any] = {}
        
        # Load existing data | 加载现有数据
        self._load()
    
    def _load(self) -> None:
        """Load memory from file | 从文件加载记忆"""
        with self._lock:
            if self.filepath.exists():
                try:
                    with open(self.filepath, 'r', encoding=self.encoding) as f:
                        self._memory = json.load(f)
                    self._logger.info(f"Loaded {len(self._memory)} memories from {self.filepath}")
                except json.JSONDecodeError as e:
                    self._logger.error(f"JSON parsing failed: {e}")
                    self._memory = {}
                except Exception as e:
                    self._logger.error(f"Load failed: {e}")
                    self._memory = {}
            else:
                self._logger.debug(f"File does not exist: {self.filepath}")
                self._memory = {}
    
    def _save(self) -> bool:
        """
        Save memory to file.
        
        保存记忆到文件。
        
        Uses atomic operation to ensure data integrity:
        使用原子操作确保数据完整性：
            1. First write to temporary file | 首先写入临时文件
            2. Delete original file (if exists) | 删除原始文件（如果存在）
            3. Rename temporary file to original filename | 将临时文件重命名为原始文件名
        
        This ensures that even if the program crashes during writing,
        the original file won't be corrupted.
        
        这确保了即使在写入过程中程序崩溃，原始文件也不会损坏。
        """
        with self._lock:
            try:
                # Ensure directory exists | 确保目录存在
                self.filepath.parent.mkdir(parents=True, exist_ok=True)
                
                # Write to temporary file then rename to avoid corruption
                # 写入临时文件然后重命名以避免损坏
                temp_path = self.filepath.with_suffix('.tmp')
                with open(temp_path, 'w', encoding=self.encoding) as f:
                    json.dump(self._memory, f, indent=2, ensure_ascii=False, default=str)
                
                # Windows requires deleting the target file first | Windows 需要先删除目标文件
                if self.filepath.exists():
                    self.filepath.unlink()
                
                temp_path.rename(self.filepath)
                
                self._logger.debug(f"Saved to {self.filepath}")
                return True
                
            except Exception as e:
                self._logger.error(f"Save failed: {e}")
                return False
    
    def save(self, key: str, value: Any) -> bool:
        """
        Save memory.
        
        保存记忆。
        
        If auto_save=True, immediately writes to file.
        
        如果 auto_save=True，立即写入文件。
        
        Args | 参数:
            key: Memory key | 记忆键
            value: Memory value | 记忆值
            
        Returns | 返回:
            bool: Whether successful | 是否成功
        """
        with self._lock:
            try:
                # Handle objects that are not JSON serializable | 处理不可 JSON 序列化的对象
                if not self._is_json_serializable(value):
                    value = self._make_json_serializable(value)
                
                self._memory[key] = value
                
                if self.auto_save:
                    return self._save()
                return True
                
            except Exception as e:
                self._logger.error(f"Failed to save {key}: {e}")
                return False
    
    def _is_json_serializable(self, obj: Any) -> bool:
        """
        Check if object is JSON serializable.
        
        检查对象是否可 JSON 序列化。
        
        JSON natively supports | JSON 原生支持:
            - str, int, float, bool, None
            - list, tuple (converted to list) | (转换为列表)
            - dict (keys must be strings) | (键必须是字符串)
        
        Args | 参数:
            obj: Object to check | 要检查的对象
            
        Returns | 返回:
            bool: Whether object can be directly JSON serialized | 对象是否可直接 JSON 序列化
        """
        try:
            json.dumps(obj)
            return True
        except:
            return False
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON serializable form.
        
        将对象转换为 JSON 可序列化形式。
        
        Recursively handles various types:
        递归处理各种类型：
            - Basic types: Keep as is | 基本类型：保持不变
            - Lists/tuples: Recursively process each element | 列表/元组：递归处理每个元素
            - Dictionaries: Recursively process each value, convert keys to strings
              字典：递归处理每个值，将键转换为字符串
            - numpy arrays: Convert to list | numpy 数组：转换为列表
            - datetime: Convert to ISO format string | 转换为 ISO 格式字符串
            - Custom objects: Convert to dictionary or string | 自定义对象：转换为字典或字符串
        
        Args | 参数:
            obj: Object to convert | 要转换的对象
            
        Returns | 返回:
            Any: JSON serializable object | JSON 可序列化对象
        """
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(k): self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return str(obj)
    
    def load(self, key: str) -> Optional[Any]:
        """Load memory | 加载记忆"""
        with self._lock:
            return self._memory.get(key)
    
    def delete(self, key: str) -> bool:
        """Delete memory | 删除记忆"""
        with self._lock:
            if key in self._memory:
                del self._memory[key]
                if self.auto_save:
                    return self._save()
                return True
            return False
    
    def list_keys(self) -> List[str]:
        """List all keys | 列出所有键"""
        with self._lock:
            return list(self._memory.keys())
    
    def clear(self) -> bool:
        """Clear all memory | 清空所有记忆"""
        with self._lock:
            self._memory.clear()
            if self.auto_save:
                return self._save()
            return True
    
    def flush(self) -> bool:
        """Force save to file | 强制保存到文件"""
        return self._save()
    
    def get_size(self) -> int:
        """Get number of records | 获取记录数"""
        with self._lock:
            return len(self._memory)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics | 获取统计信息"""
        stats = super().get_stats()
        stats.update({
            'filepath': str(self.filepath),
            'file_exists': self.filepath.exists(),
            'file_size': self.filepath.stat().st_size if self.filepath.exists() else 0,
            'auto_save': self.auto_save,
        })
        return stats
    
    def __repr__(self) -> str:
        return f"JSONMemoryBackend(file='{self.filepath.name}', size={self.get_size()})"


# ============================================================
# 3. SQLite Backend | SQLite 后端
# ============================================================

class SQLiteMemoryBackend(MemoryBackend):
    """
    SQLite-based memory backend.
    
    基于 SQLite 的记忆后端。
    
    Uses SQLite database for storage, supporting large amounts of data and transactions.
    
    使用 SQLite 数据库进行存储，支持大量数据和事务。
    
    Characteristics | 特性:
        - High performance: Suitable for large datasets (10k+ records) | 高性能：适合大数据集（1万+ 记录）
        - Transaction support: Ensures data consistency | 事务支持：确保数据一致性
        - Type storage: Uses pickle to serialize arbitrary Python objects | 类型存储：使用 pickle 序列化任意 Python 对象
        - Index optimization: Indexed by update time, supports time-based queries
          索引优化：按更新时间索引，支持基于时间的查询
    
    Suitable Scenarios | 适用场景:
        - Large-scale memory storage | 大规模记忆存储
        - Scenarios requiring transaction support | 需要事务支持的场景
        - Memories needing time-based queries | 需要基于时间查询的记忆
        - Long-running services | 长时间运行的服务
    
    Attributes | 属性:
        db_path: Database file path | 数据库文件路径
        table_name: Table name | 表名
        auto_commit: Whether to automatically commit transactions | 是否自动提交事务
    """
    
    def __init__(self, 
                 db_path: Union[str, Path] = "metathin_memory.db",
                 table_name: str = "memory",
                 auto_commit: bool = True):
        """
        Initialize SQLite memory backend.
        
        初始化 SQLite 记忆后端。
        
        Args | 参数:
            db_path: SQLite database file path | SQLite 数据库文件路径
            table_name: Database table name | 数据库表名
            auto_commit: Whether to automatically commit transactions | 是否自动提交事务
                        True: Auto-commit after each operation | 每次操作后自动提交
                        False: Need to manually commit | 需要手动提交
        """
        super().__init__()
        
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.auto_commit = auto_commit
        self._lock = threading.RLock()
        self._logger = logging.getLogger("metathin.core.memory.SQLiteMemoryBackend")
        
        # Initialize database | 初始化数据库
        self._init_db()
    
    def _get_connection(self):
        """
        Get database connection.
        
        获取数据库连接。
        
        Returns | 返回:
            sqlite3.Connection: Database connection object | 数据库连接对象
            
        Note: Creates a new connection each time to ensure thread safety
              每次创建新连接以确保线程安全
        """
        import sqlite3
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row  # Enable column name access | 启用列名访问
        return conn
    
    def _init_db(self):
        """Initialize database table | 初始化数据库表"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Create table | 创建表
                cursor.execute(f'''
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        key TEXT PRIMARY KEY,
                        value BLOB NOT NULL,
                        value_type TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        metadata TEXT
                    )
                ''')
                
                # Create index | 创建索引
                cursor.execute(f'''
                    CREATE INDEX IF NOT EXISTS idx_updated_at 
                    ON {self.table_name}(updated_at)
                ''')
                
                conn.commit()
                
            self._logger.info(f"Initialized database: {self.db_path}")
            
        except Exception as e:
            self._logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _serialize(self, value: Any) -> Tuple[bytes, str]:
        """
        Serialize value.
        
        序列化值。
        
        Uses pickle to convert arbitrary Python objects to byte stream.
        
        使用 pickle 将任意 Python 对象转换为字节流。
        
        Args | 参数:
            value: Value to serialize | 要序列化的值
            
        Returns | 返回:
            Tuple[bytes, str]: (Serialized bytes, type identifier) | (序列化后的字节, 类型标识符)
        """
        # Record type | 记录类型
        value_type = type(value).__name__
        
        # Serialize | 序列化
        return pickle.dumps(value), value_type
    
    def _deserialize(self, data: bytes) -> Any:
        """
        Deserialize value.
        
        反序列化值。
        
        Restores Python object from pickle byte stream.
        
        从 pickle 字节流恢复 Python 对象。
        
        Args | 参数:
            data: Serialized bytes | 序列化后的字节
            
        Returns | 返回:
            Any: Deserialized value, None if failed | 反序列化后的值，失败时返回 None
        """
        try:
            return pickle.loads(data)
        except Exception as e:
            self._logger.error(f"Deserialization failed: {e}")
            return None
    
    def save(self, key: str, value: Any) -> bool:
        """
        Save memory.
        
        保存记忆。
        
        Uses INSERT OR REPLACE - updates if key exists, inserts if not.
        
        使用 INSERT OR REPLACE - 如果键存在则更新，否则插入。
        
        Args | 参数:
            key: Memory key | 记忆键
            value: Memory value | 记忆值
            
        Returns | 返回:
            bool: Whether successful | 是否成功
        """
        with self._lock:
            try:
                # Serialize | 序列化
                serialized, value_type = self._serialize(value)
                now = time.time()
                
                # Insert or replace | 插入或替换
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(f'''
                        INSERT OR REPLACE INTO {self.table_name} 
                        (key, value, value_type, created_at, updated_at, metadata)
                        VALUES (?, ?, ?, 
                                COALESCE((SELECT created_at FROM {self.table_name} WHERE key = ?), ?),
                                ?, ?)
                    ''', (
                        key, serialized, value_type,
                        key, now,  # If exists, keep original created_at | 如果存在，保留原始 created_at
                        now,        # updated_at
                        json.dumps({})  # metadata
                    ))
                    
                    if self.auto_commit:
                        conn.commit()
                
                self._logger.debug(f"Saved: {key} ({value_type})")
                return True
                
            except Exception as e:
                self._logger.error(f"Failed to save {key}: {e}")
                return False
    
    def load(self, key: str) -> Optional[Any]:
        """
        Load memory.
        
        加载记忆。
        
        Args | 参数:
            key: Memory key | 记忆键
            
        Returns | 返回:
            Optional[Any]: Memory value, None if not found | 记忆值，未找到时返回 None
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f'SELECT value FROM {self.table_name} WHERE key = ?',
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    value = self._deserialize(row[0])
                    self._logger.debug(f"Loaded: {key} -> {'success' if value is not None else 'failed'}")
                    return value
                
            self._logger.debug(f"Loaded: {key} -> not found")
            return None
            
        except Exception as e:
            self._logger.error(f"Failed to load {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """
        Delete memory.
        
        删除记忆。
        
        Args | 参数:
            key: Memory key | 记忆键
            
        Returns | 返回:
            bool: Whether deletion was successful | 删除是否成功
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f'DELETE FROM {self.table_name} WHERE key = ?',
                    (key,)
                )
                deleted = cursor.rowcount > 0
                
                if self.auto_commit:
                    conn.commit()
                
                if deleted:
                    self._logger.debug(f"Deleted: {key}")
                
                return deleted
                
        except Exception as e:
            self._logger.error(f"Failed to delete {key}: {e}")
            return False
    
    def list_keys(self) -> List[str]:
        """List all keys | 列出所有键"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f'SELECT key FROM {self.table_name} ORDER BY updated_at DESC')
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self._logger.error(f"Failed to list keys: {e}")
            return []
    
    def clear(self) -> bool:
        """Clear all memory | 清空所有记忆"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f'DELETE FROM {self.table_name}')
                deleted = cursor.rowcount
                
                if self.auto_commit:
                    conn.commit()
                
                self._logger.info(f"Cleared {deleted} memories")
                return True
                
        except Exception as e:
            self._logger.error(f"Failed to clear: {e}")
            return False
    
    def get_size(self) -> int:
        """Get number of records | 获取记录数"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f'SELECT COUNT(*) FROM {self.table_name}')
                return cursor.fetchone()[0]
        except Exception as e:
            self._logger.error(f"Failed to get size: {e}")
            return 0
    
    def get_by_age(self, max_age_seconds: float) -> List[Tuple[str, Any]]:
        """
        Get memories within specified time range.
        
        获取指定时间范围内的记忆。
        
        Filters recent memories by update time.
        
        按更新时间过滤最近的记忆。
        
        Args | 参数:
            max_age_seconds: Maximum age in seconds (e.g., 3600 for last hour)
                             最大年龄（秒）（例如，3600 表示最后一小时）
            
        Returns | 返回:
            List[Tuple[str, Any]]: List of (key, value) tuples | (键, 值) 元组列表
        """
        try:
            cutoff = time.time() - max_age_seconds
            results = []
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f'SELECT key, value FROM {self.table_name} WHERE updated_at >= ?',
                    (cutoff,)
                )
                
                for row in cursor.fetchall():
                    value = self._deserialize(row[1])
                    if value is not None:
                        results.append((row[0], value))
            
            return results
            
        except Exception as e:
            self._logger.error(f"Age query failed: {e}")
            return []
    
    def vacuum(self) -> bool:
        """
        Compact database.
        
        压缩数据库。
        
        Reclaims space occupied by deleted records, reducing file size.
        Recommended for periodic maintenance, especially in scenarios with frequent deletions.
        
        回收已删除记录占用的空间，减小文件大小。
        建议定期维护，特别是在频繁删除的场景中。
        
        Returns | 返回:
            bool: Whether successful | 是否成功
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('VACUUM')
                self._logger.info("Database vacuum completed")
                return True
        except Exception as e:
            self._logger.error(f"Vacuum failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics | 获取统计信息"""
        stats = super().get_stats()
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get database size | 获取数据库大小
                cursor.execute('SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()')
                db_size = cursor.fetchone()[0]
                
                # Get type statistics | 获取类型统计
                cursor.execute(f'''
                    SELECT value_type, COUNT(*) as count 
                    FROM {self.table_name} 
                    GROUP BY value_type
                ''')
                type_stats = {row[0]: row[1] for row in cursor.fetchall()}
                
                stats.update({
                    'db_path': str(self.db_path),
                    'db_size': db_size,
                    'type_stats': type_stats,
                    'auto_commit': self.auto_commit,
                })
                
        except Exception as e:
            self._logger.error(f"Failed to get stats: {e}")
        
        return stats
    
    def __repr__(self) -> str:
        return f"SQLiteMemoryBackend(db='{self.db_path.name}', size={self.get_size()})"


# ============================================================
# Import numpy for type checking (lazy) | 导入 numpy 用于类型检查（延迟）
# ============================================================

try:
    import numpy as np
except ImportError:
    np = None


# ============================================================
# Import datetime for type checking (lazy) | 导入 datetime 用于类型检查（延迟）
# ============================================================

from datetime import datetime


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    'MemoryBackend',       # Base interface | 基础接口
    'InMemoryBackend',     # In-memory implementation | 内存实现
    'JSONMemoryBackend',   # JSON file implementation | JSON 文件实现
    'SQLiteMemoryBackend', # SQLite implementation | SQLite 实现
]