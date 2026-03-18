"""
Metathin Memory Module
=====================

Provides persistent memory capabilities for the agent, supporting multiple storage backends.
Memory serves as the agent's "long-term storage," enabling information retention across thinking cycles.

Design Philosophy:
    - Multi-level Storage: Memory cache + persistent backend, balancing speed and durability
    - Pluggable: Supports multiple storage backends (JSON, SQLite, etc.), flexible for different scenarios
    - Efficient: Cache mechanism reduces I/O operations, improving access speed
    - Reliable: Comprehensive error handling and transaction support ensure data consistency

The memory manager provides key-value storage, supporting both temporary and permanent memory.
Temporary memory exists only in memory cache, while permanent memory is persisted to the backend.

Use Cases:
    - Remember user preferences
    - Cache computation results to avoid recalculation
    - Save agent's learning experience
    - Maintain state across sessions
"""

import json
import time
import pickle
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
import threading

import numpy as np


# ============================================================
# Memory Backend Interface
# ============================================================

class MemoryBackend(ABC):
    """
    Memory backend abstract interface.
    
    All concrete storage backends must implement these methods.
    This defines a unified "database interface" - regardless of whether the underlying
    storage is a JSON file or SQLite, the upper layer can operate in the same way.
    
    Methods:
        save: Save a key-value pair
        load: Load a value by key
        delete: Delete a key
        list_keys: List all keys
        clear: Clear all data
        contains: Check if key exists
        get_size: Get storage size
        
    Extension Methods (optional implementation):
        flush: Force flush cache
        vacuum: Compact storage space
        get_stats: Get statistics
    """
    
    @abstractmethod
    def save(self, key: str, value: Any) -> bool:
        """
        Save memory.
        
        Stores a key-value pair in the backend. If the key already exists,
        it typically overwrites the old value.
        
        Args:
            key: Memory key for later retrieval
            value: Memory value, can be any Python object
            
        Returns:
            bool: Whether save was successful
        """
        pass
    
    @abstractmethod
    def load(self, key: str) -> Optional[Any]:
        """
        Load memory.
        
        Retrieves previously stored value by key from the backend.
        
        Args:
            key: Memory key
            
        Returns:
            Optional[Any]: Memory value, None if key doesn't exist
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete memory.
        
        Removes the memory entry for the specified key from the backend.
        
        Args:
            key: Memory key to delete
            
        Returns:
            bool: Whether deletion was successful (True if key existed, False otherwise)
        """
        pass
    
    @abstractmethod
    def list_keys(self) -> List[str]:
        """
        List all keys.
        
        Returns a list of all keys stored in the backend.
        
        Returns:
            List[str]: List of keys, typically in some order (often reverse chronological)
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """
        Clear all memory.
        
        Deletes all key-value pairs stored in the backend.
        
        Returns:
            bool: Whether clear was successful
        """
        pass
    
    def contains(self, key: str) -> bool:
        """
        Check if key exists.
        
        Determines whether the specified key exists in the backend.
        
        Args:
            key: Memory key to check
            
        Returns:
            bool: Whether the key exists
        """
        return key in self.list_keys()
    
    def get_size(self) -> int:
        """
        Get number of stored records.
        
        Returns:
            int: Number of stored records
        """
        return len(self.list_keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dict: Statistics about the backend storage
        """
        return {
            'type': self.__class__.__name__,
            'size': self.get_size(),
        }


# ============================================================
# In-Memory Backend
# ============================================================

class InMemoryBackend(MemoryBackend):
    """
    In-memory backend: Stores data only in RAM.
    
    Fastest but non-persistent - data is lost when the program exits.
    Suitable for temporary caching, session-level memory, and other scenarios
    that don't require persistence.
    
    Characteristics:
        - Extremely fast: Pure memory operations, no I/O overhead
        - Thread-safe: Protected by reentrant lock
        - Volatile: Data lost on program exit
        
    Example:
        >>> backend = InMemoryBackend()
        >>> backend.save("name", "Metathin")
        >>> value = backend.load("name")  # Returns "Metathin"
        >>> backend.save("config", {"theme": "dark", "lang": "zh"})
        >>> backend.list_keys()  # Returns ['name', 'config']
    """
    
    def __init__(self):
        """Initialize in-memory backend."""
        self._memory: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._logger = logging.getLogger("metathin.memory.InMemoryBackend")
    
    def save(self, key: str, value: Any) -> bool:
        """Save to memory."""
        with self._lock:
            try:
                self._memory[key] = value
                self._logger.debug(f"Saved: {key}")
                return True
            except Exception as e:
                self._logger.error(f"Failed to save {key}: {e}")
                return False
    
    def load(self, key: str) -> Optional[Any]:
        """Load from memory."""
        with self._lock:
            value = self._memory.get(key)
            self._logger.debug(f"Loaded: {key} -> {'found' if value is not None else 'not found'}")
            return value
    
    def delete(self, key: str) -> bool:
        """Delete from memory."""
        with self._lock:
            if key in self._memory:
                del self._memory[key]
                self._logger.debug(f"Deleted: {key}")
                return True
            self._logger.debug(f"Delete failed: {key} does not exist")
            return False
    
    def list_keys(self) -> List[str]:
        """List all keys."""
        with self._lock:
            return list(self._memory.keys())
    
    def clear(self) -> bool:
        """Clear all memory."""
        with self._lock:
            self._memory.clear()
            self._logger.info("Cleared all memory")
            return True
    
    def contains(self, key: str) -> bool:
        """Check if key exists."""
        with self._lock:
            return key in self._memory
    
    def get_size(self) -> int:
        """Get number of records."""
        with self._lock:
            return len(self._memory)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        with self._lock:
            return {
                'type': self.__class__.__name__,
                'size': len(self._memory),
                'keys': list(self._memory.keys())
            }


# ============================================================
# JSON File Backend
# ============================================================

class JSONMemoryBackend(MemoryBackend):
    """
    JSON file-based memory backend.
    
    Saves memory as a JSON file. Human-readable, suitable for small amounts of data.
    
    Characteristics:
        - Readability: JSON format, can be viewed and edited directly
        - Persistent: Data stored in the file system
        - Atomic operations: Uses temporary file + rename to ensure data integrity
        - Automatic serialization: Supports Python basic types, complex objects converted to strings
        
    Suitable Scenarios:
        - Configuration storage
        - Small database (< 1000 records)
        - Data that needs manual viewing or editing
        
    Attributes:
        filepath: JSON file path
        auto_save: Whether to automatically save after each modification
        
    Example:
        >>> backend = JSONMemoryBackend("my_memory.json")
        >>> backend.save("user", {"name": "Alice", "age": 30})
        >>> backend.save("preferences", {"theme": "dark"})
        >>> backend.save("list_data", [1, 2, 3, 4, 5])
        >>> # All data will be automatically saved to my_memory.json
    """
    
    def __init__(self, 
                 filepath: Union[str, Path] = "metathin_memory.json",
                 auto_save: bool = True,
                 encoding: str = 'utf-8'):
        """
        Initialize JSON memory backend.
        
        Args:
            filepath: JSON file path
            auto_save: Whether to automatically save after each modification
                      True: Immediately write to file after each save/delete/clear
                      False: Need to manually call flush() to save
            encoding: File encoding, default utf-8
        """
        super().__init__()
        
        self.filepath = Path(filepath)
        self.auto_save = auto_save
        self.encoding = encoding
        self._lock = threading.RLock()
        self._logger = logging.getLogger("metathin.memory.JSONMemoryBackend")
        
        # In-memory cache
        self._memory: Dict[str, Any] = {}
        
        # Load existing data
        self._load()
    
    def _load(self) -> None:
        """Load memory from file."""
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
        
        Uses atomic operation to ensure data integrity:
        1. First write to temporary file
        2. Delete original file (if exists)
        3. Rename temporary file to original filename
        
        This ensures that even if the program crashes during writing,
        the original file won't be corrupted.
        """
        with self._lock:
            try:
                # Ensure directory exists
                self.filepath.parent.mkdir(parents=True, exist_ok=True)
                
                # Write to temporary file then rename to avoid corruption
                temp_path = self.filepath.with_suffix('.tmp')
                with open(temp_path, 'w', encoding=self.encoding) as f:
                    json.dump(self._memory, f, indent=2, ensure_ascii=False, default=str)
                
                # Windows requires deleting the target file first
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
        
        If auto_save=True, immediately writes to file.
        
        Args:
            key: Memory key
            value: Memory value
            
        Returns:
            bool: Whether successful
        """
        with self._lock:
            try:
                # Handle objects that are not JSON serializable
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
        
        JSON natively supports:
            - str, int, float, bool, None
            - list, tuple (converted to list)
            - dict (keys must be strings)
        
        Args:
            obj: Object to check
            
        Returns:
            bool: Whether object can be directly JSON serialized
        """
        try:
            json.dumps(obj)
            return True
        except:
            return False
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON serializable form.
        
        Recursively handles various types:
            - Basic types: Keep as is
            - Lists/tuples: Recursively process each element
            - Dictionaries: Recursively process each value, convert keys to strings
            - numpy arrays: Convert to list
            - datetime: Convert to ISO format string
            - Custom objects: Convert to dictionary or string
        
        Args:
            obj: Object to convert
            
        Returns:
            Any: JSON serializable object
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
        """Load memory."""
        with self._lock:
            return self._memory.get(key)
    
    def delete(self, key: str) -> bool:
        """Delete memory."""
        with self._lock:
            if key in self._memory:
                del self._memory[key]
                if self.auto_save:
                    return self._save()
                return True
            return False
    
    def list_keys(self) -> List[str]:
        """List all keys."""
        with self._lock:
            return list(self._memory.keys())
    
    def clear(self) -> bool:
        """Clear all memory."""
        with self._lock:
            self._memory.clear()
            if self.auto_save:
                return self._save()
            return True
    
    def flush(self) -> bool:
        """Force save to file."""
        return self._save()
    
    def get_size(self) -> int:
        """Get number of records."""
        with self._lock:
            return len(self._memory)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
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
# SQLite Backend
# ============================================================

class SQLiteMemoryBackend(MemoryBackend):
    """
    SQLite-based memory backend.
    
    Uses SQLite database for storage, supporting large amounts of data and transactions.
    
    Characteristics:
        - High performance: Suitable for large datasets (10k+ records)
        - Transaction support: Ensures data consistency
        - Type storage: Uses pickle to serialize arbitrary Python objects
        - Index optimization: Indexed by update time, supports time-based queries
        
    Suitable Scenarios:
        - Large-scale memory storage
        - Scenarios requiring transaction support
        - Memories needing time-based queries
        - Long-running services
    
    Attributes:
        db_path: Database file path
        table_name: Table name
        auto_commit: Whether to automatically commit transactions
        
    Example:
        >>> backend = SQLiteMemoryBackend("metathin.db")
        >>> backend.save("key1", {"complex": "data"})
        >>> backend.save("key2", [1, 2, 3, 4, 5])
        >>> # Get memories from last hour
        >>> recent = backend.get_by_age(3600)
    """
    
    def __init__(self, 
                 db_path: Union[str, Path] = "metathin_memory.db",
                 table_name: str = "memory",
                 auto_commit: bool = True):
        """
        Initialize SQLite memory backend.
        
        Args:
            db_path: SQLite database file path
            table_name: Database table name
            auto_commit: Whether to automatically commit transactions
                        True: Auto-commit after each operation
                        False: Need to manually commit
        """
        super().__init__()
        
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.auto_commit = auto_commit
        self._lock = threading.RLock()
        self._logger = logging.getLogger("metathin.memory.SQLiteMemoryBackend")
        
        # Initialize database
        self._init_db()
    
    def _get_connection(self):
        """
        Get database connection.
        
        Returns:
            sqlite3.Connection: Database connection object
            
        Note: Creates a new connection each time to ensure thread safety
        """
        import sqlite3
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row  # Enable column name access
        return conn
    
    def _init_db(self):
        """Initialize database table."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Create table
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
                
                # Create index
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
        
        Uses pickle to convert arbitrary Python objects to byte stream.
        
        Args:
            value: Value to serialize
            
        Returns:
            Tuple[bytes, str]: (Serialized bytes, type identifier)
        """
        # Record type
        value_type = type(value).__name__
        
        # Serialize
        return pickle.dumps(value), value_type
    
    def _deserialize(self, data: bytes) -> Any:
        """
        Deserialize value.
        
        Restores Python object from pickle byte stream.
        
        Args:
            data: Serialized bytes
            
        Returns:
            Any: Deserialized value, None if failed
        """
        try:
            return pickle.loads(data)
        except Exception as e:
            self._logger.error(f"Deserialization failed: {e}")
            return None
    
    def save(self, key: str, value: Any) -> bool:
        """
        Save memory.
        
        Uses INSERT OR REPLACE - updates if key exists, inserts if not.
        
        Args:
            key: Memory key
            value: Memory value
            
        Returns:
            bool: Whether successful
        """
        with self._lock:
            try:
                # Serialize
                serialized, value_type = self._serialize(value)
                now = time.time()
                
                # Insert or replace
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
                        key, now,  # If exists, keep original created_at
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
        
        Args:
            key: Memory key
            
        Returns:
            Optional[Any]: Memory value, None if not found
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
        
        Args:
            key: Memory key
            
        Returns:
            bool: Whether deletion was successful
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
        """List all keys."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f'SELECT key FROM {self.table_name} ORDER BY updated_at DESC')
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self._logger.error(f"Failed to list keys: {e}")
            return []
    
    def clear(self) -> bool:
        """Clear all memory."""
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
        """Get number of records."""
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
        
        Filters recent memories by update time.
        
        Args:
            max_age_seconds: Maximum age in seconds (e.g., 3600 for last hour)
            
        Returns:
            List[Tuple[str, Any]]: List of (key, value) tuples
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
        
        Reclaims space occupied by deleted records, reducing file size.
        Recommended for periodic maintenance, especially in scenarios with frequent deletions.
        
        Returns:
            bool: Whether successful
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
        """Get statistics."""
        stats = super().get_stats()
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get database size
                cursor.execute('SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()')
                db_size = cursor.fetchone()[0]
                
                # Get type statistics
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
# Memory Manager
# ============================================================

class MemoryManager:
    """
    Memory Manager - Provides caching and hierarchical memory.
    
    Manages short-term memory (memory cache) and long-term memory (persistent backend).
    Provides a unified interface, automatically handling caching and persistence.
    
    Design Philosophy:
        - Two-level storage: Memory cache (fast) + Persistent backend (reliable)
        - Transparent access: Upper layer doesn't need to know where data resides
        - LRU cache: Automatically evicts least recently used cache items
        - Statistics monitoring: Provides cache hit rate and other performance metrics
    
    Attributes:
        backend: Persistent backend
        enable_cache: Whether to enable memory cache
        cache_size: Cache size limit
        
    Workflow:
        1. remember(key, value, permanent=True):
           - Save to cache
           - If permanent=True, also save to backend
           
        2. recall(key):
           - Check cache first, return if hit (cache hit)
           - If cache miss, check backend (cache miss)
           - Update cache if found in backend
           
        3. forget(key):
           - Delete from cache
           - If permanent=True, also delete from backend
    
    Example:
        >>> # Using JSON backend
        >>> manager = MemoryManager(JSONMemoryBackend("memory.json"))
        >>> 
        >>> # Remember permanently
        >>> manager.remember("user_pref", {"theme": "dark"}, permanent=True)
        >>> 
        >>> # Temporary memory (cache only)
        >>> manager.remember("temp", "temporary", permanent=False)
        >>> 
        >>> # Recall
        >>> pref = manager.recall("user_pref")
        >>> 
        >>> # Statistics
        >>> stats = manager.get_stats()
        >>> print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
    """
    
    def __init__(self, 
                 backend: Optional[MemoryBackend] = None,
                 enable_cache: bool = True,
                 cache_size: Optional[int] = 1000):
        """
        Initialize memory manager.
        
        Args:
            backend: Persistent backend, defaults to JSON backend
            enable_cache: Whether to enable memory cache
            cache_size: Cache size limit, None means unlimited
        """
        self.backend = backend or JSONMemoryBackend()
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        
        # Memory cache
        self._cache: Dict[str, Any] = {}
        self._cache_timestamp: Dict[str, float] = {}
        
        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._backend_writes = 0
        self._backend_reads = 0
        
        self._lock = threading.RLock()
        self._logger = logging.getLogger("metathin.memory.MemoryManager")
        
        self._logger.info(f"Initialized memory manager: backend={type(self.backend).__name__}, cache={enable_cache}")
    
    def _clean_cache(self):
        """
        Clean oldest cache items.
        
        When cache exceeds size limit, delete oldest items (LRU strategy).
        Sorted by timestamp, keeps the most recent cache_size items.
        """
        if self.cache_size is None or len(self._cache) <= self.cache_size:
            return
        
        # Sort by timestamp, delete oldest
        items = sorted(self._cache_timestamp.items(), key=lambda x: x[1])
        to_remove = len(self._cache) - self.cache_size
        
        for key, _ in items[:to_remove]:
            if key in self._cache:
                del self._cache[key]
            if key in self._cache_timestamp:
                del self._cache_timestamp[key]
        
        self._logger.debug(f"Cleaned {to_remove} cache items")
    
    def remember(self, key: str, value: Any, permanent: bool = True, ttl: Optional[float] = None) -> bool:
        """
        Remember information.
        
        Args:
            key: Memory key
            value: Memory value
            permanent: Whether to save permanently (save to backend)
            ttl: Time-to-live in seconds (cache only)
            
        Returns:
            bool: Whether successful
        """
        with self._lock:
            success = True
            
            # Save to cache
            if self.enable_cache:
                self._cache[key] = value
                self._cache_timestamp[key] = time.time()
                self._clean_cache()
                self._logger.debug(f"Cache saved: {key}")
            
            # Save permanently
            if permanent:
                try:
                    backend_success = self.backend.save(key, value)
                    if backend_success:
                        self._backend_writes += 1
                        self._logger.debug(f"Backend saved: {key}")
                    else:
                        self._logger.warning(f"Backend save failed: {key}")
                        success = False
                except Exception as e:
                    self._logger.error(f"Permanent save failed {key}: {e}")
                    success = False
            
            self._logger.debug(f"Remember: {key} (permanent={permanent}, cache={self.enable_cache})")
            
            return success
    
    def recall(self, key: str, default: Any = None, use_cache: bool = True) -> Any:
        """
        Recall information.
        
        Args:
            key: Memory key
            default: Default value
            use_cache: Whether to use cache
            
        Returns:
            Any: Stored value, default if not found
        """
        with self._lock:
            # Check cache first
            if self.enable_cache and use_cache and key in self._cache:
                self._cache_hits += 1
                value = self._cache[key]
                self._logger.debug(f"Cache hit: {key} -> {value}")
                return value
            
            self._cache_misses += 1
            
            # Check backend
            try:
                value = self.backend.load(key)
                self._backend_reads += 1
                
                if value is not None:
                    # Update cache
                    if self.enable_cache:
                        self._cache[key] = value
                        self._cache_timestamp[key] = time.time()
                        self._clean_cache()
                    
                    self._logger.debug(f"Backend loaded: {key} -> {value}")
                    return value
                else:
                    self._logger.debug(f"Backend not found: {key}")
                
            except Exception as e:
                self._logger.error(f"Backend load failed {key}: {e}")
            
            self._logger.debug(f"Not found: {key}, returning default {default}")
            return default
    
    def forget(self, key: str, permanent: bool = True) -> bool:
        """
        Forget information.
        
        Args:
            key: Memory key
            permanent: Whether to also delete from backend
            
        Returns:
            bool: Whether successful
        """
        with self._lock:
            success = True
            
            # Delete from cache
            if key in self._cache:
                del self._cache[key]
            if key in self._cache_timestamp:
                del self._cache_timestamp[key]
            
            # Delete from backend
            if permanent:
                try:
                    backend_success = self.backend.delete(key)
                    if not backend_success:
                        self._logger.warning(f"Backend delete failed: {key}")
                        success = False
                except Exception as e:
                    self._logger.error(f"Backend delete failed {key}: {e}")
                    success = False
            
            self._logger.debug(f"Forget: {key} (permanent={permanent})")
            
            return success
    
    def clear(self, permanent: bool = True) -> bool:
        """
        Clear all memory.
        
        Args:
            permanent: Whether to also clear backend
            
        Returns:
            bool: Whether successful
        """
        with self._lock:
            # Clear cache
            self._cache.clear()
            self._cache_timestamp.clear()
            
            # Clear backend
            if permanent:
                try:
                    success = self.backend.clear()
                    self._logger.info("Cleared all memory (including backend)")
                    return success
                except Exception as e:
                    self._logger.error(f"Failed to clear backend: {e}")
                    return False
            
            self._logger.info("Cleared cache")
            return True
    
    def list_all(self, include_cache: bool = True) -> List[str]:
        """
        List all memory keys.
        
        Args:
            include_cache: Whether to include cache keys
            
        Returns:
            List[str]: List of keys
        """
        keys = set()
        
        # Backend keys
        try:
            if hasattr(self.backend, 'list_keys'):
                backend_keys = self.backend.list_keys()
                if backend_keys:
                    keys.update(backend_keys)
        except Exception as e:
            self._logger.error(f"Failed to list backend keys: {e}")
        
        # Cache keys
        if include_cache:
            keys.update(self._cache.keys())
        
        return sorted(list(keys))
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dict: Statistics including cache hit rate, backend reads/writes, etc.
        """
        total_cache_requests = self._cache_hits + self._cache_misses
        
        return {
            'cached': len(self._cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': self._cache_hits / max(1, total_cache_requests),
            'backend_reads': self._backend_reads,
            'backend_writes': self._backend_writes,
            'backend': self.backend.get_stats(),
            'cache_enabled': self.enable_cache,
            'cache_size_limit': self.cache_size,
        }
    
    def flush(self) -> bool:
        """
        Force flush (for supported backends).
        
        For JSON backend, force write to file;
        For SQLite backend, force commit transaction.
        
        Returns:
            bool: Whether successful
        """
        if hasattr(self.backend, 'flush'):
            return self.backend.flush()
        return True
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self.recall(key, default=None) is not None
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        value = self.recall(key)
        if value is None:
            raise KeyError(f"Key not found: {key}")
        return value
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-style assignment."""
        self.remember(key, value, permanent=True)
    
    def __delitem__(self, key: str) -> None:
        """Dictionary-style deletion."""
        if not self.forget(key):
            raise KeyError(f"Key not found: {key}")
    
    def __len__(self) -> int:
        """Return number of memories."""
        return len(self.list_all())
    
    def __repr__(self) -> str:
        return (f"MemoryManager(cache={len(self._cache)} items, "
                f"backend={type(self.backend).__name__}, "
                f"hit_rate={self._cache_hits/max(1, self._cache_hits+self._cache_misses):.2%})")


# ============================================================
# TTL Memory Manager
# ============================================================

class TTLMemoryManager(MemoryManager):
    """
    Time-To-Live Memory Manager.
    
    Automatically cleans up expired memory items.
    
    Characteristics:
        - Auto-expiration: Memory items can have a time-to-live
        - Periodic cleanup: Background automatic cleanup of expired items
        - Default TTL: Can set global default expiration time
        - On-demand query: Can query remaining TTL
    
    Attributes:
        default_ttl: Default time-to-live in seconds
        cleanup_interval: Cleanup interval in seconds
        
    Example:
        >>> manager = TTLMemoryManager(default_ttl=3600)  # 1 hour expiration
        >>> manager.remember("session", data, ttl=300)    # 5 minute expiration
        >>> 
        >>> # Get remaining time
        >>> remaining = manager.get_ttl("session")
        >>> print(f"{remaining} seconds left")
    """
    
    def __init__(self, 
                 backend: Optional[MemoryBackend] = None,
                 enable_cache: bool = True,
                 cache_size: Optional[int] = 1000,
                 default_ttl: Optional[float] = None,
                 cleanup_interval: float = 60):
        """
        Initialize TTL memory manager.
        
        Args:
            backend: Persistent backend
            enable_cache: Whether to enable cache
            cache_size: Cache size limit
            default_ttl: Default time-to-live in seconds, None means never expire
            cleanup_interval: Cleanup interval in seconds for periodic cleanup
        """
        super().__init__(backend, enable_cache, cache_size)
        
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        # Store TTL information
        self._ttl: Dict[str, float] = {}  # key -> expiry timestamp
        
        # Last cleanup time
        self._last_cleanup = time.time()
        
        self._logger.info(f"TTL memory manager: default_ttl={default_ttl}, cleanup_interval={cleanup_interval}")
    
    def remember(self, key: str, value: Any, permanent: bool = True, ttl: Optional[float] = None) -> bool:
        """
        Remember information (with TTL).
        
        Args:
            key: Memory key
            value: Memory value
            permanent: Whether to save permanently
            ttl: Time-to-live in seconds, None uses default
            
        Returns:
            bool: Whether successful
        """
        # Calculate expiration time
        ttl_value = ttl if ttl is not None else self.default_ttl
        if ttl_value is not None:
            expiry = time.time() + ttl_value
            self._ttl[key] = expiry
            self._logger.debug(f"Set TTL: {key} -> {ttl_value}s")
        
        # Call parent method
        return super().remember(key, value, permanent)
    
    def recall(self, key: str, default: Any = None, use_cache: bool = True) -> Any:
        """
        Recall information (check expiration).
        
        Args:
            key: Memory key
            default: Default value
            use_cache: Whether to use cache
            
        Returns:
            Any: Stored value, default if expired
        """
        # Check if expired
        if key in self._ttl:
            if time.time() > self._ttl[key]:
                # Expired, auto delete
                self._logger.debug(f"Memory expired: {key}")
                self.forget(key, permanent=True)
                return default
        
        return super().recall(key, default, use_cache)
    
    def _cleanup_expired(self) -> int:
        """
        Clean up expired memories.
        
        Iterates through all TTL records, deleting expired items.
        
        Returns:
            int: Number of items cleaned
        """
        now = time.time()
        expired = []
        
        for key, expiry in self._ttl.items():
            if now > expiry:
                expired.append(key)
        
        for key in expired:
            self.forget(key, permanent=True)
        
        if expired:
            self._logger.info(f"Cleaned {len(expired)} expired memories")
        
        return len(expired)
    
    def periodic_cleanup(self) -> int:
        """
        Periodic cleanup based on cleanup_interval.
        
        Checks if cleanup interval has been reached, executes cleanup if so.
        
        Returns:
            int: Number of items cleaned
        """
        now = time.time()
        if now - self._last_cleanup >= self.cleanup_interval:
            cleaned = self._cleanup_expired()
            self._last_cleanup = now
            return cleaned
        return 0
    
    def get_ttl(self, key: str) -> Optional[float]:
        """
        Get remaining time-to-live.
        
        Args:
            key: Memory key
            
        Returns:
            Optional[float]: Remaining seconds, None means permanent or doesn't exist
        """
        if key in self._ttl:
            remaining = self._ttl[key] - time.time()
            return max(0, remaining)
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        stats = super().get_stats()
        
        # Calculate TTL statistics
        now = time.time()
        ttl_values = [expiry - now for expiry in self._ttl.values() if expiry > now]
        
        stats.update({
            'ttl_enabled': True,
            'ttl_count': len(self._ttl),
            'ttl_avg': float(np.mean(ttl_values)) if ttl_values else 0,
            'ttl_min': float(np.min(ttl_values)) if ttl_values else 0,
            'ttl_max': float(np.max(ttl_values)) if ttl_values else 0,
            'default_ttl': self.default_ttl,
            'cleanup_interval': self.cleanup_interval,
            'last_cleanup': self._last_cleanup,
        })
        
        return stats
    
    def __repr__(self) -> str:
        base_repr = super().__repr__()
        return f"TTL{base_repr}, ttl_items={len(self._ttl)}"


# ============================================================
# Export
# ============================================================

__all__ = [
    'MemoryBackend',
    'InMemoryBackend',
    'JSONMemoryBackend',
    'SQLiteMemoryBackend',
    'MemoryManager',
    'TTLMemoryManager',
]