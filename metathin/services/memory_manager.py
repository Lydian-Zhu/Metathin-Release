"""
Memory Manager Service - Two-Tier Caching System | 记忆管理器服务 - 二级缓存系统
===============================================================================

Provides a memory management service with two-tier architecture:
- Fast in-memory cache (L1) for quick access
- Persistent backend storage (L2) for durability

提供具有二级架构的记忆管理服务：
- 快速内存缓存 (L1) 用于快速访问
- 持久化后端存储 (L2) 用于持久性

Features | 特性:
    - LRU cache eviction when cache fills up | 缓存满时 LRU 淘汰
    - Automatic synchronization between cache and backend | 缓存与后端自动同步
    - Optional TTL (Time-To-Live) for auto-expiring items | 可选的 TTL 自动过期
    - Comprehensive statistics (hit rate, size, etc.) | 全面的统计信息（命中率、大小等）

Design Philosophy | 设计理念:
    - Transparent: Upper layer doesn't know where data resides
      透明：上层不知道数据存储在哪里
    - Configurable: Cache size, backend type, TTL all adjustable
      可配置：缓存大小、后端类型、TTL 都可调整
    - Observable: Full statistics for monitoring
      可观测：完整的统计信息用于监控
"""

import time
import threading
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
import logging

from ..core.memory_backend import MemoryBackend
from ..core.exceptions import MetathinError


# ============================================================
# Memory Manager Service | 记忆管理器服务
# ============================================================

class MemoryManager:
    """
    Memory Manager - Two-tier caching memory service.
    
    记忆管理器 - 二级缓存记忆服务。
    
    This service provides a unified interface for storing and retrieving
    information with automatic caching for performance.
    
    这个服务提供了统一的存储和检索接口，具有自动缓存以提高性能。
    
    Architecture | 架构:
        ┌─────────────────────────────────────────────┐
        │              MemoryManager                  │
        │  ┌─────────────────────────────────────┐   │
        │  │         L1: In-Memory Cache         │   │
        │  │    (Fast, volatile, LRU eviction)   │   │
        │  └─────────────────┬───────────────────┘   │
        │                    │                        │
        │  ┌─────────────────▼───────────────────┐   │
        │  │      L2: Persistent Backend         │   │
        │  │  (Slower, durable, JSON/SQLite)     │   │
        │  └─────────────────────────────────────┘   │
        └─────────────────────────────────────────────┘
    
    Example | 示例:
        >>> from metathin.core.memory_backend import JSONMemoryBackend
        >>> 
        >>> # Create memory manager with JSON backend
        >>> manager = MemoryManager(
        ...     backend=JSONMemoryBackend("memory.json"),
        ...     cache_size=1000
        ... )
        >>> 
        >>> # Store information
        >>> manager.remember("user_pref", {"theme": "dark"}, permanent=True)
        >>> 
        >>> # Retrieve information
        >>> pref = manager.recall("user_pref")
        >>> 
        >>> # Check statistics
        >>> stats = manager.get_stats()
        >>> print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
    """
    
    def __init__(
        self,
        backend: Optional[MemoryBackend] = None,
        cache_size: Optional[int] = 1000,
        enable_cache: bool = True,
        default_ttl: Optional[float] = None,
        cleanup_interval: float = 60.0,
    ):
        """
        Initialize memory manager.
        
        初始化记忆管理器。
        
        Args | 参数:
            backend: Persistent backend (creates default JSON backend if None)
                     持久化后端（为 None 时创建默认 JSON 后端）
            cache_size: Maximum cache size, None for unlimited | 最大缓存大小，None 表示无限制
            enable_cache: Whether to enable L1 cache | 是否启用 L1 缓存
            default_ttl: Default time-to-live in seconds (None = never expire)
                         默认生存时间（秒），None 表示永不过期
            cleanup_interval: Cleanup interval for expired items (seconds)
                              过期项清理间隔（秒）
        """
        # Backend initialization | 后端初始化
        if backend is None:
            from ..core.memory_backend import JSONMemoryBackend
            backend = JSONMemoryBackend("metathin_memory.json")
        
        self.backend = backend
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        # L1: In-memory cache | L1：内存缓存
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        """Cache: key -> (value, timestamp) | 缓存：键 -> (值, 时间戳)"""
        
        self._ttl: Dict[str, float] = {}
        """TTL expiry timestamps: key -> expiry_time | TTL 过期时间戳：键 -> 过期时间"""
        
        # Statistics | 统计信息
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._backend_reads: int = 0
        self._backend_writes: int = 0
        
        # Thread safety | 线程安全
        self._lock = threading.RLock()
        
        # Cleanup tracking | 清理追踪
        self._last_cleanup: float = time.time()
        
        # Logger | 日志记录器
        self._logger = logging.getLogger("metathin.services.MemoryManager")
        
        self._logger.info(
            f"MemoryManager initialized: cache={enable_cache}, "
            f"cache_size={cache_size}, ttl={default_ttl}"
        )
    
    # ============================================================
    # Core Operations | 核心操作
    # ============================================================
    
    def remember(
        self,
        key: str,
        value: Any,
        permanent: bool = True,
        ttl: Optional[float] = None,
    ) -> bool:
        """
        Store information in memory.
        
        在记忆中存储信息。
        
        Args | 参数:
            key: Memory key | 记忆键
            value: Memory value | 记忆值
            permanent: Whether to persist to backend | 是否持久化到后端
            ttl: Time-to-live in seconds (overrides default) | 生存时间（秒），覆盖默认值
            
        Returns | 返回:
            bool: True if successful | 成功返回 True
        """
        with self._lock:
            success = True
            
            # Calculate expiry | 计算过期时间
            ttl_value = ttl if ttl is not None else self.default_ttl
            if ttl_value is not None:
                expiry = time.time() + ttl_value
                self._ttl[key] = expiry
                self._logger.debug(f"Set TTL for '{key}': {ttl_value}s")
            
            # Store in cache (L1) | 存储到缓存 (L1)
            if self.enable_cache:
                self._cache[key] = (value, time.time())
                self._enforce_cache_limit()
                self._logger.debug(f"Cached: '{key}'")
            
            # Store in backend (L2) | 存储到后端 (L2)
            if permanent:
                try:
                    backend_success = self.backend.save(key, value)
                    if backend_success:
                        self._backend_writes += 1
                        self._logger.debug(f"Persisted: '{key}'")
                    else:
                        self._logger.warning(f"Failed to persist: '{key}'")
                        success = False
                except Exception as e:
                    self._logger.error(f"Backend save failed for '{key}': {e}")
                    success = False
            
            return success
    
    def recall(self, key: str, default: Any = None, use_cache: bool = True) -> Any:
        """
        Retrieve information from memory.
        
        从记忆中检索信息。
        
        Retrieves in order: L1 cache → L2 backend → default.
        
        按顺序检索：L1 缓存 → L2 后端 → 默认值。
        
        Args | 参数:
            key: Memory key | 记忆键
            default: Default value if key not found | 键不存在时的默认值
            use_cache: Whether to check L1 cache first | 是否先检查 L1 缓存
            
        Returns | 返回:
            Any: Stored value or default | 存储的值或默认值
        """
        with self._lock:
            # Check if expired | 检查是否过期
            if self._is_expired(key):
                self._logger.debug(f"Key '{key}' expired, deleting")
                self.forget(key, permanent=True)
                return default
            
            # Try L1 cache first | 先尝试 L1 缓存
            if self.enable_cache and use_cache and key in self._cache:
                self._cache_hits += 1
                value, _ = self._cache[key]
                # Move to end (LRU) | 移动到末尾（LRU）
                self._cache.move_to_end(key)
                self._logger.debug(f"Cache hit: '{key}'")
                return value
            
            self._cache_misses += 1
            
            # Try L2 backend | 尝试 L2 后端
            try:
                value = self.backend.load(key)
                self._backend_reads += 1
                
                if value is not None:
                    # Update cache with retrieved value | 用检索到的值更新缓存
                    if self.enable_cache:
                        self._cache[key] = (value, time.time())
                        self._enforce_cache_limit()
                    self._logger.debug(f"Backend load: '{key}'")
                    return value
                else:
                    self._logger.debug(f"Key not found: '{key}'")
                    
            except Exception as e:
                self._logger.error(f"Backend load failed for '{key}': {e}")
            
            return default
    
    def forget(self, key: str, permanent: bool = True) -> bool:
        """
        Delete information from memory.
        
        从记忆中删除信息。
        
        Args | 参数:
            key: Memory key | 记忆键
            permanent: Whether to delete from backend as well | 是否也从后端删除
            
        Returns | 返回:
            bool: True if successful | 成功返回 True
        """
        with self._lock:
            success = True
            
            # Remove from cache | 从缓存删除
            if key in self._cache:
                del self._cache[key]
            if key in self._ttl:
                del self._ttl[key]
            
            # Remove from backend | 从后端删除
            if permanent:
                try:
                    backend_success = self.backend.delete(key)
                    if not backend_success:
                        self._logger.warning(f"Backend delete failed: '{key}'")
                        success = False
                except Exception as e:
                    self._logger.error(f"Backend delete failed for '{key}': {e}")
                    success = False
            
            self._logger.debug(f"Forgotten: '{key}'")
            return success
    
    def clear(self, permanent: bool = True) -> bool:
        """
        Clear all memory.
        
        清空所有记忆。
        
        Args | 参数:
            permanent: Whether to clear backend as well | 是否也清空后端
            
        Returns | 返回:
            bool: True if successful | 成功返回 True
        """
        with self._lock:
            # Clear cache | 清空缓存
            self._cache.clear()
            self._ttl.clear()
            
            # Clear backend | 清空后端
            if permanent:
                try:
                    success = self.backend.clear()
                    self._logger.info("Cleared all memory (including backend)")
                    return success
                except Exception as e:
                    self._logger.error(f"Failed to clear backend: {e}")
                    return False
            
            self._logger.info("Cleared cache only")
            return True
    
    def list_all(self, include_cache: bool = True) -> List[str]:
        """
        List all memory keys.
        
        列出所有记忆键。
        
        Args | 参数:
            include_cache: Whether to include cache-only keys | 是否包含仅缓存的键
            
        Returns | 返回:
            List[str]: List of keys | 键列表
        """
        keys = set()
        
        # Get backend keys | 获取后端键
        try:
            if hasattr(self.backend, 'list_keys'):
                backend_keys = self.backend.list_keys()
                if backend_keys:
                    keys.update(backend_keys)
        except Exception as e:
            self._logger.error(f"Failed to list backend keys: {e}")
        
        # Get cache keys | 获取缓存键
        if include_cache:
            keys.update(self._cache.keys())
        
        return sorted(list(keys))
    
    # ============================================================
    # Cache Management | 缓存管理
    # ============================================================
    
    def _enforce_cache_limit(self) -> None:
        """
        Enforce cache size limit (LRU eviction).
        
        强制执行缓存大小限制（LRU 淘汰）。
        """
        if self.cache_size is None or len(self._cache) <= self.cache_size:
            return
        
        # Remove oldest items (LRU) | 移除最旧的项（LRU）
        items_to_remove = len(self._cache) - self.cache_size
        for _ in range(items_to_remove):
            self._cache.popitem(last=False)
        
        self._logger.debug(f"Cache evicted {items_to_remove} items")
    
    def _is_expired(self, key: str) -> bool:
        """
        Check if a key has expired.
        
        检查键是否已过期。
        
        Args | 参数:
            key: Memory key | 记忆键
            
        Returns | 返回:
            bool: True if expired | 已过期返回 True
        """
        if key in self._ttl:
            if time.time() > self._ttl[key]:
                return True
        return False
    
    def _cleanup_expired(self) -> int:
        """
        Clean up expired memory items.
        
        清理过期的记忆项。
        
        Returns | 返回:
            int: Number of items cleaned | 清理的项数
        """
        now = time.time()
        expired_keys = [key for key, expiry in self._ttl.items() if now > expiry]
        
        for key in expired_keys:
            self.forget(key, permanent=True)
        
        if expired_keys:
            self._logger.debug(f"Cleaned {len(expired_keys)} expired items")
        
        return len(expired_keys)
    
    def periodic_cleanup(self) -> int:
        """
        Perform periodic cleanup if interval has elapsed.
        
        如果间隔时间已到，执行周期性清理。
        
        Returns | 返回:
            int: Number of items cleaned | 清理的项数
        """
        now = time.time()
        if now - self._last_cleanup >= self.cleanup_interval:
            cleaned = self._cleanup_expired()
            self._last_cleanup = now
            return cleaned
        return 0
    
    def get_ttl(self, key: str) -> Optional[float]:
        """
        Get remaining time-to-live for a key.
        
        获取键的剩余生存时间。
        
        Args | 参数:
            key: Memory key | 记忆键
            
        Returns | 返回:
            Optional[float]: Remaining seconds, None if no TTL or key doesn't exist
                             剩余秒数，无 TTL 或键不存在时返回 None
        """
        if key in self._ttl:
            remaining = self._ttl[key] - time.time()
            return max(0.0, remaining)
        return None
    
    # ============================================================
    # Statistics | 统计信息
    # ============================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory manager statistics.
        
        获取记忆管理器统计信息。
        
        Returns | 返回:
            Dict: Statistics including hit rate, sizes, etc.
                  包含命中率、大小等的统计字典
        """
        total_requests = self._cache_hits + self._cache_misses
        
        # Calculate TTL statistics | 计算 TTL 统计
        now = time.time()
        ttl_values = [expiry - now for expiry in self._ttl.values() if expiry > now]
        
        return {
            'enabled': True,
            'cache_enabled': self.enable_cache,
            'cache_size': len(self._cache),
            'cache_capacity': self.cache_size,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': self._cache_hits / max(1, total_requests),
            'backend_reads': self._backend_reads,
            'backend_writes': self._backend_writes,
            'backend_type': self.backend.__class__.__name__,
            'ttl_enabled': self.default_ttl is not None,
            'ttl_count': len(self._ttl),
            'ttl_avg': float(sum(ttl_values) / len(ttl_values)) if ttl_values else 0,
        }
    
    def flush(self) -> bool:
        """
        Force flush (for backends that support it).
        
        强制刷新（对于支持的后端）。
        
        Returns | 返回:
            bool: True if successful | 成功返回 True
        """
        if hasattr(self.backend, 'flush'):
            return self.backend.flush()
        return True
    
    # ============================================================
    # Magic Methods | 魔术方法
    # ============================================================
    
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
        return (
            f"MemoryManager(cache={len(self._cache)} items, "
            f"backend={self.backend.__class__.__name__}, "
            f"hit_rate={self._cache_hits/max(1, self._cache_hits+self._cache_misses):.2%})"
        )


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    'MemoryManager',  # Main memory manager service | 主记忆管理器服务
]