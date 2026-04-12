"""
Unit tests for MemoryManager service.
记忆管理器服务单元测试。
"""

import pytest
import time
from metathin.services import MemoryManager
from metathin.core.memory_backend import InMemoryBackend, JSONMemoryBackend


class TestMemoryManager:
    """Test MemoryManager service."""
    
    def test_initialization(self):
        """Memory manager should initialize correctly."""
        manager = MemoryManager(backend=InMemoryBackend())
        
        assert manager.enable_cache is True
        assert manager.cache_size == 1000
        assert manager.default_ttl is None
    
    def test_remember_and_recall(self):
        """Should store and retrieve values."""
        manager = MemoryManager(backend=InMemoryBackend())
        
        success = manager.remember("key1", "value1")
        assert success is True
        
        value = manager.recall("key1")
        assert value == "value1"
    
    def test_recall_default_value(self):
        """Should return default when key not found."""
        manager = MemoryManager(backend=InMemoryBackend())
        
        value = manager.recall("nonexistent", default="default")
        assert value == "default"
    
    def test_recall_returns_none_when_not_found_no_default(self):
        """Should return None when key not found and no default."""
        manager = MemoryManager(backend=InMemoryBackend())
        
        value = manager.recall("nonexistent")
        assert value is None
    
    def test_forget_removes_key(self):
        """Should remove key from memory."""
        manager = MemoryManager(backend=InMemoryBackend())
        
        manager.remember("key1", "value1")
        assert manager.recall("key1") == "value1"
        
        success = manager.forget("key1")
        assert success is True
        assert manager.recall("key1") is None
    
    def test_forget_nonexistent_key(self):
        """Should return False when forgetting nonexistent key."""
        manager = MemoryManager(backend=InMemoryBackend())
        
        success = manager.forget("nonexistent")
        assert success is False
    
    def test_clear_removes_all_keys(self):
        """Should clear all keys from memory."""
        manager = MemoryManager(backend=InMemoryBackend())
        
        manager.remember("key1", "value1")
        manager.remember("key2", "value2")
        manager.remember("key3", "value3")
        
        assert len(manager.list_all()) == 3
        
        success = manager.clear(permanent=True)
        assert success is True
        assert len(manager.list_all()) == 0
    
    def test_list_all_returns_keys(self):
        """Should return list of all keys."""
        manager = MemoryManager(backend=InMemoryBackend())
        
        manager.remember("a", 1)
        manager.remember("b", 2)
        manager.remember("c", 3)
        
        keys = manager.list_all()
        assert set(keys) == {"a", "b", "c"}
    
    def test_contains_operator(self):
        """Should support 'in' operator."""
        manager = MemoryManager(backend=InMemoryBackend())
        
        manager.remember("key1", "value1")
        
        assert "key1" in manager
        assert "key2" not in manager
    
    def test_getitem_operator(self):
        """Should support dictionary-style access."""
        manager = MemoryManager(backend=InMemoryBackend())
        
        manager.remember("key1", "value1")
        
        assert manager["key1"] == "value1"
    
    def test_getitem_raises_key_error(self):
        """Should raise KeyError when key not found."""
        manager = MemoryManager(backend=InMemoryBackend())
        
        with pytest.raises(KeyError):
            _ = manager["nonexistent"]
    
    def test_setitem_operator(self):
        """Should support dictionary-style assignment."""
        manager = MemoryManager(backend=InMemoryBackend())
        
        manager["key1"] = "value1"
        assert manager["key1"] == "value1"
    
    def test_delitem_operator(self):
        """Should support dictionary-style deletion."""
        manager = MemoryManager(backend=InMemoryBackend())
        
        manager["key1"] = "value1"
        assert "key1" in manager
        
        del manager["key1"]
        assert "key1" not in manager
    
    def test_len_operator(self):
        """Should return number of keys."""
        manager = MemoryManager(backend=InMemoryBackend())
        
        manager.remember("a", 1)
        manager.remember("b", 2)
        
        assert len(manager) == 2
    
    def test_cache_hit_returns_cached_value(self):
        """Should return cached value on hit."""
        manager = MemoryManager(backend=InMemoryBackend(), cache_size=10)
        
        manager.remember("key", "value", permanent=True)
        
        # First access - cache miss, load from backend
        value1 = manager.recall("key")
        # Second access - cache hit
        value2 = manager.recall("key")
        
        assert value1 == value2 == "value"
        stats = manager.get_stats()
        assert stats['cache_hits'] >= 1
    
    def test_cache_eviction_lru(self):
        """Should evict least recently used items when cache is full."""
        manager = MemoryManager(backend=InMemoryBackend(), cache_size=2)
        
        manager.remember("a", 1)
        manager.remember("b", 2)
        manager.remember("c", 3)  # This should evict 'a'
        
        # 'a' should be evicted from cache but still in backend
        # Need to check backend directly
        backend_value = manager.backend.load("a")
        assert backend_value == 1
        
        # 'b' and 'c' should be in cache
        stats = manager.get_stats()
        assert stats['cache_size'] <= 2
    
    def test_permanent_vs_temporary(self):
        """Permanent keys should persist, temporary keys should not."""
        manager = MemoryManager(backend=InMemoryBackend())
        
        manager.remember("permanent", "value", permanent=True)
        manager.remember("temporary", "value", permanent=False)
        
        # Clear cache only (permanent=False)
        manager.clear(permanent=False)
        
        # Permanent should still be in backend
        assert manager.recall("permanent") == "value"
        
        # Temporary should be gone
        assert manager.recall("temporary") is None
    
    def test_get_stats_returns_statistics(self):
        """Should return comprehensive statistics."""
        manager = MemoryManager(backend=InMemoryBackend())
        
        manager.remember("a", 1)
        manager.recall("a")
        manager.recall("a")
        
        stats = manager.get_stats()
        
        assert 'enabled' in stats
        assert 'cache_enabled' in stats
        assert 'cache_hits' in stats
        assert 'backend_type' in stats
        assert stats['backend_type'] == 'InMemoryBackend'


class TestMemoryManagerWithTTL:
    """Test MemoryManager with TTL functionality."""
    
    def test_ttl_expiration(self):
        """Items should expire after TTL."""
        manager = MemoryManager(
            backend=InMemoryBackend(),
            default_ttl=0.1  # 100ms
        )
        
        manager.remember("expiring", "value")
        
        # Should be available immediately
        assert manager.recall("expiring") == "value"
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Should be expired
        assert manager.recall("expiring") is None
    
    def test_ttl_can_be_overridden_per_item(self):
        """TTL can be overridden per item."""
        manager = MemoryManager(
            backend=InMemoryBackend(),
            default_ttl=1.0
        )
        
        manager.remember("long", "value", ttl=2.0)
        manager.remember("short", "value", ttl=0.1)
        
        time.sleep(0.15)
        
        assert manager.recall("short") is None
        assert manager.recall("long") == "value"
    
    def test_ttl_none_means_no_expiration(self):
        """TTL=None should mean no expiration."""
        manager = MemoryManager(
            backend=InMemoryBackend(),
            default_ttl=None
        )
        
        manager.remember("permanent", "value")
        
        time.sleep(0.1)
        assert manager.recall("permanent") == "value"
    
    def test_get_ttl_returns_remaining_time(self):
        """get_ttl() should return remaining time."""
        manager = MemoryManager(backend=InMemoryBackend())
        
        manager.remember("key", "value", ttl=1.0)
        
        remaining = manager.get_ttl("key")
        assert 0.9 <= remaining <= 1.0
    
    def test_get_ttl_returns_none_for_no_ttl(self):
        """get_ttl() should return None for keys without TTL."""
        manager = MemoryManager(backend=InMemoryBackend(), default_ttl=None)
        
        manager.remember("key", "value")
        
        assert manager.get_ttl("key") is None
    
    def test_ttl_stats_included(self):
        """TTL statistics should be included in get_stats()."""
        manager = MemoryManager(
            backend=InMemoryBackend(),
            default_ttl=1.0
        )
        
        manager.remember("key1", "value", ttl=0.5)
        manager.remember("key2", "value", ttl=2.0)
        
        stats = manager.get_stats()
        assert 'ttl_enabled' in stats
        assert 'ttl_count' in stats
        assert stats['ttl_count'] == 2


class TestMemoryManagerWithJSONBackend:
    """Test MemoryManager with JSON backend."""
    
    def test_json_backend_persistence(self, tmp_path):
        """JSON backend should persist data to disk."""
        json_path = tmp_path / "test_memory.json"
        backend = JSONMemoryBackend(json_path, auto_save=True)
        manager = MemoryManager(backend=backend)
        
        manager.remember("key", "value", permanent=True)
        
        # Create new manager with same file
        backend2 = JSONMemoryBackend(json_path, auto_save=True)
        manager2 = MemoryManager(backend=backend2)
        
        assert manager2.recall("key") == "value"
    
    def test_json_backend_loads_existing_data(self, tmp_path):
        """JSON backend should load existing data on initialization."""
        json_path = tmp_path / "test_memory.json"
        
        # First manager saves data
        backend1 = JSONMemoryBackend(json_path, auto_save=True)
        manager1 = MemoryManager(backend=backend1)
        manager1.remember("persisted", "data", permanent=True)
        
        # Second manager loads existing data
        backend2 = JSONMemoryBackend(json_path, auto_save=True)
        manager2 = MemoryManager(backend=backend2)
        
        assert manager2.recall("persisted") == "data"
    
    def test_complex_object_serialization(self, tmp_path):
        """JSON backend should handle complex objects."""
        json_path = tmp_path / "test_memory.json"
        backend = JSONMemoryBackend(json_path, auto_save=True)
        manager = MemoryManager(backend=backend)
        
        complex_obj = {
            'list': [1, 2, 3],
            'dict': {'a': 1, 'b': 2},
            'tuple': (1, 2, 3),
            'float': 3.14159,
            'bool': True,
            'none': None
        }
        
        manager.remember("complex", complex_obj, permanent=True)
        
        # Reload and verify
        backend2 = JSONMemoryBackend(json_path, auto_save=True)
        manager2 = MemoryManager(backend=backend2)
        retrieved = manager2.recall("complex")
        
        assert retrieved['list'] == [1, 2, 3]
        assert retrieved['dict'] == {'a': 1, 'b': 2}
        assert retrieved['tuple'] == [1, 2, 3]  # JSON serialization converts tuple to list
        assert retrieved['float'] == 3.14159
        assert retrieved['bool'] is True
        assert retrieved['none'] is None