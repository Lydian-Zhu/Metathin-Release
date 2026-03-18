"""
Function Memory Bank - Store and Manage Function Knowledge
======================================================================

Provides persistent storage, retrieval, and management of discovered mathematical functions.
The memory system enables the scientific discovery agent to remember and reuse previously
learned patterns, accelerating future discoveries through similarity-based retrieval.

Design Philosophy:
    - Persistent: Functions survive across discovery sessions
    - Retrievable: Fast similarity-based search using feature vectors
    - Extensible: Support for multiple storage backends
    - Scalable: Handles large collections of functions efficiently
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import pickle
import hashlib
import logging
from pathlib import Path

# Import Metathin core memory system if available
try:
    from metathin.core.memory import MemoryManager, JSONMemoryBackend, SQLiteMemoryBackend, InMemoryBackend
    METATHIN_AVAILABLE = True
except ImportError:
    METATHIN_AVAILABLE = False
    print("Warning: Metathin core not found, using simplified memory system")


# ============================================================
# Function Memory Data Class
# ============================================================

@dataclass
class FunctionMemory:
    """
    Function Memory Data Class.
    
    Represents a single discovered function stored in memory.
    
    Attributes:
        expression: String representation of the function
        parameters: Dictionary of parameter values
        feature_vector: Feature vector for similarity matching
        accuracy: Recognition accuracy (0-1)
        usage_count: Number of times this memory has been used
        domain: Valid domain (min, max)
        tags: List of tags for categorization
        source: Source of the function (e.g., 'generated', 'discovered', 'pretrained')
        created_at: Creation timestamp
        last_used: Last access timestamp
        metadata: Additional metadata
    """
    
    # Required fields
    expression: str
    parameters: Dict[str, float]
    feature_vector: np.ndarray
    
    # Optional fields
    accuracy: float = 1.0
    usage_count: int = 0
    domain: Tuple[float, float] = (-10.0, 10.0)
    tags: List[str] = field(default_factory=list)
    source: str = "generated"
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    last_used: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # id is a computed property, not an initialization parameter
    @property
    def id(self) -> str:
        """Generate unique ID (computed property)."""
        content = f"{self.expression}_{self.parameters}_{self.created_at}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Process feature vector
        if self.feature_vector is None:
            self.feature_vector = np.array([0.0], dtype=np.float64)
        elif not isinstance(self.feature_vector, np.ndarray):
            try:
                self.feature_vector = np.array(self.feature_vector, dtype=np.float64)
            except:
                self.feature_vector = np.array([0.0], dtype=np.float64)
        
        # Ensure feature is 1-dimensional
        if self.feature_vector.ndim > 1:
            self.feature_vector = self.feature_vector.flatten()
        
        # Ensure feature is not empty
        if len(self.feature_vector) == 0:
            self.feature_vector = np.array([0.0], dtype=np.float64)
        
        # Process parameters dictionary
        if self.parameters is None:
            self.parameters = {}
        
        # Process tags
        if self.tags is None:
            self.tags = []
        
        # Process metadata
        if self.metadata is None:
            self.metadata = {}
        
        # Ensure accuracy is within valid range
        self.accuracy = max(0.0, min(1.0, self.accuracy))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)."""
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
        """Create from dictionary."""
        # Remove id field as it won't be accepted by __init__
        data_copy = data.copy()
        data_copy.pop('id', None)
        
        # Process feature vector
        if 'feature_vector' in data_copy:
            if isinstance(data_copy['feature_vector'], list):
                data_copy['feature_vector'] = np.array(data_copy['feature_vector'], dtype=np.float64)
            elif not isinstance(data_copy['feature_vector'], np.ndarray):
                data_copy['feature_vector'] = np.array([0.0], dtype=np.float64)
        
        # Process domain
        if 'domain' in data_copy and isinstance(data_copy['domain'], list):
            data_copy['domain'] = tuple(data_copy['domain'])
        
        return cls(**data_copy)
    
    def update_usage(self):
        """Update usage statistics."""
        self.usage_count += 1
        self.last_used = datetime.now().timestamp()
    
    def similarity(self, other: 'FunctionMemory') -> float:
        """Calculate similarity with another function."""
        if other is None:
            return 0.0
        
        if other.feature_vector is None:
            return 0.0
        
        v1 = self.feature_vector
        v2 = other.feature_vector
        
        if v1 is None or v2 is None:
            return 0.0
        
        if len(v1) == 0 or len(v2) == 0:
            return 0.0
        
        # Handle dimension mismatch
        if len(v1) != len(v2):
            min_len = min(len(v1), len(v2))
            v1 = v1[:min_len]
            v2 = v2[:min_len]
        
        # Compute cosine similarity
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cos_sim = np.dot(v1, v2) / (norm1 * norm2)
        # Map [-1,1] to [0,1]
        return float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))
    
    def __repr__(self) -> str:
        return f"FunctionMemory(expr='{self.expression}', acc={self.accuracy:.2f}, used={self.usage_count})"


# ============================================================
# Memory Backend Interface (Fallback when Metathin unavailable)
# ============================================================

class SimpleMemoryBackend:
    """
    Simplified Memory Backend (In-memory only).
    
    Provides an interface compatible with Metathin memory system:
        - save: Save key-value pair
        - load: Load value by key
        - delete: Delete key
        - list_keys: List all keys
        - clear: Clear all
    """
    
    def __init__(self):
        """Initialize in-memory storage."""
        self._memory: Dict[str, Any] = {}
        self._logger = logging.getLogger("metathin_sci.memory.SimpleMemoryBackend")
    
    def save(self, key: str, value: Any) -> bool:
        """Save memory."""
        try:
            self._memory[key] = value
            self._logger.debug(f"Saved: {key}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to save {key}: {e}")
            return False
    
    def load(self, key: str) -> Optional[Any]:
        """Load memory."""
        value = self._memory.get(key)
        self._logger.debug(f"Loaded: {key} -> {'found' if value is not None else 'not found'}")
        return value
    
    def delete(self, key: str) -> bool:
        """Delete memory."""
        if key in self._memory:
            del self._memory[key]
            self._logger.debug(f"Deleted: {key}")
            return True
        return False
    
    def list_keys(self) -> List[str]:
        """List all keys."""
        return list(self._memory.keys())
    
    def clear(self) -> bool:
        """Clear all memory."""
        self._memory.clear()
        self._logger.info("Cleared all memory")
        return True
    
    def get_size(self) -> int:
        """Get number of records."""
        return len(self._memory)


# ============================================================
# Custom JSON Memory Backend (Fallback when Metathin unavailable)
# ============================================================

class SimpleJSONMemoryBackend:
    """
    Simplified JSON Memory Backend.
    
    Provides JSON storage interface compatible with Metathin memory system.
    """
    
    def __init__(self, filepath: Union[str, Path] = "function_memory.json"):
        """Initialize JSON memory backend."""
        self.filepath = Path(filepath)
        self._memory: Dict[str, Any] = {}
        self._logger = logging.getLogger("metathin_sci.memory.SimpleJSONMemoryBackend")
        self._load()
    
    def _load(self):
        """Load from file."""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    self._memory = json.load(f)
                self._logger.info(f"Loaded {len(self._memory)} memories from {self.filepath}")
            except Exception as e:
                self._logger.error(f"Load failed: {e}")
                self._memory = {}
    
    def _save(self):
        """Save to file."""
        try:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self._memory, f, indent=2, ensure_ascii=False, default=str)
            return True
        except Exception as e:
            self._logger.error(f"Save failed: {e}")
            return False
    
    def save(self, key: str, value: Any) -> bool:
        """Save memory."""
        try:
            self._memory[key] = value
            self._save()
            return True
        except Exception as e:
            self._logger.error(f"Failed to save {key}: {e}")
            return False
    
    def load(self, key: str) -> Optional[Any]:
        """Load memory."""
        return self._memory.get(key)
    
    def delete(self, key: str) -> bool:
        """Delete memory."""
        if key in self._memory:
            del self._memory[key]
            self._save()
            return True
        return False
    
    def list_keys(self) -> List[str]:
        """List all keys."""
        return list(self._memory.keys())
    
    def clear(self) -> bool:
        """Clear all memory."""
        self._memory.clear()
        self._save()
        return True


# ============================================================
# Function Memory Bank Manager
# ============================================================

class FunctionMemoryBank:
    """
    Function Memory Bank Manager.
    
    Manages storage, retrieval, and learning of function memories.
    Based on Metathin core memory system, providing persistent storage and fast retrieval.
    
    Parameters:
        memory_backend: Memory backend type ('json', 'sqlite', 'memory')
        memory_path: Memory storage path
        auto_save: Whether to auto-save changes
        
    Example:
        >>> bank = FunctionMemoryBank(memory_backend='json', memory_path='my_functions.json')
        >>> 
        >>> # Add a function
        >>> features = np.array([0.5, 0.2, 0.8])
        >>> func = FunctionMemory(
        ...     expression="a*sin(ω*x)",
        ...     parameters={'a': 2.0, 'ω': 1.5},
        ...     feature_vector=features,
        ...     tags=['periodic']
        ... )
        >>> bank.add(func)
        >>> 
        >>> # Find similar functions
        >>> query = np.array([0.45, 0.25, 0.75])
        >>> similar = bank.find_similar(query, k=3)
    """
    
    def __init__(self,
                 memory_backend: str = 'json',
                 memory_path: Optional[str] = None,
                 auto_save: bool = True):
        """
        Initialize function memory bank.
        
        Args:
            memory_backend: Memory backend type ('json', 'sqlite', 'memory')
            memory_path: Memory storage path
            auto_save: Whether to auto-save changes
        """
        self.auto_save = auto_save
        self.logger = logging.getLogger("metathin_sci.memory.FunctionMemoryBank")
        
        # Initialize memory backend
        if METATHIN_AVAILABLE:
            # Use Metathin core memory system
            from metathin.core.memory import MemoryBackend, JSONMemoryBackend, SQLiteMemoryBackend, InMemoryBackend, MemoryManager
            
            if memory_backend == 'json':
                path = memory_path or "function_memory.json"
                backend = JSONMemoryBackend(path)
                self.memory = MemoryManager(backend=backend)
                self.logger.info(f"Using JSON memory backend: {path}")
            elif memory_backend == 'sqlite':
                path = memory_path or "function_memory.db"
                backend = SQLiteMemoryBackend(path)
                self.memory = MemoryManager(backend=backend)
                self.logger.info(f"Using SQLite memory backend: {path}")
            else:
                backend = InMemoryBackend()
                self.memory = MemoryManager(backend=backend)
                self.logger.info("Using in-memory memory backend")
        else:
            # Use simplified memory system
            if memory_backend == 'json':
                path = memory_path or "function_memory.json"
                self.memory = SimpleJSONMemoryBackend(path)
                self.logger.info(f"Using simplified JSON memory backend: {path}")
            else:
                self.memory = SimpleMemoryBackend()
                self.logger.info("Using simplified in-memory memory system")
        
        # Index for fast similarity matching
        self.functions: List[FunctionMemory] = []
        self.feature_matrix: Optional[np.ndarray] = None
        self._need_rebuild_index = False
        
        # Load existing memories
        self._load_all()
        
        self.logger.info(f"Function memory bank initialized with {len(self.functions)} functions")
    
    def _load_all(self):
        """Load all memories from backend."""
        try:
            if METATHIN_AVAILABLE:
                # Metathin core mode
                keys = self.memory.list_all()
                for key in keys:
                    try:
                        data = self.memory.recall(key)
                        if data:
                            if isinstance(data, dict):
                                func = FunctionMemory.from_dict(data)
                            else:
                                func = data
                            if func is not None and isinstance(func, FunctionMemory):
                                self.functions.append(func)
                    except Exception as e:
                        self.logger.debug(f"Failed to load single memory {key}: {e}")
                        continue
            else:
                # Simplified mode
                if hasattr(self.memory, 'list_keys'):
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
                            self.logger.debug(f"Failed to load single memory {key}: {e}")
                            continue
                else:
                    self.logger.warning("Memory backend does not have list_keys method")
            
            if self.functions:
                self._rebuild_index()
                self.logger.info(f"Loaded {len(self.functions)} function memories")
                
        except Exception as e:
            self.logger.error(f"Failed to load memories: {e}")
    
    def _rebuild_index(self):
        """Rebuild feature index for similarity search."""
        if not self.functions:
            self.feature_matrix = None
            return
        
        # Build feature matrix
        features = []
        valid_indices = []
        
        for i, func in enumerate(self.functions):
            if func is not None and func.feature_vector is not None:
                # Ensure feature vector is valid
                fv = func.feature_vector
                if len(fv) > 0 and not np.any(np.isnan(fv)) and not np.any(np.isinf(fv)):
                    features.append(fv)
                    valid_indices.append(i)
        
        if not features:
            self.feature_matrix = None
            return
        
        # Ensure all feature vectors have same dimension
        max_dim = max(len(f) for f in features)
        aligned_features = []
        
        for f in features:
            if len(f) < max_dim:
                # Pad to max dimension
                aligned_f = np.pad(f, (0, max_dim - len(f)), mode='constant', constant_values=0)
            else:
                aligned_f = f[:max_dim]
            aligned_features.append(aligned_f)
        
        self.feature_matrix = np.array(aligned_features)
        self._need_rebuild_index = False
        
        self.logger.debug(f"Index rebuilt, feature matrix shape: {self.feature_matrix.shape}")
    
    def add(self, func: FunctionMemory) -> bool:
        """
        Add function memory.
        
        Args:
            func: Function memory object
            
        Returns:
            bool: Success status
        """
        if func is None:
            self.logger.warning("Attempted to add None function")
            return False
        
        if not isinstance(func, FunctionMemory):
            self.logger.warning(f"Attempted to add non-FunctionMemory object: {type(func)}")
            return False
        
        try:
            # Generate storage key
            key = f"func_{func.id}"
            
            # Save to memory system
            if METATHIN_AVAILABLE:
                # Metathin core mode
                success = self.memory.remember(key, func.to_dict(), permanent=True)
            else:
                # Simplified mode
                success = self.memory.save(key, func.to_dict() if hasattr(self.memory, 'save') else func)
            
            if success:
                self.functions.append(func)
                self._need_rebuild_index = True
                self.logger.debug(f"Added function: {func.expression}")
                
                if self.auto_save:
                    self.save()
                
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to add function: {e}")
            return False
    
    def add_batch(self, funcs: List[FunctionMemory]) -> int:
        """
        Add multiple function memories.
        
        Args:
            funcs: List of function memories
            
        Returns:
            int: Number successfully added
        """
        success_count = 0
        for func in funcs:
            if self.add(func):
                success_count += 1
        
        self.logger.info(f"Batch add complete: {success_count}/{len(funcs)}")
        return success_count
    
    def remove(self, func_id: str) -> bool:
        """
        Remove function memory by ID.
        
        Args:
            func_id: Function ID
            
        Returns:
            bool: Success status
        """
        try:
            # Remove from list
            removed = False
            remove_idx = None
            for i, func in enumerate(self.functions):
                if func is not None and func.id == func_id:
                    remove_idx = i
                    removed = True
                    break
            
            if not removed or remove_idx is None:
                self.logger.warning(f"Function not found: {func_id}")
                return False
            
            # Remove from list
            self.functions.pop(remove_idx)
            
            # Remove from memory system
            key = f"func_{func_id}"
            if METATHIN_AVAILABLE:
                success = self.memory.forget(key)
            else:
                success = self.memory.delete(key) if hasattr(self.memory, 'delete') else False
            
            if success:
                self._need_rebuild_index = True
                self.logger.debug(f"Removed function: {func_id}")
                
                if self.auto_save:
                    self.save()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to remove function: {e}")
            return False
    
    def get(self, func_id: str) -> Optional[FunctionMemory]:
        """
        Get function memory by ID.
        
        Args:
            func_id: Function ID
            
        Returns:
            Optional[FunctionMemory]: Function memory object
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
        Find similar functions by feature vector.
        
        Args:
            query_features: Query feature vector
            k: Maximum number to return
            threshold: Similarity threshold
            min_similarity: Minimum similarity threshold (fallback)
            
        Returns:
            List[FunctionMemory]: List of similar functions
        """
        # Check input
        if query_features is None:
            self.logger.debug("Query features are None")
            return []
        
        if not self.functions:
            self.logger.debug("Memory bank is empty")
            return []
        
        # Ensure feature is 1-dimensional
        if isinstance(query_features, np.ndarray):
            if query_features.ndim > 1:
                query_features = query_features.flatten()
        else:
            try:
                query_features = np.array(query_features, dtype=np.float64)
                if query_features.ndim > 1:
                    query_features = query_features.flatten()
            except Exception as e:
                self.logger.error(f"Query feature conversion failed: {e}")
                return []
        
        # Check if features are valid
        if len(query_features) == 0 or np.any(np.isnan(query_features)) or np.any(np.isinf(query_features)):
            self.logger.debug("Query features are invalid")
            return []
        
        # Rebuild index if needed
        if self._need_rebuild_index:
            self._rebuild_index()
        
        if self.feature_matrix is None:
            self.logger.debug("Feature matrix is empty")
            return []
        
        # Ensure feature dimensions match
        if len(query_features) != self.feature_matrix.shape[1]:
            self.logger.debug(f"Feature dimension mismatch: query {len(query_features)} vs bank {self.feature_matrix.shape[1]}")
            if len(query_features) > self.feature_matrix.shape[1]:
                query_features = query_features[:self.feature_matrix.shape[1]]
            else:
                query_features = np.pad(query_features, 
                                      (0, self.feature_matrix.shape[1] - len(query_features)),
                                      mode='constant', constant_values=0)
        
        # Compute cosine similarity
        try:
            # Compute query vector norm
            query_norm = np.linalg.norm(query_features)
            if query_norm < 1e-10:
                self.logger.debug("Query vector norm near zero")
                return []
            
            # Compute norms for each bank vector
            norms = np.linalg.norm(self.feature_matrix, axis=1)
            
            # Prevent division by zero
            norms = np.maximum(norms, 1e-10)
            
            # Compute dot products
            dot_products = np.dot(self.feature_matrix, query_features)
            
            # Compute cosine similarities
            similarities = dot_products / (norms * query_norm)
            
            # Map [-1,1] to [0,1]
            similarities = (similarities + 1.0) / 2.0
            
            # Ensure in [0,1] range
            similarities = np.clip(similarities, 0.0, 1.0)
            
            self.logger.debug(f"Similarity range: [{np.min(similarities):.3f}, {np.max(similarities):.3f}]")
            
            # Get indices above threshold
            above_threshold = np.where(similarities >= threshold)[0]
            
            results = []
            
            if len(above_threshold) > 0:
                # Sort by similarity
                sorted_indices = above_threshold[np.argsort(similarities[above_threshold])[::-1]]
                # Take top k
                result_indices = sorted_indices[:k]
                
                for idx in result_indices:
                    if idx < len(self.functions) and self.functions[idx] is not None:
                        func = self.functions[idx]
                        func.update_usage()
                        results.append(func)
                        self.logger.debug(f"  Candidate: {func.expression} similarity={similarities[idx]:.3f}")
                
                self.logger.debug(f"Found {len(results)} similar functions (threshold={threshold})")
                return results
            
            # If none above threshold, try lower threshold
            self.logger.debug(f"No functions above threshold {threshold}, trying lower threshold {min_similarity}")
            
            # Find all above minimum threshold
            above_min = np.where(similarities >= min_similarity)[0]
            
            if len(above_min) > 0:
                # Sort by similarity
                sorted_indices = above_min[np.argsort(similarities[above_min])[::-1]]
                # Take top k
                result_indices = sorted_indices[:k]
                
                for idx in result_indices:
                    if idx < len(self.functions) and self.functions[idx] is not None:
                        func = self.functions[idx]
                        func.update_usage()
                        results.append(func)
                        self.logger.debug(f"  Candidate (fallback): {func.expression} similarity={similarities[idx]:.3f}")
                
                self.logger.debug(f"Found {len(results)} similar functions (fallback threshold={min_similarity})")
                return results
            
            # If still none, return the most similar one (if any)
            if len(self.functions) > 0:
                max_idx = np.argmax(similarities)
                if max_idx < len(self.functions) and self.functions[max_idx] is not None:
                    max_sim = similarities[max_idx]
                    if max_sim > 0.1:  # As long as there's some similarity
                        func = self.functions[max_idx]
                        func.update_usage()
                        self.logger.debug(f"Returning most similar function: {func.expression} similarity={max_sim:.3f}")
                        return [func]
            
        except Exception as e:
            self.logger.error(f"Similarity computation failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        self.logger.debug("No similar functions found")
        return []
    
    def search_by_expression(self, pattern: str, limit: int = 10) -> List[FunctionMemory]:
        """
        Search by expression pattern.
        
        Args:
            pattern: Search pattern (e.g., "*sin*")
            limit: Maximum results to return
            
        Returns:
            List[FunctionMemory]: Matching functions
        """
        import fnmatch
        
        results = []
        for func in self.functions:
            if func is not None and fnmatch.fnmatch(func.expression.lower(), pattern.lower()):
                results.append(func)
                if len(results) >= limit:
                    break
        
        return results
    
    def search_by_tag(self, tag: str) -> List[FunctionMemory]:
        """
        Search by tag.
        
        Args:
            tag: Tag name
            
        Returns:
            List[FunctionMemory]: Matching functions
        """
        return [func for func in self.functions if func is not None and tag in func.tags]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory bank statistics.
        
        Returns:
            Dict: Statistics about the memory bank
        """
        # Filter out None values
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
        
        # Tag distribution
        tag_count = {}
        for func in valid_functions:
            for tag in func.tags:
                tag_count[tag] = tag_count.get(tag, 0) + 1
        
        # Source distribution
        source_count = {}
        for func in valid_functions:
            source_count[func.source] = source_count.get(func.source, 0) + 1
        
        # Accuracy statistics
        accuracies = [func.accuracy for func in valid_functions if func.accuracy is not None]
        
        # Usage statistics
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
        Save memory bank to file.
        
        Args:
            filename: Output filename, None for default
            
        Returns:
            bool: Success status
        """
        try:
            if METATHIN_AVAILABLE:
                # Metathin memory system auto-saves
                if hasattr(self.memory, 'flush'):
                    self.memory.flush()
                self.logger.info("Memory bank saved")
                return True
            else:
                # Simplified manual save
                if filename is None:
                    filename = "function_memory_backup.json"
                
                # Filter out None values
                valid_functions = [func for func in self.functions if func is not None]
                data = [func.to_dict() for func in valid_functions]
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"Memory bank saved to {filename}")
                return True
                
        except Exception as e:
            self.logger.error(f"Save failed: {e}")
            return False
    
    def load(self, filename: str) -> bool:
        """
        Load memory bank from file.
        
        Args:
            filename: Input filename
            
        Returns:
            bool: Success status
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Clear current memory
            self.clear()
            
            # Load new memories
            success_count = 0
            for item in data:
                try:
                    func = FunctionMemory.from_dict(item)
                    if self.add(func):
                        success_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to load single function: {e}")
                    continue
            
            self.logger.info(f"Loaded {success_count} functions from {filename}")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Load failed: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all memories."""
        self.functions.clear()
        self.feature_matrix = None
        self._need_rebuild_index = False
        
        if METATHIN_AVAILABLE:
            try:
                self.memory.clear()
            except:
                pass
        else:
            try:
                if hasattr(self.memory, 'clear'):
                    self.memory.clear()
            except:
                pass
        
        self.logger.info("Memory bank cleared")
    
    def __len__(self) -> int:
        """Return number of valid functions."""
        return len([f for f in self.functions if f is not None])
    
    def __iter__(self):
        """Iterator, skipping None values."""
        return (f for f in self.functions if f is not None)
    
    def __getitem__(self, idx: int) -> FunctionMemory:
        """Get function by index."""
        if idx < 0:
            idx = len(self.functions) + idx
        if 0 <= idx < len(self.functions) and self.functions[idx] is not None:
            return self.functions[idx]
        raise IndexError(f"Index {idx} out of range or function is None")