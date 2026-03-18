"""
Metathin+Sci Memory Module
=====================================================

Provides storage, management, and retrieval capabilities for function memories.
This module enables the scientific discovery agent to remember and reuse
previously discovered mathematical patterns.

Contains:
    - FunctionMemory: Individual function memory data class
    - FunctionMemoryBank: Function memory bank manager
    - pretrained: Pre-trained function libraries for common mathematical patterns

The memory system is designed for:
    - Persistent storage of discovered functions
    - Fast similarity-based retrieval
    - Learning from past discoveries
    - Sharing knowledge across discovery tasks
"""

from .function_memory import FunctionMemory, FunctionMemoryBank

# Note: pretrained is a submodule and should be imported via .pretrained
from . import pretrained

__all__ = [
    'FunctionMemory',
    """
    FunctionMemory: Individual function memory entry.
    
    Represents a single discovered function with its expression, parameters,
    feature vector, and metadata. Used for storing and retrieving learned patterns.
    
    Attributes:
        id: Unique identifier for the memory entry
        expression: String representation of the function
        parameters: Dictionary of parameter values
        feature_vector: Feature vector representation (for similarity matching)
        accuracy: How well this function fits its original data (0-1)
        tags: List of tags for categorization
        source: Source of the function (e.g., 'discovery', 'pretrained')
        created_at: Creation timestamp
        usage_count: Number of times this memory has been used
        last_accessed: Last access timestamp
    """,
    
    'FunctionMemoryBank',
    """
    FunctionMemoryBank: Manager for collections of function memories.
    
    Provides persistent storage, indexing, and retrieval of function memories.
    Supports multiple storage backends and efficient similarity search.
    
    Features:
        - Persistent storage (JSON/SQLite backends)
        - Add, remove, and update memories
        - Find similar functions by feature vectors
        - Get statistics about memory usage
        - Export/import capabilities
    """,
    
    'pretrained',
    """
    pretrained: Pre-trained function library module.
    
    Provides access to pre-defined collections of common mathematical functions
    across different domains. These serve as initial knowledge for the discovery agent.
    
    Available libraries:
        - basic: Fundamental mathematical functions (polynomials, trigonometric, etc.)
        - physics: Common physical models (oscillators, decay, wave equations)
        - chemistry: Chemical kinetics and reaction models
        - biology: Population dynamics and growth models
    
    Functions:
        create_pretrained_bank: Create a memory bank with pre-trained functions
        list_available_pretrained: List all available pre-trained libraries
        get_pretrained_library: Get a specific pre-trained library
    """
]

# ============================================================
# Usage Examples
# ============================================================
"""
>>> from metathin_plus.sci.memory import FunctionMemory, FunctionMemoryBank
>>> import numpy as np
>>> 
>>> # 1. Create individual function memory
>>> features = np.array([0.5, 0.2, 0.8, 0.1])  # Example feature vector
>>> 
>>> memory = FunctionMemory(
...     expression="a*sin(ω*x + φ)",
...     parameters={'a': 2.0, 'ω': 1.5, 'φ': 0.0},
...     feature_vector=features,
...     accuracy=0.95,
...     tags=['periodic', 'trigonometric'],
...     source='discovery'
... )
>>> 
>>> # 2. Create memory bank
>>> bank = FunctionMemoryBank(memory_backend='json', memory_path='my_memory.json')
>>> 
>>> # 3. Add memory to bank
>>> bank.add(memory)
>>> 
>>> # 4. Find similar functions
>>> query_features = np.array([0.45, 0.25, 0.75, 0.15])
>>> similar = bank.find_similar(query_features, k=5)
>>> 
>>> for mem in similar:
...     print(f"Expression: {mem.expression}, Similarity: {mem.similarity_score}")
>>> 
>>> # 5. Load pre-trained library
>>> from metathin_plus.sci.memory import pretrained
>>> 
>>> # List available libraries
>>> libraries = pretrained.list_available_pretrained()
>>> print("Available libraries:", libraries)
>>> 
>>> # Create bank with pre-trained functions
>>> pretrained_bank = pretrained.create_pretrained_bank(
...     library_keys=['basic', 'physics'],
...     limits={'basic': 10, 'physics': 5}
... )
>>> 
>>> # 6. Save and load
>>> bank.save('my_memory_backup.json')
>>> bank.load('my_memory_backup.json')
"""

# ============================================================
# Module Initialization Logging
# ============================================================

import logging
logger = logging.getLogger(__name__)
logger.debug("Metathin+Sci memory module initialized successfully")