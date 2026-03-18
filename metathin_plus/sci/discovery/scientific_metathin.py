"""
Scientific Discovery Agent - Top-Level Interface Integrating All Functionality
================================================================================

This module provides the main entry point for scientific discovery, orchestrating
the entire discovery pipeline from data ingestion to pattern discovery and learning.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict
import time
import logging
from pathlib import Path

try:
    from metathin import Metathin, MetathinConfig
    from metathin.core import MemoryManager
    METATHIN_AVAILABLE = True
except ImportError:
    METATHIN_AVAILABLE = False

from .adaptive_extrapolator import AdaptiveExtrapolator, SymbolicForm
from .report_generator import DiscoveryReport, DiscoveryPhase
from ..memory.function_memory import FunctionMemoryBank, FunctionMemory
from ..memory.pretrained import create_pretrained_bank, list_available_pretrained
from ..core.feature_extractor import FeatureExtractor
from ..core.similarity_matcher import SimilarityMatcher


class ScientificMetathin:
    """
    Scientific Discovery Agent - Main Entry Point.
    
    Integrates all discovery components (feature extraction, similarity matching,
    adaptive extrapolation, memory) into a cohesive pipeline for automated
    scientific discovery.
    
    Key Capabilities:
        - Real-time pattern discovery from data streams
        - Memory-assisted discovery using previously learned functions
        - Learning from discoveries to improve future performance
        - Comprehensive reporting and visualization
    
    Parameters:
        name: Agent name for identification
        memory_bank: External function memory bank (optional)
        pretrained: Whether to load pretrained memory banks
        pretrained_libraries: List of pretrained libraries to load
        N: Number of steps used for extrapolation
        delta: Error threshold for extrapolation
        enable_learning: Whether to enable learning from discoveries
        memory_path: Path for memory bank storage
        feature_dim: Feature dimension (must match FeatureExtractor output)
    """
    
    def __init__(self,
                 name: str = "ScientificMetathin",
                 memory_bank: Optional[FunctionMemoryBank] = None,
                 pretrained: bool = True,
                 pretrained_libraries: Optional[List[str]] = None,
                 N: int = 50,
                 delta: float = 0.1,
                 enable_learning: bool = True,
                 memory_path: Optional[str] = None,
                 feature_dim: int = 33):
        """
        Initialize Scientific Discovery Agent.
        
        Args:
            name: Agent name for logging and identification
            memory_bank: External function memory bank (if None, creates new one)
            pretrained: Whether to load pretrained memory banks
            pretrained_libraries: List of pretrained library names to load
            N: Number of steps for extrapolation
            delta: Error threshold for extrapolation
            enable_learning: Whether to enable learning from discoveries
            memory_path: Path for memory bank storage
            feature_dim: Feature dimension (must match FeatureExtractor output)
        """
        self.name = name
        self.enable_learning = enable_learning
        self.delta = delta
        self.N = N
        self.feature_dim = feature_dim
        
        self.logger = logging.getLogger(f"metathin_sci.{name}")
        
        # ===== 1. Initialize Memory Bank =====
        if memory_bank:
            self.memory_bank = memory_bank
            self.logger.info(f"Using external memory bank with {len(self.memory_bank)} functions")
        else:
            self.memory_bank = FunctionMemoryBank(
                memory_backend='json',
                memory_path=memory_path or f"{name}_memory.json"
            )
            
            if pretrained:
                libraries = pretrained_libraries or ['basic', 'physics', 'chemistry']
                pretrained_bank = create_pretrained_bank(
                    library_keys=libraries,
                    limits={'basic': 15, 'physics': 10, 'chemistry': 8}
                )
                for func in pretrained_bank:
                    # Ensure pretrained function feature dimensions match current setting
                    if len(func.feature_vector) != feature_dim:
                        if len(func.feature_vector) < feature_dim:
                            # Pad to target dimension
                            new_features = np.pad(
                                func.feature_vector, 
                                (0, feature_dim - len(func.feature_vector)),
                                mode='constant',
                                constant_values=0
                            )
                        else:
                            # Truncate to target dimension
                            new_features = func.feature_vector[:feature_dim]
                        func.feature_vector = new_features
                    self.memory_bank.add(func)
                self.logger.info(f"Loaded pretrained memory banks: {libraries}")
        
        # ===== 2. Initialize Feature Extractor =====
        self.feature_extractor = FeatureExtractor(normalize=True)
        
        # ===== 3. Initialize Similarity Matcher =====
        self.similarity_matcher = SimilarityMatcher(threshold=0.7)
        self._update_matcher_index()
        
        # ===== 4. Initialize Adaptive Extrapolator =====
        self.extrapolator = AdaptiveExtrapolator(N=N, delta=delta)
        
        # ===== 5. Statistics Tracking =====
        self.stats = {
            'total_discoveries': 0,
            'memory_hits': 0,
            'memory_misses': 0,
            'learning_events': 0,
            'start_time': time.time()
        }
        
        # ===== 6. Discovery History =====
        self.current_report: Optional[DiscoveryReport] = None
        self.discovery_history: List[DiscoveryReport] = []
        
        self.logger.info(f"✅ ScientificMetathin '{name}' initialized successfully")
        self.logger.info(f"   Memory bank size: {len(self.memory_bank)}")
        self.logger.info(f"   Feature dimension: {feature_dim}")
        self.logger.info(f"   Extrapolation parameters: N={N}, delta={delta}")
    
    def _update_matcher_index(self):
        """Update similarity matcher index with current memory bank."""
        if len(self.memory_bank) == 0:
            return
        
        features = []
        metadata = []
        
        for func in self.memory_bank:
            # Ensure feature dimension consistency
            if len(func.feature_vector) != self.feature_dim:
                if len(func.feature_vector) < self.feature_dim:
                    feat = np.pad(func.feature_vector, 
                                  (0, self.feature_dim - len(func.feature_vector)),
                                  mode='constant', constant_values=0)
                else:
                    feat = func.feature_vector[:self.feature_dim]
            else:
                feat = func.feature_vector
            
            features.append(feat)
            metadata.append({
                'id': func.id,
                'expression': func.expression,
                'accuracy': func.accuracy
            })
        
        features = np.array(features)
        self.similarity_matcher.build_index(features, metadata)
        
        self.logger.debug(f"Similarity matcher index updated with {len(features)} functions")
    
    def _get_memory_assistance(self, data_segment: np.ndarray) -> Optional[Dict]:
        """
        Get assistance from memory bank for a data segment.
        
        Args:
            data_segment: Data segment to analyze
            
        Returns:
            Optional[Dict]: Assistance information or None if no matches
        """
        if len(self.memory_bank) == 0:
            return None
        
        # Extract features
        features = self.feature_extractor.extract(data_segment)
        
        # Ensure feature dimension consistency
        if len(features) != self.feature_dim:
            if len(features) < self.feature_dim:
                features = np.pad(features, 
                                  (0, self.feature_dim - len(features)),
                                  mode='constant', constant_values=0)
            else:
                features = features[:self.feature_dim]
        
        # Find similar functions
        matches = self.similarity_matcher.find_similar(
            features, k=3, threshold=0.7
        )
        
        if matches:
            self.stats['memory_hits'] += 1
            
            assistance = {
                'similar_functions': [],
                'best_match': None,
                'suggested_forms': []
            }
            
            for match in matches:
                func_info = {
                    'expression': match.metadata['expression'],
                    'similarity': match.score,
                    'accuracy': match.metadata['accuracy']
                }
                assistance['similar_functions'].append(func_info)
                
                if match.score > 0.8 and not assistance['best_match']:
                    assistance['best_match'] = func_info
            
            self.logger.debug(f"Memory assistance: found {len(matches)} similar functions")
            return assistance
        else:
            self.stats['memory_misses'] += 1
            return None
    
    def _learn_from_discovery(self, form: SymbolicForm, data: np.ndarray):
        """
        Learn a discovered pattern and store it in memory.
        
        Args:
            form: Discovered symbolic form
            data: Data segment where the pattern was found
        """
        if not self.enable_learning:
            return
        
        try:
            # Extract features from the data segment
            features = self.feature_extractor.extract(data)
            
            # Ensure feature dimension consistency
            if len(features) != self.feature_dim:
                if len(features) < self.feature_dim:
                    features = np.pad(features, 
                                      (0, self.feature_dim - len(features)),
                                      mode='constant', constant_values=0)
                else:
                    features = features[:self.feature_dim]
            
            # Create memory entry
            func_memory = FunctionMemory(
                expression=form.expression,
                parameters=form.params,
                feature_vector=features,
                accuracy=1.0 - form.error / self.delta,
                tags=['discovered'],
                source='scientific_metathin'
            )
            
            # Add to memory bank
            self.memory_bank.add(func_memory)
            self.stats['learning_events'] += 1
            
            self.logger.info(f"Learned new function: {form.expression}, error={form.error:.6f}")
            
            # Update similarity matcher index
            self._update_matcher_index()
            
        except Exception as e:
            self.logger.error(f"Learning failed: {e}")
    
    def discover(self,
                y_data: np.ndarray,
                x_data: Optional[np.ndarray] = None,
                use_memory: bool = True,
                learn: bool = True) -> DiscoveryReport:
        """
        Discover patterns in data.
        
        Args:
            y_data: Y values (dependent variable)
            x_data: X values (independent variable), defaults to indices
            use_memory: Whether to use memory assistance
            learn: Whether to learn discovered patterns
            
        Returns:
            DiscoveryReport: Report of discovered patterns
            
        Raises:
            ValueError: If x_data and y_data lengths don't match
        """
        if x_data is None:
            x_data = np.arange(len(y_data))
        
        if len(x_data) != len(y_data):
            raise ValueError("x_data and y_data must have the same length")
        
        self.logger.info(f"Starting discovery process, data length: {len(y_data)}")
        
        # Initialize report
        report = DiscoveryReport(
            title=f"Scientific Discovery Report - {self.name}",
            data_source="Real-time discovery"
        )
        
        # Reset extrapolator
        self.extrapolator.reset()
        
        # Process data point by point
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            y_pred = self.extrapolator.predict(x)
            self.extrapolator.update(x, y)
            
            # Check memory assistance periodically
            if use_memory and i % self.N == 0 and i > self.N:
                segment = y_data[max(0, i-self.N):i]
                assistance = self._get_memory_assistance(segment)
                if assistance and assistance.get('best_match'):
                    self.logger.info(f"Memory assistance suggested: {assistance['best_match']['expression']}")
        
        # Convert discovered forms to report phases
        for form in self.extrapolator.get_history():
            phase = DiscoveryPhase(
                formula=form.expression,
                params=form.params,
                range=form.valid_range,
                error=form.error,
                confidence=1.0 - form.error / self.delta
            )
            report.add_phase(phase)
            
            # Learn from discovery if enabled
            if learn:
                start_idx = max(0, int(form.valid_range[0]))
                end_idx = min(len(y_data), int(form.valid_range[1]) + 1)
                segment = y_data[start_idx:end_idx]
                self._learn_from_discovery(form, segment)
        
        # Update statistics
        self.stats['total_discoveries'] += 1
        self.current_report = report
        self.discovery_history.append(report)
        
        self.logger.info(f"Discovery complete, found {len(report.phases)} patterns")
        
        return report
    
    def analyze(self,
               data: np.ndarray,
               x_data: Optional[np.ndarray] = None,
               save_report: bool = True) -> Dict[str, Any]:
        """
        Analyze data and return structured results.
        
        Args:
            data: Y values (dependent variable)
            x_data: X values (independent variable), defaults to indices
            save_report: Whether to save the report to file
            
        Returns:
            Dict: Structured analysis results
        """
        report = self.discover(data, x_data)
        
        result = {
            'n_phases': len(report.phases),
            'phases': [],
            'statistics': self.get_statistics(),
            'memory_usage': len(self.memory_bank)
        }
        
        for phase in report.phases:
            result['phases'].append({
                'formula': phase.formula,
                'range': phase.range,
                'error': phase.error,
                'confidence': phase.confidence
            })
        
        if save_report:
            filename = f"discovery_report_{int(time.time())}.json"
            report.save(filename)
            result['report_file'] = filename
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the agent's performance.
        
        Returns:
            Dict: Statistics including runtime, memory usage, hit rates, etc.
        """
        runtime = time.time() - self.stats['start_time']
        
        extrapolator_stats = self.extrapolator.get_stats()
        memory_stats = self.memory_bank.get_statistics() if hasattr(self.memory_bank, 'get_statistics') else {}
        
        total_memory_queries = self.stats['memory_hits'] + self.stats['memory_misses']
        memory_hit_rate = self.stats['memory_hits'] / total_memory_queries if total_memory_queries > 0 else 0
        
        return {
            'name': self.name,
            'runtime': runtime,
            'total_discoveries': self.stats['total_discoveries'],
            'memory_hits': self.stats['memory_hits'],
            'memory_misses': self.stats['memory_misses'],
            'memory_hit_rate': memory_hit_rate,
            'learning_events': self.stats['learning_events'],
            'memory_size': len(self.memory_bank),
            'current_extrapolator': extrapolator_stats,
            'memory_statistics': memory_stats
        }
    
    def save_memory(self, filename: Optional[str] = None) -> bool:
        """
        Save memory bank to file.
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            bool: Success status
        """
        if filename is None:
            filename = f"{self.name}_memory.json"
        return self.memory_bank.save(filename)
    
    def load_memory(self, filename: str) -> bool:
        """
        Load memory bank from file.
        
        Args:
            filename: Input filename
            
        Returns:
            bool: Success status
        """
        success = self.memory_bank.load(filename)
        if success:
            self._update_matcher_index()
            self.logger.info(f"Loaded memory bank from {filename}")
        return success
    
    def clear_memory(self):
        """Clear memory bank."""
        self.memory_bank.clear()
        self._update_matcher_index()
        self.logger.info("Memory bank cleared")
    
    def get_report(self) -> Optional[DiscoveryReport]:
        """Get the current discovery report."""
        return self.current_report
    
    def get_history(self) -> List[DiscoveryReport]:
        """Get history of all discovery reports."""
        return self.discovery_history.copy()
    
    def reset(self):
        """Reset the agent to initial state."""
        self.extrapolator.reset()
        self.current_report = None
        self.stats = {
            'total_discoveries': 0,
            'memory_hits': 0,
            'memory_misses': 0,
            'learning_events': 0,
            'start_time': time.time()
        }
        self.logger.info("Agent reset")