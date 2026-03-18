"""
Similarity Matcher - Function Similarity Search Based on Feature Vectors
==========================================================================

Finds similar functions by comparing feature vectors using various distance metrics.
Essential for retrieving known functions that match observed patterns.

Design Philosophy:
    - Efficient indexing: Uses KD-Tree for fast nearest neighbor search
    - Multiple metrics: Supports cosine, euclidean, manhattan, and mahalanobis distances
    - Dimensionality reduction: Optional PCA for high-dimensional features
    - Scalable: Handles large datasets with batch processing
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import logging
from enum import Enum
from dataclasses import dataclass, field
import pickle

# Lazy imports to avoid circular dependencies
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    StandardScaler = None
    PCA = None


class DistanceMetric(Enum):
    """Distance metric enumeration."""
    COSINE = "cosine"          # Cosine distance: 1 - cos(u,v)
    EUCLIDEAN = "euclidean"    # Euclidean distance: ||u - v||₂
    MANHATTAN = "manhattan"    # Manhattan distance: ||u - v||₁
    MAHALANOBIS = "mahalanobis" # Mahalanobis distance (requires covariance)


@dataclass
class MatchResult:
    """
    Match result data class.
    
    Attributes:
        index: Index in the original dataset
        score: Similarity score (higher is more similar, range [0,1])
        distance: Distance value (metric-dependent)
        metadata: Associated metadata (function type, parameters, etc.)
    """
    index: int
    score: float
    distance: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Sort by score descending (higher score first)."""
        return self.score > other.score


class SimilarityMatcher:
    """
    Similarity Matcher - Finds similar functions using feature vectors.
    
    Builds an efficient index structure for fast similarity search and supports
    multiple distance metrics. Can optionally apply PCA for dimensionality reduction.
    
    Parameters:
        metric: Distance metric (cosine, euclidean, manhattan, mahalanobis)
        threshold: Similarity threshold for filtering results [0,1]
        use_pca: Whether to apply PCA for dimensionality reduction
        n_components: Number of PCA components (if use_pca=True)
    
    Example:
        >>> matcher = SimilarityMatcher(metric=DistanceMetric.COSINE, threshold=0.7)
        >>> 
        >>> # Build index from features
        >>> matcher.build_index(features, metadata_list)
        >>> 
        >>> # Find similar functions
        >>> results = matcher.find_similar(query_features, k=5)
        >>> for r in results:
        ...     print(f"Score: {r.score:.3f}, Metadata: {r.metadata}")
    """
    
    def __init__(self,
                 metric: Union[str, DistanceMetric] = DistanceMetric.COSINE,
                 threshold: float = 0.7,
                 use_pca: bool = False,
                 n_components: Optional[int] = None):
        """
        Initialize similarity matcher.
        
        Args:
            metric: Distance metric to use
            threshold: Similarity threshold for filtering [0,1]
            use_pca: Whether to apply PCA
            n_components: Number of PCA components
        """
        if isinstance(metric, str):
            self.metric = DistanceMetric(metric)
        else:
            self.metric = metric
        
        self.threshold = threshold
        self.use_pca = use_pca
        self.n_components = n_components
        
        self.logger = logging.getLogger("metathin_sci.core.SimilarityMatcher")
        
        # Data storage
        self.features: Optional[np.ndarray] = None
        self.metadata_list: List[Dict[str, Any]] = []
        self.kdtree: Optional[KDTree] = None
        
        # Preprocessing objects
        self.pca = None
        self.scaler = None
        
        # Statistics
        self.n_samples = 0
        self.feature_dim = 0
        
        self.logger.info(f"Similarity matcher initialized with metric={self.metric.value}")
    
    def _cosine_distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """Compute cosine distance: 1 - cos(u,v)."""
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        
        if norm_u == 0 or norm_v == 0:
            return 1.0
        
        cos_sim = np.dot(u, v) / (norm_u * norm_v)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        
        return 1.0 - cos_sim
    
    def _cosine_similarity(self, u: np.ndarray, v: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        
        if norm_u == 0 or norm_v == 0:
            return 0.0
        
        cos_sim = np.dot(u, v) / (norm_u * norm_v)
        return np.clip(cos_sim, -1.0, 1.0)
    
    def distance_to_score(self, distance: float) -> float:
        """Convert distance to similarity score."""
        if self.metric == DistanceMetric.COSINE:
            return 1.0 - distance / 2.0
        else:
            return 1.0 / (1.0 + distance)
    
    def score_to_distance(self, score: float) -> float:
        """Convert similarity score to distance."""
        if self.metric == DistanceMetric.COSINE:
            return (1.0 - score) * 2.0
        else:
            return (1.0 - score) / score if score > 0 else float('inf')
    
    def _preprocess_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Preprocess features: normalization and optional PCA.
        
        Args:
            features: Input feature array
            fit: Whether to fit preprocessing objects (for training)
            
        Returns:
            np.ndarray: Preprocessed features
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        processed = features.copy().astype(np.float64)
        
        # Skip if sklearn not available
        if not SKLEARN_AVAILABLE:
            return processed
        
        # Standardize
        if fit:
            if StandardScaler is not None:
                self.scaler = StandardScaler()
                processed = self.scaler.fit_transform(processed)
        elif self.scaler is not None:
            processed = self.scaler.transform(processed)
        
        # PCA
        if self.use_pca and self.n_components and PCA is not None:
            if fit:
                self.pca = PCA(n_components=min(self.n_components, features.shape[1]))
                processed = self.pca.fit_transform(processed)
            elif self.pca is not None:
                processed = self.pca.transform(processed)
        
        return processed
    
    def build_index(self,
                   features: np.ndarray,
                   metadata_list: Optional[List[Dict]] = None) -> None:
        """
        Build feature index for similarity search.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            metadata_list: List of metadata dictionaries for each sample
            
        Raises:
            ValueError: If metadata_list length doesn't match features
        """
        if metadata_list is not None and len(metadata_list) != len(features):
            raise ValueError("metadata_list length must match features")
        
        self.n_samples = len(features)
        self.feature_dim = features.shape[1]
        
        # Preprocess features
        processed_features = self._preprocess_features(features, fit=True)
        
        self.features = processed_features
        self.metadata_list = metadata_list or [{} for _ in range(self.n_samples)]
        
        # Build KD-Tree (normalize for cosine distance)
        if self.metric == DistanceMetric.COSINE:
            norms = np.linalg.norm(processed_features, axis=1, keepdims=True)
            normalized = processed_features / (norms + 1e-8)
            self.kdtree = KDTree(normalized)
        else:
            self.kdtree = KDTree(processed_features)
        
        self.logger.info(f"Index built: {self.n_samples} samples, dimension {self.feature_dim}")
    
    def add_samples(self,
                   features: np.ndarray,
                   metadata_list: Optional[List[Dict]] = None) -> None:
        """
        Incrementally add samples to the index.
        
        Args:
            features: New feature matrix
            metadata_list: List of metadata dictionaries for new samples
        """
        if metadata_list is not None and len(metadata_list) != len(features):
            raise ValueError("metadata_list length must match features")
        
        # Preprocess new features
        new_features = self._preprocess_features(features, fit=False)
        
        # Append to existing data
        if self.features is None:
            self.features = new_features
            self.metadata_list = metadata_list or [{} for _ in range(len(new_features))]
        else:
            self.features = np.vstack([self.features, new_features])
            if metadata_list:
                self.metadata_list.extend(metadata_list)
            else:
                self.metadata_list.extend([{} for _ in range(len(new_features))])
        
        # Rebuild index
        self.build_index(self.features, self.metadata_list)
        
        self.logger.info(f"Added {len(features)} samples, total {self.n_samples}")
    
    def find_similar(self,
                    query_features: np.ndarray,
                    k: int = 5,
                    threshold: Optional[float] = None) -> List[MatchResult]:
        """
        Find similar samples to the query.
        
        Args:
            query_features: Feature vector to query
            k: Number of nearest neighbors to consider
            threshold: Similarity threshold (uses self.threshold if None)
            
        Returns:
            List[MatchResult]: Matches sorted by score descending
        """
        if self.kdtree is None:
            self.logger.warning("Index not built, cannot query")
            return []
        
        thresh = threshold if threshold is not None else self.threshold
        
        # Preprocess query
        query = self._preprocess_features(query_features.reshape(1, -1), fit=False)
        
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Query KD-Tree
        if self.metric == DistanceMetric.COSINE:
            norm = np.linalg.norm(query)
            query_normalized = query / (norm + 1e-8)
            distances, indices = self.kdtree.query(query_normalized, k=min(k, self.n_samples))
        else:
            distances, indices = self.kdtree.query(query, k=min(k, self.n_samples))
        
        # Handle different return types from KDTree
        if isinstance(distances, np.ndarray):
            if distances.ndim == 0:  # Scalar
                distances = [float(distances)]
                indices = [int(indices)] if isinstance(indices, (int, np.integer)) else [indices]
            elif distances.ndim == 1:  # 1D array
                distances = [float(d) for d in distances]
                indices = [int(idx) for idx in indices]
            else:  # 2D array
                distances = [float(d[0]) for d in distances]
                indices = [int(idx[0]) for idx in indices]
        elif isinstance(distances, (list, tuple)):
            # Already a list, ensure each element is scalar
            new_distances = []
            new_indices = []
            for i, (d, idx) in enumerate(zip(distances, indices)):
                if isinstance(d, (list, tuple, np.ndarray)):
                    new_distances.append(float(d[0]) if d else 0.0)
                else:
                    new_distances.append(float(d))
                
                if isinstance(idx, (list, tuple, np.ndarray)):
                    new_indices.append(int(idx[0]) if idx else 0)
                else:
                    new_indices.append(int(idx))
            distances = new_distances
            indices = new_indices
        
        # Convert to MatchResult objects
        results = []
        for dist, idx in zip(distances, indices):
            score = self.distance_to_score(dist)
            
            if score >= thresh:
                results.append(MatchResult(
                    index=int(idx),
                    score=float(score),
                    distance=float(dist),
                    metadata=self.metadata_list[idx] if idx < len(self.metadata_list) else {}
                ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def find_k_nearest(self,
                      query_features: np.ndarray,
                      k: int = 5) -> List[MatchResult]:
        """
        Find k nearest neighbors (no threshold filtering).
        
        Args:
            query_features: Feature vector to query
            k: Number of nearest neighbors
            
        Returns:
            List[MatchResult]: k nearest matches
        """
        if self.kdtree is None:
            self.logger.warning("Index not built, cannot query")
            return []
        
        query = self._preprocess_features(query_features.reshape(1, -1), fit=False)
        
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Query KD-Tree
        if self.metric == DistanceMetric.COSINE:
            norm = np.linalg.norm(query)
            query_normalized = query / (norm + 1e-8)
            distances, indices = self.kdtree.query(query_normalized, k=min(k, self.n_samples))
        else:
            distances, indices = self.kdtree.query(query, k=min(k, self.n_samples))
        
        # Handle different return types from KDTree
        if isinstance(distances, np.ndarray):
            if distances.ndim == 0:
                distances = [float(distances)]
                indices = [int(indices)]
            elif distances.ndim == 1:
                distances = [float(d) for d in distances]
                indices = [int(idx) for idx in indices]
            else:
                distances = [float(d[0]) for d in distances]
                indices = [int(idx[0]) for idx in indices]
        elif isinstance(distances, (list, tuple)):
            new_distances = []
            new_indices = []
            for d, idx in zip(distances, indices):
                if isinstance(d, (list, tuple, np.ndarray)):
                    new_distances.append(float(d[0]) if d else 0.0)
                else:
                    new_distances.append(float(d))
                
                if isinstance(idx, (list, tuple, np.ndarray)):
                    new_indices.append(int(idx[0]) if idx else 0)
                else:
                    new_indices.append(int(idx))
            distances = new_distances
            indices = new_indices
        
        # Convert to MatchResult objects
        results = []
        for dist, idx in zip(distances, indices):
            score = self.distance_to_score(dist)
            
            results.append(MatchResult(
                index=int(idx),
                score=float(score),
                distance=float(dist),
                metadata=self.metadata_list[idx] if idx < len(self.metadata_list) else {}
            ))
        
        return results
    
    def find_within_distance(self,
                            query_features: np.ndarray,
                            max_distance: float) -> List[MatchResult]:
        """
        Find all samples within a specified distance.
        
        Args:
            query_features: Feature vector to query
            max_distance: Maximum distance threshold
            
        Returns:
            List[MatchResult]: Matches within distance
        """
        if self.kdtree is None:
            self.logger.warning("Index not built, cannot query")
            return []
        
        query = self._preprocess_features(query_features.reshape(1, -1), fit=False)
        
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Query KD-Tree for points within ball
        if self.metric == DistanceMetric.COSINE:
            norm = np.linalg.norm(query)
            query_normalized = query / (norm + 1e-8)
            indices = self.kdtree.query_ball_point(query_normalized, max_distance)
        else:
            indices = self.kdtree.query_ball_point(query, max_distance)
        
        if not indices or not indices[0]:
            return []
        
        # Compute actual distances for found points
        results = []
        for idx in indices[0]:
            if self.metric == DistanceMetric.COSINE:
                ref = self.features[idx]
                dist = self._cosine_distance(query[0], ref)
            else:
                ref = self.features[idx]
                dist = np.linalg.norm(query[0] - ref)
            
            dist = float(dist)
            score = self.distance_to_score(dist)
            
            results.append(MatchResult(
                index=int(idx),
                score=float(score),
                distance=dist,
                metadata=self.metadata_list[idx]
            ))
        
        results.sort(key=lambda x: x.distance)
        
        return results
    
    def batch_find_similar(self,
                          query_features_batch: np.ndarray,
                          k: int = 5,
                          threshold: Optional[float] = None) -> List[List[MatchResult]]:
        """
        Find similar samples for multiple queries (batch processing).
        
        Args:
            query_features_batch: Batch of feature vectors
            k: Number of nearest neighbors per query
            threshold: Similarity threshold
            
        Returns:
            List[List[MatchResult]]: Results for each query
        """
        results = []
        for query in query_features_batch:
            matches = self.find_similar(query, k, threshold)
            results.append(matches)
        return results
    
    def compute_pairwise_similarity(self,
                                    features1: np.ndarray,
                                    features2: np.ndarray) -> np.ndarray:
        """
        Compute pairwise similarity between two sets of features.
        
        Args:
            features1: First feature matrix (n1, d)
            features2: Second feature matrix (n2, d)
            
        Returns:
            np.ndarray: Similarity matrix (n1, n2)
        """
        # Preprocess features
        f1 = self._preprocess_features(features1, fit=False)
        f2 = self._preprocess_features(features2, fit=False)
        
        # Compute similarity matrix
        if self.metric == DistanceMetric.COSINE:
            # Normalize for cosine similarity
            norms1 = np.linalg.norm(f1, axis=1, keepdims=True)
            norms2 = np.linalg.norm(f2, axis=1, keepdims=True)
            f1_norm = f1 / (norms1 + 1e-8)
            f2_norm = f2 / (norms2 + 1e-8)
            
            # Cosine similarity
            sim_matrix = np.dot(f1_norm, f2_norm.T)
            sim_matrix = np.clip(sim_matrix, -1.0, 1.0)
            # Convert to [0,1] range
            sim_matrix = (sim_matrix + 1.0) / 2.0
        else:
            # Distance matrix -> similarity matrix
            dist_matrix = cdist(f1, f2, metric=self.metric.value)
            sim_matrix = 1.0 / (1.0 + dist_matrix)
        
        return sim_matrix
    
    def get_neighborhood_stats(self, query_features: np.ndarray, radius: float) -> Dict:
        """
        Get statistical information about the neighborhood around a query point.
        
        Args:
            query_features: Query feature vector
            radius: Neighborhood radius
            
        Returns:
            Dict: Neighborhood statistics
        """
        neighbors = self.find_within_distance(query_features, radius)
        
        if not neighbors:
            return {
                'count': 0,
                'avg_distance': 0,
                'avg_score': 0,
                'std_distance': 0
            }
        
        distances = [n.distance for n in neighbors]
        scores = [n.score for n in neighbors]
        
        return {
            'count': len(neighbors),
            'avg_distance': float(np.mean(distances)),
            'avg_score': float(np.mean(scores)),
            'std_distance': float(np.std(distances)),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances))
        }
    
    def save(self, filename: str) -> bool:
        """
        Save matcher to file.
        
        Args:
            filename: Output filename
            
        Returns:
            bool: Success status
        """
        try:
            data = {
                'metric': self.metric.value,
                'threshold': self.threshold,
                'use_pca': self.use_pca,
                'n_components': self.n_components,
                'features': self.features.tolist() if self.features is not None else None,
                'metadata_list': self.metadata_list,
                'feature_dim': self.feature_dim,
                'n_samples': self.n_samples
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.info(f"Matcher saved to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Save failed: {e}")
            return False
    
    def load(self, filename: str) -> bool:
        """
        Load matcher from file.
        
        Args:
            filename: Input filename
            
        Returns:
            bool: Success status
        """
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.metric = DistanceMetric(data['metric'])
            self.threshold = data['threshold']
            self.use_pca = data['use_pca']
            self.n_components = data['n_components']
            self.feature_dim = data['feature_dim']
            self.n_samples = data['n_samples']
            
            if data['features'] is not None:
                self.features = np.array(data['features'])
            self.metadata_list = data['metadata_list']
            
            # Rebuild index
            if self.features is not None:
                self.build_index(self.features, self.metadata_list)
            
            self.logger.info(f"Matcher loaded from {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Load failed: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all data and reset the matcher."""
        self.features = None
        self.metadata_list = []
        self.kdtree = None
        self.pca = None
        self.scaler = None
        self.n_samples = 0
        self.feature_dim = 0
        self.logger.info("Matcher cleared")