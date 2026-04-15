# metathin_sci/core/similarity_matcher.py
"""
Similarity Matcher - Function Similarity Search Based on Feature Vectors | 相似度匹配器 - 基于特征向量的函数相似度查找
=======================================================================================================================

Provides efficient similarity search for function features using KD-tree indexing.
Supports multiple distance metrics and batch queries.

提供基于特征向量的函数相似度查找，使用KD-tree索引实现高效检索。
支持多种距离度量和批量查询。

Features | 特性:
    - Fast retrieval: KD-tree index for efficient nearest neighbor search | 快速检索：KD-tree索引
    - Multiple metrics: Cosine, Euclidean, Manhattan, Mahalanobis | 多种度量
    - Batch queries: Process multiple queries efficiently | 批量查询
    - Persistence: Save and load index state | 持久化：保存和加载索引状态

Distance Metrics | 距离度量:
    - COSINE: Cosine distance (1 - cosine similarity) | 余弦距离
    - EUCLIDEAN: Euclidean distance | 欧几里得距离
    - MANHATTAN: Manhattan distance (L1) | 曼哈顿距离
    - MAHALANOBIS: Mahalanobis distance (requires covariance) | 马哈拉诺比斯距离
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import logging
from enum import Enum
from dataclasses import dataclass, field
import pickle

# Optional sklearn imports | 可选的sklearn导入
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    StandardScaler = None
    PCA = None


# ============================================================
# Distance Metric Enum | 距离度量枚举
# ============================================================

class DistanceMetric(Enum):
    """Distance metric enumeration | 距离度量枚举"""
    COSINE = "cosine"           # Cosine distance | 余弦距离
    EUCLIDEAN = "euclidean"     # Euclidean distance | 欧几里得距离
    MANHATTAN = "manhattan"     # Manhattan distance (L1) | 曼哈顿距离
    MAHALANOBIS = "mahalanobis" # Mahalanobis distance | 马哈拉诺比斯距离


# ============================================================
# Match Result Data Class | 匹配结果数据类
# ============================================================

@dataclass
class MatchResult:
    """
    Match result data class | 匹配结果数据类
    
    Contains information about a single similarity match.
    
    包含单个相似度匹配的信息。
    
    Attributes | 属性:
        index: Index in the feature matrix | 特征矩阵中的索引
        score: Similarity score (0-1, higher = more similar) | 相似度分数
        distance: Distance value (lower = more similar) | 距离值
        metadata: Additional metadata | 额外元数据
    """
    index: int
    score: float
    distance: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Sort by score descending | 按分数降序排序"""
        return self.score > other.score


# ============================================================
# Similarity Matcher | 相似度匹配器
# ============================================================

class SimilarityMatcher:
    """
    Similarity Matcher - Efficient feature-based similarity search | 相似度匹配器 - 基于特征向量的高效相似度查找
    
    Uses KD-tree indexing for fast nearest neighbor search.
    Supports multiple distance metrics and incremental updates.
    
    使用KD-tree索引实现快速最近邻搜索。
    支持多种距离度量和增量更新。
    
    Example | 示例:
        >>> matcher = SimilarityMatcher(metric=DistanceMetric.COSINE, threshold=0.7)
        >>> 
        >>> # Build index | 构建索引
        >>> matcher.build_index(features, metadata_list)
        >>> 
        >>> # Find similar items | 查找相似项
        >>> results = matcher.find_similar(query_features, k=5)
        >>> 
        >>> # Save for later | 保存供后续使用
        >>> matcher.save("matcher.pkl")
    """
    
    def __init__(self,
                 metric: Union[str, DistanceMetric] = DistanceMetric.COSINE,
                 threshold: float = 0.7,
                 use_pca: bool = False,
                 n_components: Optional[int] = None):
        """
        Initialize similarity matcher | 初始化相似度匹配器
        
        Args | 参数:
            metric: Distance metric | 距离度量
            threshold: Similarity threshold (0-1) | 相似度阈值
            use_pca: Whether to use PCA for dimensionality reduction | 是否使用PCA降维
            n_components: Number of PCA components | PCA组件数量
        """
        # Parse metric | 解析度量
        if isinstance(metric, str):
            self.metric = DistanceMetric(metric)
        else:
            self.metric = metric
        
        self.threshold = threshold
        self.use_pca = use_pca
        self.n_components = n_components
        
        self.logger = logging.getLogger("metathin_sci.core.SimilarityMatcher")
        
        # Data storage | 数据存储
        self.features: Optional[np.ndarray] = None
        self.metadata_list: List[Dict[str, Any]] = []
        self.kdtree: Optional[KDTree] = None
        
        # Preprocessing | 预处理
        self.pca = None
        self.scaler = None
        
        # Statistics | 统计
        self.n_samples = 0
        self.feature_dim = 0
        
        self.logger.info(f"SimilarityMatcher initialized: metric={self.metric.value}, threshold={threshold} | 初始化完成")
    
    # ============================================================
    # Distance and Similarity Functions | 距离和相似度函数
    # ============================================================
    
    def _cosine_distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Compute cosine distance | 计算余弦距离
        
        distance = 1 - cosine_similarity
        
        Args | 参数:
            u: First vector | 第一个向量
            v: Second vector | 第二个向量
            
        Returns | 返回:
            float: Cosine distance in [0, 2] | 余弦距离
        """
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        
        if norm_u == 0 or norm_v == 0:
            return 1.0
        
        cos_sim = np.dot(u, v) / (norm_u * norm_v)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        
        return 1.0 - cos_sim
    
    def _cosine_similarity(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Compute cosine similarity | 计算余弦相似度
        
        Args | 参数:
            u: First vector | 第一个向量
            v: Second vector | 第二个向量
            
        Returns | 返回:
            float: Cosine similarity in [-1, 1] | 余弦相似度
        """
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        
        if norm_u == 0 or norm_v == 0:
            return 0.0
        
        cos_sim = np.dot(u, v) / (norm_u * norm_v)
        return np.clip(cos_sim, -1.0, 1.0)
    
    def distance_to_score(self, distance: float) -> float:
        """
        Convert distance to similarity score | 将距离转换为相似度分数
        
        Maps distance to [0, 1] where higher = more similar.
        
        将距离映射到[0,1]范围，值越大表示越相似。
        
        Args | 参数:
            distance: Distance value | 距离值
            
        Returns | 返回:
            float: Similarity score (0-1) | 相似度分数
        """
        if self.metric == DistanceMetric.COSINE:
            # Cosine distance: 0-2, map to 0-1 | 余弦距离：0-2，映射到0-1
            return 1.0 - distance / 2.0
        else:
            # Other metrics: use reciprocal mapping | 其他度量：使用倒数映射
            return 1.0 / (1.0 + distance)
    
    def score_to_distance(self, score: float) -> float:
        """
        Convert similarity score to distance | 将相似度分数转换为距离
        
        Args | 参数:
            score: Similarity score (0-1) | 相似度分数
            
        Returns | 返回:
            float: Distance value | 距离值
        """
        if self.metric == DistanceMetric.COSINE:
            return (1.0 - score) * 2.0
        else:
            return (1.0 - score) / score if score > 0 else float('inf')
    
    # ============================================================
    # Preprocessing | 预处理
    # ============================================================
    
    def _preprocess_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Preprocess features (normalization, PCA) | 预处理特征（归一化、PCA）
        
        Args | 参数:
            features: Input features | 输入特征
            fit: Whether to fit preprocessing parameters | 是否拟合预处理参数
            
        Returns | 返回:
            np.ndarray: Preprocessed features | 预处理后的特征
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        processed = features.copy().astype(np.float64)
        
        # Normalization (Z-score) | 归一化
        if not SKLEARN_AVAILABLE:
            return processed
        
        if fit:
            if StandardScaler is not None:
                self.scaler = StandardScaler()
                processed = self.scaler.fit_transform(processed)
        elif self.scaler is not None:
            processed = self.scaler.transform(processed)
        
        # PCA dimensionality reduction | PCA降维
        if self.use_pca and self.n_components and PCA is not None:
            if fit:
                n_comp = min(self.n_components, features.shape[1])
                self.pca = PCA(n_components=n_comp)
                processed = self.pca.fit_transform(processed)
            elif self.pca is not None:
                processed = self.pca.transform(processed)
        
        return processed
    
    # ============================================================
    # Index Building | 索引构建
    # ============================================================
    
    def build_index(self,
                   features: np.ndarray,
                   metadata_list: Optional[List[Dict]] = None) -> None:
        """
        Build feature index for fast retrieval | 构建特征索引
        
        Args | 参数:
            features: Feature matrix (n_samples, n_features) | 特征矩阵
            metadata_list: Optional metadata for each sample | 每个样本的元数据
            
        Raises | 抛出:
            ValueError: If metadata length doesn't match features | 元数据长度不匹配
        """
        if metadata_list is not None and len(metadata_list) != len(features):
            raise ValueError(f"metadata_list length ({len(metadata_list)}) must match features ({len(features)}) | 长度必须相同")
        
        self.n_samples = len(features)
        self.feature_dim = features.shape[1]
        
        # Preprocess features | 预处理特征
        processed_features = self._preprocess_features(features, fit=True)
        
        self.features = processed_features
        self.metadata_list = metadata_list or [{} for _ in range(self.n_samples)]
        
        # Build KD-tree | 构建KD-tree
        if self.metric == DistanceMetric.COSINE:
            # For cosine distance, normalize vectors | 对于余弦距离，归一化向量
            norms = np.linalg.norm(processed_features, axis=1, keepdims=True)
            normalized = processed_features / (norms + 1e-8)
            self.kdtree = KDTree(normalized)
        else:
            self.kdtree = KDTree(processed_features)
        
        self.logger.info(f"Index built: {self.n_samples} samples, {self.feature_dim} dimensions | 索引构建完成")
    
    def add_samples(self,
                   features: np.ndarray,
                   metadata_list: Optional[List[Dict]] = None) -> None:
        """
        Incrementally add samples (rebuilds index) | 增量添加样本（重建索引）
        
        Args | 参数:
            features: New feature matrix | 新特征矩阵
            metadata_list: Metadata for new samples | 新样本的元数据
        """
        if metadata_list is not None and len(metadata_list) != len(features):
            raise ValueError(f"metadata_list length ({len(metadata_list)}) must match features ({len(features)}) | 长度必须相同")
        
        new_features = self._preprocess_features(features, fit=False)
        
        if self.features is None:
            self.features = new_features
            self.metadata_list = metadata_list or [{} for _ in range(len(new_features))]
        else:
            self.features = np.vstack([self.features, new_features])
            if metadata_list:
                self.metadata_list.extend(metadata_list)
            else:
                self.metadata_list.extend([{} for _ in range(len(new_features))])
        
        # Rebuild index | 重建索引
        self.build_index(self.features, self.metadata_list)
        
        self.logger.info(f"Added {len(features)} samples, total {self.n_samples} | 已添加")
    
    # ============================================================
    # Query Methods | 查询方法
    # ============================================================
    
    def find_similar(self,
                    query_features: np.ndarray,
                    k: int = 5,
                    threshold: Optional[float] = None) -> List[MatchResult]:
        """
        Find similar samples | 查找相似样本
        
        Args | 参数:
            query_features: Query feature vector | 查询特征向量
            k: Number of results to return | 返回结果数量
            threshold: Similarity threshold (uses self.threshold if None) | 相似度阈值
            
        Returns | 返回:
            List[MatchResult]: List of matching results (sorted by similarity) | 匹配结果列表
        """
        if self.kdtree is None:
            self.logger.warning("Index not built, cannot query | 索引未构建，无法查询")
            return []
        
        thresh = threshold if threshold is not None else self.threshold
        
        # Preprocess query | 预处理查询
        query = self._preprocess_features(query_features.reshape(1, -1), fit=False)
        
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Query based on metric | 根据度量查询
        if self.metric == DistanceMetric.COSINE:
            norm = np.linalg.norm(query)
            query_normalized = query / (norm + 1e-8)
            distances, indices = self.kdtree.query(query_normalized, k=min(k, self.n_samples))
        else:
            distances, indices = self.kdtree.query(query, k=min(k, self.n_samples))
        
        # Normalize distances | 归一化距离
        distances = self._normalize_distances(distances, indices)
        
        # Build results | 构建结果
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
        
        self.logger.debug(f"Found {len(results)} similar samples | 找到 {len(results)} 个相似样本")
        return results
    
    def find_k_nearest(self,
                      query_features: np.ndarray,
                      k: int = 5) -> List[MatchResult]:
        """
        Find k nearest neighbors (no threshold filtering) | 查找K个最近邻（无阈值过滤）
        
        Args | 参数:
            query_features: Query feature vector | 查询特征向量
            k: Number of results to return | 返回结果数量
            
        Returns | 返回:
            List[MatchResult]: List of matching results | 匹配结果列表
        """
        if self.kdtree is None:
            self.logger.warning("Index not built, cannot query | 索引未构建，无法查询")
            return []
        
        query = self._preprocess_features(query_features.reshape(1, -1), fit=False)
        
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        if self.metric == DistanceMetric.COSINE:
            norm = np.linalg.norm(query)
            query_normalized = query / (norm + 1e-8)
            distances, indices = self.kdtree.query(query_normalized, k=min(k, self.n_samples))
        else:
            distances, indices = self.kdtree.query(query, k=min(k, self.n_samples))
        
        distances = self._normalize_distances(distances, indices)
        
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
        Find all samples within a distance radius | 查找指定距离内的所有样本
        
        Args | 参数:
            query_features: Query feature vector | 查询特征向量
            max_distance: Maximum distance | 最大距离
            
        Returns | 返回:
            List[MatchResult]: List of matching results | 匹配结果列表
        """
        if self.kdtree is None:
            self.logger.warning("Index not built, cannot query | 索引未构建，无法查询")
            return []
        
        query = self._preprocess_features(query_features.reshape(1, -1), fit=False)
        
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        if self.metric == DistanceMetric.COSINE:
            norm = np.linalg.norm(query)
            query_normalized = query / (norm + 1e-8)
            indices = self.kdtree.query_ball_point(query_normalized, max_distance)
        else:
            indices = self.kdtree.query_ball_point(query, max_distance)
        
        if not indices or not indices[0]:
            return []
        
        results = []
        for idx in indices[0]:
            # Compute actual distance | 计算实际距离
            if self.metric == DistanceMetric.COSINE:
                ref = self.features[idx]
                dist = self._cosine_distance(query[0], ref)
            else:
                ref = self.features[idx]
                dist = np.linalg.norm(query[0] - ref)
            
            score = self.distance_to_score(dist)
            results.append(MatchResult(
                index=int(idx),
                score=float(score),
                distance=float(dist),
                metadata=self.metadata_list[idx]
            ))
        
        results.sort(key=lambda x: x.distance)
        return results
    
    def batch_find_similar(self,
                          query_features_batch: np.ndarray,
                          k: int = 5,
                          threshold: Optional[float] = None) -> List[List[MatchResult]]:
        """
        Batch find similar samples | 批量查找相似样本
        
        Args | 参数:
            query_features_batch: Batch of query feature vectors | 查询特征向量批次
            k: Number of results per query | 每个查询的结果数量
            threshold: Similarity threshold | 相似度阈值
            
        Returns | 返回:
            List[List[MatchResult]]: List of results for each query | 每个查询的结果列表
        """
        results = []
        for query in query_features_batch:
            matches = self.find_similar(query, k, threshold)
            results.append(matches)
        return results
    
    def _normalize_distances(self, distances, indices) -> List[float]:
        """
        Normalize distance values | 归一化距离值
        
        Handles scalar, 1D, and 2D distance arrays.
        
        处理标量、一维和二维距离数组。
        """
        if isinstance(distances, np.ndarray):
            if distances.ndim == 0:  # Scalar | 标量
                return [float(distances)]
            elif distances.ndim == 1:  # 1D array | 一维数组
                return [float(d) for d in distances]
            else:  # 2D array | 二维数组
                return [float(d[0]) for d in distances]
        elif isinstance(distances, (list, tuple)):
            result = []
            for d in distances:
                if isinstance(d, (list, tuple, np.ndarray)):
                    result.append(float(d[0]) if d else 0.0)
                else:
                    result.append(float(d))
            return result
        return [float(distances)]
    
    # ============================================================
    # Utility Methods | 工具方法
    # ============================================================
    
    def compute_pairwise_similarity(self,
                                    features1: np.ndarray,
                                    features2: np.ndarray) -> np.ndarray:
        """
        Compute pairwise similarity between two sets of features | 计算两组特征之间的成对相似度
        
        Args | 参数:
            features1: First feature set | 第一组特征
            features2: Second feature set | 第二组特征
            
        Returns | 返回:
            np.ndarray: Similarity matrix | 相似度矩阵
        """
        f1 = self._preprocess_features(features1, fit=False)
        f2 = self._preprocess_features(features2, fit=False)
        
        if self.metric == DistanceMetric.COSINE:
            norms1 = np.linalg.norm(f1, axis=1, keepdims=True)
            norms2 = np.linalg.norm(f2, axis=1, keepdims=True)
            f1_norm = f1 / (norms1 + 1e-8)
            f2_norm = f2 / (norms2 + 1e-8)
            
            sim_matrix = np.dot(f1_norm, f2_norm.T)
            sim_matrix = np.clip(sim_matrix, -1.0, 1.0)
            sim_matrix = (sim_matrix + 1.0) / 2.0  # Map to [0,1] | 映射到[0,1]
        else:
            dist_matrix = cdist(f1, f2, metric=self.metric.value)
            sim_matrix = 1.0 / (1.0 + dist_matrix)
        
        return sim_matrix
    
    def get_neighborhood_stats(self, query_features: np.ndarray, radius: float) -> Dict:
        """
        Get neighborhood statistics for a query point | 获取查询点邻域的统计信息
        
        Args | 参数:
            query_features: Query feature vector | 查询特征向量
            radius: Search radius | 搜索半径
            
        Returns | 返回:
            Dict: Statistics dictionary | 统计信息
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
    
    # ============================================================
    # Persistence | 持久化
    # ============================================================
    
    def save(self, filename: str) -> bool:
        """
        Save matcher state to file | 保存匹配器状态到文件
        
        Args | 参数:
            filename: Output filename | 输出文件名
            
        Returns | 返回:
            bool: True if successful | 是否成功
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
            
            self.logger.info(f"Matcher saved to {filename} | 匹配器已保存")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save: {e} | 保存失败")
            return False
    
    def load(self, filename: str) -> bool:
        """
        Load matcher state from file | 从文件加载匹配器状态
        
        Args | 参数:
            filename: Input filename | 文件名
            
        Returns | 返回:
            bool: True if successful | 是否成功
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
            
            # Rebuild index | 重建索引
            if self.features is not None:
                self.build_index(self.features, self.metadata_list)
            
            self.logger.info(f"Matcher loaded from {filename} | 匹配器已加载")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load: {e} | 加载失败")
            return False
    
    def clear(self) -> None:
        """Clear all data | 清空所有数据"""
        self.features = None
        self.metadata_list = []
        self.kdtree = None
        self.pca = None
        self.scaler = None
        self.n_samples = 0
        self.feature_dim = 0
        self.logger.info("Matcher cleared | 匹配器已清空")
    
    def __len__(self) -> int:
        """Return number of samples | 返回样本数量"""
        return self.n_samples