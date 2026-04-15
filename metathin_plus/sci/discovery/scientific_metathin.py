# metathin_sci/discovery/scientific_metathin.py
"""
Scientific Metathin Agent - Main Discovery Interface | 科学发现智能体 - 主发现接口
===================================================================================

Integrates all scientific discovery components into a unified Metathin-compatible agent.
This agent follows the quintuple (P, B, S, D, Ψ) architecture.

将所有科学发现组件整合到统一的兼容 Metathin 的智能体中。
该智能体遵循五元组 (P, B, S, D, Ψ) 架构。

Components | 组件:
    - P: ChaosPatternSpace (or custom pattern space) | 混沌模式空间
    - B: Various prediction behaviors | 各种预测行为
    - S: ChaosSelector (fitness based on error) | 混沌选择器
    - D: Decision strategies (min-error, weighted, adaptive) | 决策策略
    - Ψ: Learning mechanisms (error-based, reward-based) | 学习机制
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Type
import time
import logging
from dataclasses import dataclass

# Metathin core imports | Metathin 核心导入
from metathin import Metathin, MetathinBuilder
from metathin.core.types import FeatureVector
from metathin.core.exceptions import MetathinError, NoBehaviorError

# Local imports | 本地导入
from ..core.feature_extractor import FeatureExtractor
from ..core.similarity_matcher import SimilarityMatcher
from ..memory.function_memory import FunctionMemory, FunctionMemoryBank
from .adaptive_extrapolator import AdaptiveExtrapolator, SymbolicForm
from .report_generator import DiscoveryReport, DiscoveryPhase


# ============================================================
# Pattern Space (P) for Scientific Discovery | 科学发现模式空间
# ============================================================

class ScientificPatternSpace:
    """
    Pattern Space for scientific discovery.
    
    科学发现的模式空间。
    
    Extracts features from time series data for function identification.
    
    从时间序列数据中提取特征用于函数识别。
    """
    
    def __init__(self,
                 feature_extractor: Optional[FeatureExtractor] = None,
                 feature_dim: int = 33,
                 normalize: bool = True):
        """
        Initialize scientific pattern space.
        
        初始化科学发现模式空间。
        
        Args | 参数:
            feature_extractor: Feature extractor instance | 特征提取器实例
            feature_dim: Expected feature dimension | 期望的特征维度
            normalize: Whether to normalize features | 是否归一化特征
        """
        self.feature_extractor = feature_extractor or FeatureExtractor(normalize=normalize)
        self.feature_dim = feature_dim
        self.normalize = normalize
        self._logger = logging.getLogger("metathin_sci.discovery.ScientificPatternSpace")
    
    def extract(self, raw_input: np.ndarray) -> FeatureVector:
        """
        Extract features from time series data.
        
        从时间序列数据中提取特征。
        
        Args | 参数:
            raw_input: Time series data | 时间序列数据
            
        Returns | 返回:
            FeatureVector: Feature vector | 特征向量
        """
        features = self.feature_extractor.extract(raw_input)
        
        # Ensure consistent dimension | 确保维度一致
        if len(features) != self.feature_dim:
            if len(features) < self.feature_dim:
                features = np.pad(features, 
                                  (0, self.feature_dim - len(features)),
                                  mode='constant', constant_values=0)
            else:
                features = features[:self.feature_dim]
        
        return features.astype(np.float64)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names | 获取特征名称"""
        return self.feature_extractor.get_feature_names()
    
    def get_feature_dimension(self) -> int:
        """Get feature dimension | 获取特征维度"""
        return self.feature_dim


# ============================================================
# Behavior (B) Placeholders | 行为占位符
# ============================================================

# Note: The actual prediction behaviors are implemented in the chaos module
# and can be reused here. For standalone use, we provide simple wrappers.
# 注意：实际的预测行为实现在混沌模块中，可在此复用。
# 为独立使用，我们提供简单的包装器。

class BaseScientificBehavior:
    """
    Base class for scientific discovery behaviors.
    
    科学发现行为的基类。
    """
    
    def __init__(self, name: str):
        self._name = name
        self._execution_count = 0
        self._logger = logging.getLogger(f"metathin_sci.behavior.{name}")
    
    @property
    def name(self) -> str:
        return self._name
    
    def execute(self, features: FeatureVector, **kwargs) -> Any:
        """Execute behavior | 执行行为"""
        self._execution_count += 1
        raise NotImplementedError
    
    def can_execute(self, features: FeatureVector) -> bool:
        """Check if behavior can execute | 检查行为是否可执行"""
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get behavior statistics | 获取行为统计"""
        return {
            'name': self.name,
            'execution_count': self._execution_count,
        }


# ============================================================
# Scientific Metathin Agent | 科学发现智能体
# ============================================================

class ScientificMetathin:
    """
    Scientific Metathin Agent - Main discovery interface.
    
    科学发现智能体 - 主发现接口。
    
    This agent integrates all five Metathin components (P, B, S, D, Ψ)
    for scientific discovery from time series data.
    
    这个智能体整合了所有五个 Metathin 组件 (P, B, S, D, Ψ)，
    用于从时间序列数据中进行科学发现。
    
    Example | 示例:
        >>> agent = ScientificMetathin()
        >>> 
        >>> # Analyze data and discover patterns | 分析数据并发现规律
        >>> report = agent.discover(y_data)
        >>> 
        >>> # Print discovered formulas | 打印发现的公式
        >>> for phase in report.phases:
        ...     print(f"Formula: {phase.formula}")
        ...     print(f"Range: {phase.range}")
    """
    
    def __init__(self,
                 name: str = "ScientificMetathin",
                 memory_bank: Optional[FunctionMemoryBank] = None,
                 pretrained: bool = True,
                 pretrained_libraries: Optional[List[str]] = None,
                 feature_dim: int = 33,
                 N: int = 50,
                 delta: float = 0.1,
                 enable_learning: bool = True,
                 memory_path: Optional[str] = None,
                 verbose: bool = False):
        """
        Initialize Scientific Metathin agent.
        
        初始化科学发现智能体。
        
        Args | 参数:
            name: Agent name | 智能体名称
            memory_bank: Function memory bank | 函数记忆库
            pretrained: Whether to load pretrained memory | 是否加载预训练记忆库
            pretrained_libraries: Pretrained libraries to load | 要加载的预训练库
            feature_dim: Feature dimension | 特征维度
            N: Extrapolation window size | 外推窗口大小
            delta: Error threshold for restart | 重启误差阈值
            enable_learning: Whether to enable learning | 是否启用学习
            memory_path: Memory storage path | 记忆存储路径
            verbose: Enable verbose logging | 启用详细日志
        """
        self.name = name
        self.enable_learning = enable_learning
        self.delta = delta
        self.N = N
        self.feature_dim = feature_dim
        self.verbose = verbose
        
        self._logger = logging.getLogger(f"metathin_sci.{name}")
        if verbose:
            self._logger.setLevel(logging.DEBUG)
        
        # ============================================================
        # 1. Pattern Space (P) | 感知层
        # ============================================================
        self.pattern_space = ScientificPatternSpace(feature_dim=feature_dim)
        
        # ============================================================
        # 2. Memory Bank (Knowledge Base) | 记忆库（知识库）
        # ============================================================
        if memory_bank:
            self.memory_bank = memory_bank
        else:
            self.memory_bank = FunctionMemoryBank(
                memory_backend='json',
                memory_path=memory_path or f"{name}_memory.json"
            )
            
            if pretrained:
                from ..memory.pretrained import create_pretrained_bank
                libraries = pretrained_libraries or ['basic', 'physics', 'chemistry']
                pretrained_bank = create_pretrained_bank(
                    library_keys=libraries,
                    limits={'basic': 15, 'physics': 10, 'chemistry': 8},
                    expected_dim=feature_dim
                )
                for func in pretrained_bank:
                    self.memory_bank.add(func)
                self._logger.info(f"Loaded pretrained memory: {libraries} | 加载预训练记忆库")
        
        # ============================================================
        # 3. Feature Extractor & Similarity Matcher | 特征提取器与相似度匹配器
        # ============================================================
        self.feature_extractor = FeatureExtractor(normalize=True)
        self.similarity_matcher = SimilarityMatcher(threshold=0.7)
        self._update_matcher_index()
        
        # ============================================================
        # 4. Adaptive Extrapolator (Core Discovery Algorithm) | 自适应外推器（核心发现算法）
        # ============================================================
        self.extrapolator = AdaptiveExtrapolator(N=N, delta=delta)
        
        # ============================================================
        # 5. State Management | 状态管理
        # ============================================================
        self._history: List[float] = []
        self._timestamps: List[float] = []
        self.current_report: Optional[DiscoveryReport] = None
        self.discovery_history: List[DiscoveryReport] = []
        
        # Statistics | 统计信息
        self.stats = {
            'total_discoveries': 0,
            'memory_hits': 0,
            'memory_misses': 0,
            'learning_events': 0,
            'start_time': time.time()
        }
        
        self._logger.info(f"✅ ScientificMetathin '{name}' initialized | 初始化完成")
        self._logger.info(f"   Memory size: {len(self.memory_bank)} | 记忆库大小")
        self._logger.info(f"   Feature dimension: {feature_dim} | 特征维度")
        self._logger.info(f"   Extrapolation: N={N}, delta={delta} | 外推参数")
    
    def _update_matcher_index(self):
        """Update similarity matcher index | 更新相似度匹配器索引"""
        if len(self.memory_bank) == 0:
            return
        
        features = []
        metadata = []
        
        for func in self.memory_bank:
            # Ensure consistent feature dimension | 确保特征维度一致
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
        
        self._logger.debug(f"Matcher index updated: {len(features)} functions | 匹配器索引已更新")
    
    def _get_memory_assistance(self, data_segment: np.ndarray) -> Optional[Dict]:
        """
        Get assistance from memory bank.
        
        从记忆库获取辅助信息。
        
        Args | 参数:
            data_segment: Data segment to analyze | 要分析的数据段
            
        Returns | 返回:
            Optional[Dict]: Assistance information | 辅助信息
        """
        if len(self.memory_bank) == 0:
            return None
        
        features = self.feature_extractor.extract(data_segment)
        
        # Ensure dimension consistency | 确保维度一致
        if len(features) != self.feature_dim:
            if len(features) < self.feature_dim:
                features = np.pad(features, 
                                  (0, self.feature_dim - len(features)),
                                  mode='constant', constant_values=0)
            else:
                features = features[:self.feature_dim]
        
        matches = self.similarity_matcher.find_similar(features, k=3, threshold=0.7)
        
        if matches:
            self.stats['memory_hits'] += 1
            
            assistance = {
                'similar_functions': [],
                'best_match': None,
                'suggested_forms': []
            }
            
            for match in matches:
                func_info = {
                    'expression': match.metadata.get('expression', 'unknown'),
                    'similarity': match.score,
                    'accuracy': match.metadata.get('accuracy', 0.0)
                }
                assistance['similar_functions'].append(func_info)
                
                if match.score > 0.8 and not assistance['best_match']:
                    assistance['best_match'] = func_info
            
            self._logger.debug(f"Memory assistance: found {len(matches)} similar functions | 找到 {len(matches)} 个相似函数")
            return assistance
        else:
            self.stats['memory_misses'] += 1
            return None
    
    def _learn_from_discovery(self, form: SymbolicForm, data: np.ndarray):
        """
        Learn from discovered pattern.
        
        从发现的规律中学习。
        
        Args | 参数:
            form: Discovered symbolic form | 发现的符号形式
            data: Data segment | 数据段
        """
        if not self.enable_learning:
            return
        
        try:
            features = self.feature_extractor.extract(data)
            
            # Ensure dimension consistency | 确保维度一致
            if len(features) != self.feature_dim:
                if len(features) < self.feature_dim:
                    features = np.pad(features, 
                                      (0, self.feature_dim - len(features)),
                                      mode='constant', constant_values=0)
                else:
                    features = features[:self.feature_dim]
            
            func_memory = FunctionMemory(
                expression=form.expression,
                parameters=form.params,
                feature_vector=features,
                accuracy=1.0 - form.error / self.delta,
                tags=['discovered'],
                source='scientific_metathin'
            )
            
            self.memory_bank.add(func_memory)
            self.stats['learning_events'] += 1
            
            self._logger.info(f"Learned new function: {form.expression}, error={form.error:.6f} | 学习新函数")
            
            self._update_matcher_index()
            
        except Exception as e:
            self._logger.error(f"Learning failed: {e} | 学习失败")
    
    def discover(self,
                 y_data: np.ndarray,
                 x_data: Optional[np.ndarray] = None,
                 use_memory: bool = True,
                 learn: bool = True) -> DiscoveryReport:
        """
        Discover patterns from data.
        
        从数据中发现规律。
        
        Args | 参数:
            y_data: Time series data | 时间序列数据
            x_data: Optional domain values | 可选的域值
            use_memory: Whether to use memory assistance | 是否使用记忆辅助
            learn: Whether to learn from discoveries | 是否从发现中学习
            
        Returns | 返回:
            DiscoveryReport: Discovery report | 发现报告
        """
        if x_data is None:
            x_data = np.arange(len(y_data))
        
        if len(x_data) != len(y_data):
            raise ValueError("x_data and y_data must have same length | x_data 和 y_data 长度必须相同")
        
        self._logger.info(f"Starting discovery, data length: {len(y_data)} | 开始发现过程")
        
        # Create report | 创建报告
        report = DiscoveryReport(
            title=f"Scientific Discovery Report - {self.name}",
            data_source="Real-time discovery | 实时发现"
        )
        
        # Reset extrapolator | 重置外推器
        self.extrapolator.reset()
        
        # Process data sequentially | 顺序处理数据
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            y_pred = self.extrapolator.predict(x)
            self.extrapolator.update(x, y)
            
            # Periodically check memory | 定期检查记忆
            if use_memory and i % self.N == 0 and i > self.N:
                segment = y_data[max(0, i-self.N):i]
                assistance = self._get_memory_assistance(segment)
                if assistance and assistance.get('best_match'):
                    self._logger.info(f"Memory suggestion: {assistance['best_match']['expression']} | 记忆建议")
        
        # Build report from discovered forms | 从发现的形式构建报告
        for form in self.extrapolator.get_history():
            phase = DiscoveryPhase(
                formula=form.expression,
                params=form.params,
                range=form.valid_range,
                error=form.error,
                confidence=1.0 - form.error / self.delta,
                description=f"Discovered symbolic form | 发现的符号形式"
            )
            report.add_phase(phase)
            
            # Learn from discovery | 从发现中学习
            if learn:
                start_idx = max(0, int(form.valid_range[0]))
                end_idx = min(len(y_data), int(form.valid_range[1]) + 1)
                segment = y_data[start_idx:end_idx]
                self._learn_from_discovery(form, segment)
        
        self.stats['total_discoveries'] += 1
        self.current_report = report
        self.discovery_history.append(report)
        
        self._logger.info(f"Discovery complete: found {len(report.phases)} patterns | 发现完成")
        
        return report
    
    def analyze(self,
                data: np.ndarray,
                x_data: Optional[np.ndarray] = None,
                save_report: bool = True) -> Dict[str, Any]:
        """
        Analyze data and return structured results.
        
        分析数据并返回结构化结果。
        
        Args | 参数:
            data: Time series data | 时间序列数据
            x_data: Optional domain values | 可选的域值
            save_report: Whether to save report to file | 是否保存报告到文件
            
        Returns | 返回:
            Dict: Analysis results | 分析结果
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
            import time
            filename = f"discovery_report_{int(time.time())}.json"
            report.save(filename)
            result['report_file'] = filename
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        获取智能体统计信息。
        
        Returns | 返回:
            Dict: Statistics dictionary | 统计字典
        """
        runtime = time.time() - self.stats['start_time']
        
        extrapolator_stats = self.extrapolator.get_stats()
        memory_stats = self.memory_bank.get_statistics() if hasattr(self.memory_bank, 'get_statistics') else {}
        
        total_memory_queries = self.stats['memory_hits'] + self.stats['memory_misses']
        memory_hit_rate = self.stats['memory_hits'] / max(1, total_memory_queries)
        
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
        
        保存记忆库到文件。
        
        Args | 参数:
            filename: Output filename | 输出文件名
            
        Returns | 返回:
            bool: True if successful | 成功返回True
        """
        if filename is None:
            filename = f"{self.name}_memory.json"
        return self.memory_bank.save(filename)
    
    def load_memory(self, filename: str) -> bool:
        """
        Load memory bank from file.
        
        从文件加载记忆库。
        
        Args | 参数:
            filename: Input filename | 输入文件名
            
        Returns | 返回:
            bool: True if successful | 成功返回True
        """
        success = self.memory_bank.load(filename)
        if success:
            self._update_matcher_index()
            self._logger.info(f"Loaded memory from: {filename} | 加载记忆库")
        return success
    
    def clear_memory(self) -> None:
        """Clear memory bank | 清空记忆库"""
        self.memory_bank.clear()
        self._update_matcher_index()
        self._logger.info("Memory cleared | 记忆库已清空")
    
    def get_report(self) -> Optional[DiscoveryReport]:
        """Get current discovery report | 获取当前发现报告"""
        return self.current_report
    
    def get_history(self) -> List[DiscoveryReport]:
        """Get discovery history | 获取发现历史"""
        return self.discovery_history.copy()
    
    def reset(self) -> None:
        """Reset agent state | 重置智能体状态"""
        self.extrapolator.reset()
        self.current_report = None
        self._history.clear()
        self._timestamps.clear()
        self.stats = {
            'total_discoveries': 0,
            'memory_hits': 0,
            'memory_misses': 0,
            'learning_events': 0,
            'start_time': time.time()
        }
        self._logger.info("Agent reset | 智能体已重置")
    
    def __repr__(self) -> str:
        return (f"ScientificMetathin(name='{self.name}', "
                f"discoveries={self.stats['total_discoveries']}, "
                f"memory={len(self.memory_bank)})")