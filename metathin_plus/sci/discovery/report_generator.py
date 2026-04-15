# metathin_sci/discovery/report_generator.py
"""
Discovery Report Generator | 科学发现报告生成器
================================================

Organizes discovered patterns into readable, shareable scientific reports.
Supports multiple output formats: text, JSON, LaTeX, HTML.

将发现的规律整理成可读、可分享的科学报告。
支持多种输出格式：文本、JSON、LaTeX、HTML。
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import logging


@dataclass
class DiscoveryPhase:
    """
    Discovery phase data class | 发现阶段数据类
    
    Records a discovered pattern within a continuous interval.
    
    记录一个连续区间内发现的规律。
    
    Attributes | 属性:
        formula: Discovered formula (string) | 发现的公式（字符串形式）
        params: Parameter values | 参数值
        range: Valid range (x_min, x_max) | 有效范围
        error: Fitting error | 拟合误差
        confidence: Confidence level (0-1) | 置信度
        description: Description text | 描述信息
        metadata: Additional metadata | 额外元数据
    """
    formula: str
    params: Dict[str, float]
    range: Tuple[float, float]
    error: float
    confidence: float
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation | 初始化后验证"""
        if self.error < 0:
            self.error = abs(self.error)
        self.confidence = max(0.0, min(1.0, self.confidence))
        if self.range[0] > self.range[1]:
            self.range = (self.range[1], self.range[0])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary | 转换为字典"""
        return {
            'formula': self.formula,
            'params': self.params.copy(),
            'range': list(self.range),
            'error': float(self.error),
            'confidence': float(self.confidence),
            'description': self.description,
            'metadata': self.metadata.copy()
        }
    
    def to_latex(self) -> str:
        """Convert to LaTeX format | 转换为LaTeX格式"""
        params_str = ", ".join([f"{k}={v:.3f}" for k, v in self.params.items()])
        range_str = f"$x \\in [{self.range[0]:.2f}, {self.range[1]:.2f}]$"
        
        # Simple LaTeX conversion | 简单的LaTeX转换
        formula_latex = self.formula.replace('*', '')
        formula_latex = formula_latex.replace('sin', '\\sin')
        formula_latex = formula_latex.replace('cos', '\\cos')
        formula_latex = formula_latex.replace('exp', '\\exp')
        formula_latex = formula_latex.replace('log', '\\log')
        
        return (f"\\item \\textbf{{{formula_latex}}} \\\\\n"
                f"      Parameters | 参数: {params_str} \\\\\n"
                f"      Range | 范围: {range_str} \\\\\n"
                f"      Error | 误差: $\\epsilon = {self.error:.6f}$ \\\\\n"
                f"      Confidence | 置信度: ${self.confidence:.1%}$")
    
    def __repr__(self) -> str:
        return f"DiscoveryPhase(formula='{self.formula}', range={self.range}, error={self.error:.4f})"


@dataclass
class DiscoveryReport:
    """
    Discovery report data class | 科学发现报告数据类
    
    Contains all information from a complete discovery process.
    
    包含一次完整发现过程的所有信息。
    
    Attributes | 属性:
        title: Report title | 报告标题
        data_source: Data source description | 数据来源描述
        phases: Discovered phases | 发现的各个阶段
        created_at: Creation timestamp | 创建时间
        metadata: Additional metadata | 元数据
    """
    title: str = "Scientific Discovery Report | 科学发现报告"
    data_source: str = "Unknown | 未知"
    phases: List[DiscoveryPhase] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self._logger = logging.getLogger("metathin_sci.discovery.DiscoveryReport")
    
    def add_phase(self, phase: DiscoveryPhase):
        """Add a discovery phase | 添加发现阶段"""
        if not isinstance(phase, DiscoveryPhase):
            raise TypeError(f"phase must be DiscoveryPhase, got {type(phase)}")
        self.phases.append(phase)
    
    def add_phases(self, phases: List[DiscoveryPhase]):
        """Add multiple phases | 批量添加发现阶段"""
        for phase in phases:
            self.add_phase(phase)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics | 获取摘要信息
        
        Returns | 返回:
            Dict: Summary statistics | 摘要统计
        """
        if not self.phases:
            return {
                'n_phases': 0,
                'avg_error': 0.0,
                'min_error': 0.0,
                'max_error': 0.0,
                'avg_confidence': 0.0,
                'total_range': 0.0,
                'coverage': 0.0
            }
        
        errors = [p.error for p in self.phases]
        confidences = [p.confidence for p in self.phases]
        ranges = [p.range[1] - p.range[0] for p in self.phases]
        
        if len(self.phases) > 1:
            total_range = self.phases[-1].range[1] - self.phases[0].range[0]
        else:
            total_range = ranges[0] if ranges else 0.0
        
        return {
            'n_phases': len(self.phases),
            'avg_error': float(np.mean(errors)) if errors else 0.0,
            'min_error': float(np.min(errors)) if errors else 0.0,
            'max_error': float(np.max(errors)) if errors else 0.0,
            'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'total_range': float(total_range),
            'coverage': float(sum(ranges)) / total_range if total_range > 0 else 1.0
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary | 转换为字典"""
        return {
            'title': self.title,
            'data_source': self.data_source,
            'created_at': self.created_at.isoformat(),
            'phases': [p.to_dict() for p in self.phases],
            'summary': self.get_summary(),
            'metadata': self.metadata.copy()
        }
    
    def to_text(self) -> str:
        """Convert to plain text | 转换为纯文本"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"  {self.title}")
        lines.append("=" * 60)
        lines.append(f"Data source | 数据来源: {self.data_source}")
        lines.append(f"Created at | 创建时间: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        summary = self.get_summary()
        lines.append(f"📊 Discovery Summary | 发现摘要")
        lines.append(f"   Number of phases | 阶段数量: {summary['n_phases']}")
        if summary['n_phases'] > 0:
            lines.append(f"   Average error | 平均误差: {summary['avg_error']:.6f}")
            lines.append(f"   Min error | 最小误差: {summary['min_error']:.6f}")
            lines.append(f"   Max error | 最大误差: {summary['max_error']:.6f}")
            lines.append(f"   Average confidence | 平均置信度: {summary['avg_confidence']:.1%}")
            lines.append(f"   Coverage | 覆盖范围: {summary['coverage']:.1%}")
        lines.append("")
        
        if self.phases:
            lines.append("📈 Discovery Phases | 发现阶段")
            for i, phase in enumerate(self.phases):
                lines.append(f"\nPhase {i+1} | 阶段 {i+1}:")
                lines.append(f"   Formula | 公式: {phase.formula}")
                lines.append(f"   Parameters | 参数: {phase.params}")
                lines.append(f"   Range | 范围: [{phase.range[0]:.2f}, {phase.range[1]:.2f}]")
                lines.append(f"   Error | 误差: {phase.error:.6f}")
                lines.append(f"   Confidence | 置信度: {phase.confidence:.1%}")
                if phase.description:
                    lines.append(f"   Description | 描述: {phase.description}")
        else:
            lines.append("⚠️ No patterns discovered | 未发现任何规律")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def to_html(self) -> str:
        """Convert to HTML format | 转换为HTML格式"""
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("<meta charset='utf-8'>")
        html.append(f"<title>{self.title}</title>")
        html.append("""
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            .header { background: #f0f0f0; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
            .phase { background: #f9f9f9; margin: 10px 0; padding: 15px; border-left: 5px solid #4CAF50; border-radius: 0 5px 5px 0; }
            .formula { font-size: 1.2em; font-weight: bold; color: #2196F3; margin-bottom: 10px; }
            .stats { color: #666; font-size: 0.9em; }
            .warning { color: #f44336; font-weight: bold; }
            .summary { background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; }
        </style>
        """)
        html.append("</head>")
        html.append("<body>")
        
        # Header | 头部
        html.append(f"<div class='header'>")
        html.append(f"<h1>{self.title}</h1>")
        html.append(f"<p><strong>Data source | 数据来源:</strong> {self.data_source}</p>")
        html.append(f"<p><strong>Created at | 创建时间:</strong> {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>")
        html.append("</div>")
        
        # Summary | 摘要
        summary = self.get_summary()
        html.append("<div class='summary'>")
        html.append("<h2>📊 Discovery Summary | 发现摘要</h2>")
        html.append(f"<p><strong>Number of phases | 阶段数量:</strong> {summary['n_phases']}</p>")
        if summary['n_phases'] > 0:
            html.append(f"<p><strong>Average error | 平均误差:</strong> {summary['avg_error']:.6f}</p>")
            html.append(f"<p><strong>Average confidence | 平均置信度:</strong> {summary['avg_confidence']:.1%}</p>")
            html.append(f"<p><strong>Coverage | 覆盖范围:</strong> {summary['coverage']:.1%}</p>")
        html.append("</div>")
        
        # Phases | 阶段
        if self.phases:
            html.append("<h2>📈 Discovered Patterns | 发现的规律</h2>")
            for i, phase in enumerate(self.phases):
                html.append(f"<div class='phase'>")
                html.append(f"<div class='formula'>{phase.formula}</div>")
                html.append(f"<div class='stats'>")
                html.append(f"<strong>Parameters | 参数:</strong> {phase.params}<br>")
                html.append(f"<strong>Range | 范围:</strong> [{phase.range[0]:.2f}, {phase.range[1]:.2f}]<br>")
                html.append(f"<strong>Error | 误差:</strong> {phase.error:.6f} | ")
                html.append(f"<strong>Confidence | 置信度:</strong> {phase.confidence:.1%}")
                if phase.description:
                    html.append(f"<br><strong>Description | 描述:</strong> {phase.description}")
                html.append("</div>")
                html.append("</div>")
        else:
            html.append("<p class='warning'>⚠️ No patterns discovered | 未发现任何规律</p>")
        
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)
    
    def save(self, filename: str, format: str = 'json') -> bool:
        """
        Save report to file | 保存报告到文件
        
        Args | 参数:
            filename: Output filename | 输出文件名
            format: Output format ('json', 'txt', 'html') | 输出格式
            
        Returns | 返回:
            bool: True if successful | 成功返回True
        """
        try:
            filepath = Path(filename)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'json':
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            
            elif format == 'txt':
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.to_text())
            
            elif format == 'html':
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.to_html())
            
            else:
                raise ValueError(f"Unsupported format: {format} | 不支持的格式")
            
            self._logger.info(f"Report saved to {filename} | 报告已保存")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to save report: {e} | 保存失败")
            return False
    
    def print(self):
        """Print report to console | 打印报告到控制台"""
        print(self.to_text())
    
    def __len__(self) -> int:
        """Return number of phases | 返回阶段数量"""
        return len(self.phases)
    
    def __getitem__(self, index: int) -> DiscoveryPhase:
        """Support indexing | 支持索引访问"""
        return self.phases[index]
    
    def __iter__(self):
        """Support iteration | 支持迭代"""
        return iter(self.phases)