"""
Scientific Discovery Report Generator
======================================================================

Organizes discovered patterns into readable, shareable scientific reports.
Supports multiple output formats: text, JSON, LaTeX, HTML, PDF.

Design Philosophy:
    - Readability: Human-friendly report formats
    - Completeness: Contains all discovered patterns and metadata
    - Reproducible: Records complete discovery process
    - Shareable: Supports multiple export formats
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import logging


@dataclass
class DiscoveryPhase:
    """
    Discovery phase data class.
    
    Records a pattern discovered within a continuous interval.
    
    Attributes:
        formula: Discovered formula (string representation)
        params: Parameter values
        range: Valid range (x_min, x_max)
        error: Fitting error
        confidence: Confidence level (0-1)
        description: Description of the phase
        metadata: Additional metadata
    """
    formula: str
    params: Dict[str, float]
    range: Tuple[float, float]
    error: float
    confidence: float
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation."""
        # Ensure error is non-negative
        if self.error < 0:
            self.error = abs(self.error)
        
        # Ensure confidence is in [0,1] range
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        # Ensure range is properly formatted
        if self.range[0] > self.range[1]:
            self.range = (self.range[1], self.range[0])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
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
        """
        Convert to LaTeX format.
        
        Returns:
            str: LaTeX representation of the discovery phase
        """
        # Build parameters string
        params_str = ", ".join([f"{k}={v:.3f}" for k, v in self.params.items()])
        
        # Range string
        range_str = f"$x \\in [{self.range[0]:.2f}, {self.range[1]:.2f}]$"
        
        # Formula string - simple LaTeX conversion
        formula_latex = self.formula.replace('*', '')
        formula_latex = formula_latex.replace('sin', '\\sin')
        formula_latex = formula_latex.replace('cos', '\\cos')
        formula_latex = formula_latex.replace('exp', '\\exp')
        formula_latex = formula_latex.replace('log', '\\log')
        
        return (f"\\item \\textbf{{{formula_latex}}} \\\\\n"
                f"     Parameters: {params_str} \\\\\n"
                f"     Range: {range_str} \\\\\n"
                f"     Error: $\\epsilon = {self.error:.6f}$ \\\\\n"
                f"     Confidence: ${self.confidence:.1%}$")
    
    def __repr__(self) -> str:
        return f"DiscoveryPhase(formula='{self.formula}', range={self.range}, error={self.error:.4f})"


@dataclass
class DiscoveryReport:
    """
    Scientific Discovery Report Data Class.
    
    Contains all information from a complete discovery process.
    
    Attributes:
        title: Report title
        data_source: Description of data source
        phases: List of discovered phases
        created_at: Creation timestamp
        metadata: Additional metadata
    """
    title: str = "Scientific Discovery Report"
    data_source: str = "Unknown"
    phases: List[DiscoveryPhase] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization setup."""
        self._logger = logging.getLogger("metathin_sci.discovery.DiscoveryReport")
    
    def add_phase(self, phase: DiscoveryPhase):
        """Add a discovery phase."""
        if not isinstance(phase, DiscoveryPhase):
            raise TypeError(f"phase must be of type DiscoveryPhase, got {type(phase)}")
        self.phases.append(phase)
    
    def add_phases(self, phases: List[DiscoveryPhase]):
        """Add multiple discovery phases."""
        for phase in phases:
            self.add_phase(phase)
    
    def remove_phase(self, index: int) -> Optional[DiscoveryPhase]:
        """
        Remove phase at specified index.
        
        Args:
            index: Phase index
            
        Returns:
            Optional[DiscoveryPhase]: Removed phase, None if index invalid
        """
        if 0 <= index < len(self.phases):
            return self.phases.pop(index)
        return None
    
    def get_phase(self, index: int) -> Optional[DiscoveryPhase]:
        """Get phase at specified index."""
        if 0 <= index < len(self.phases):
            return self.phases[index]
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary information.
        
        Returns:
            Dict: Statistics including number of phases, average error, etc.
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
        
        # Calculate total coverage range
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
        """Convert to dictionary for serialization."""
        return {
            'title': self.title,
            'data_source': self.data_source,
            'created_at': self.created_at.isoformat(),
            'phases': [p.to_dict() for p in self.phases],
            'summary': self.get_summary(),
            'metadata': self.metadata.copy()
        }
    
    def to_text(self) -> str:
        """Convert to plain text format."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"  {self.title}")
        lines.append("=" * 60)
        lines.append(f"Data Source: {self.data_source}")
        lines.append(f"Created: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        summary = self.get_summary()
        lines.append(f"📊 Discovery Summary")
        lines.append(f"   Number of phases: {summary['n_phases']}")
        if summary['n_phases'] > 0:
            lines.append(f"   Average error: {summary['avg_error']:.6f}")
            lines.append(f"   Minimum error: {summary['min_error']:.6f}")
            lines.append(f"   Maximum error: {summary['max_error']:.6f}")
            lines.append(f"   Average confidence: {summary['avg_confidence']:.1%}")
            lines.append(f"   Coverage: {summary['coverage']:.1%}")
        lines.append("")
        
        if self.phases:
            lines.append("📈 Discovered Phases")
            for i, phase in enumerate(self.phases):
                lines.append(f"\nPhase {i+1}:")
                lines.append(f"  Formula: {phase.formula}")
                lines.append(f"  Parameters: {phase.params}")
                lines.append(f"  Range: [{phase.range[0]:.2f}, {phase.range[1]:.2f}]")
                lines.append(f"  Error: {phase.error:.6f}")
                lines.append(f"  Confidence: {phase.confidence:.1%}")
                if phase.description:
                    lines.append(f"  Description: {phase.description}")
        else:
            lines.append("⚠️ No patterns discovered")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def to_latex(self) -> str:
        """Convert to LaTeX format."""
        lines = []
        lines.append("\\documentclass{article}")
        lines.append("\\usepackage{amsmath}")
        lines.append("\\usepackage{geometry}")
        lines.append("\\geometry{a4paper, margin=1in}")
        lines.append("\\title{" + self.title + "}")
        lines.append("\\date{" + self.created_at.strftime('%Y-%m-%d') + "}")
        lines.append("\\begin{document}")
        lines.append("\\maketitle")
        
        lines.append("\\section*{Basic Information}")
        lines.append(f"Data Source: {self.data_source}\\\\")
        
        summary = self.get_summary()
        lines.append(f"Number of phases: {summary['n_phases']}\\\\")
        if summary['n_phases'] > 0:
            lines.append(f"Average error: ${summary['avg_error']:.6f}$\\\\")
            lines.append(f"Average confidence: ${summary['avg_confidence']:.1%}$")
        
        if self.phases:
            lines.append("\\section*{Discovered Patterns}")
            lines.append("\\begin{enumerate}")
            for phase in self.phases:
                lines.append(phase.to_latex())
            lines.append("\\end{enumerate}")
        else:
            lines.append("\\section*{Note}")
            lines.append("No patterns discovered.")
        
        lines.append("\\end{document}")
        
        return "\n".join(lines)
    
    def to_html(self) -> str:
        """Convert to HTML format."""
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("<meta charset='utf-8'>")
        html.append(f"<title>{self.title}</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }")
        html.append(".header { background: #f0f0f0; padding: 20px; border-radius: 10px; margin-bottom: 20px; }")
        html.append(".phase { background: #f9f9f9; margin: 10px 0; padding: 15px; border-left: 5px solid #4CAF50; border-radius: 0 5px 5px 0; }")
        html.append(".formula { font-size: 1.2em; font-weight: bold; color: #2196F3; margin-bottom: 10px; }")
        html.append(".stats { color: #666; font-size: 0.9em; }")
        html.append(".warning { color: #f44336; font-weight: bold; }")
        html.append(".summary { background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; }")
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        
        # Header
        html.append(f"<div class='header'>")
        html.append(f"<h1>{self.title}</h1>")
        html.append(f"<p><strong>Data Source:</strong> {self.data_source}</p>")
        html.append(f"<p><strong>Created:</strong> {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>")
        html.append("</div>")
        
        # Summary
        summary = self.get_summary()
        html.append("<div class='summary'>")
        html.append("<h2>📊 Discovery Summary</h2>")
        html.append(f"<p><strong>Number of phases:</strong> {summary['n_phases']}</p>")
        if summary['n_phases'] > 0:
            html.append(f"<p><strong>Average error:</strong> {summary['avg_error']:.6f}</p>")
            html.append(f"<p><strong>Average confidence:</strong> {summary['avg_confidence']:.1%}</p>")
            html.append(f"<p><strong>Coverage:</strong> {summary['coverage']:.1%}</p>")
        html.append("</div>")
        
        # Phases
        if self.phases:
            html.append("<h2>📈 Discovered Patterns</h2>")
            for i, phase in enumerate(self.phases):
                html.append(f"<div class='phase'>")
                html.append(f"<div class='formula'>{phase.formula}</div>")
                html.append(f"<div class='stats'>")
                html.append(f"<strong>Parameters:</strong> {phase.params}<br>")
                html.append(f"<strong>Range:</strong> [{phase.range[0]:.2f}, {phase.range[1]:.2f}]<br>")
                html.append(f"<strong>Error:</strong> {phase.error:.6f} | ")
                html.append(f"<strong>Confidence:</strong> {phase.confidence:.1%}")
                if phase.description:
                    html.append(f"<br><strong>Description:</strong> {phase.description}")
                html.append("</div>")
                html.append("</div>")
        else:
            html.append("<p class='warning'>⚠️ No patterns discovered</p>")
        
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)
    
    def plot(self, 
             figsize: Tuple[int, int] = (12, 8),
             save_path: Optional[str] = None,
             show: bool = True,
             dpi: int = 100):
        """
        Generate visualization of the discovery results.
        
        Args:
            figsize: Figure size (width, height)
            save_path: Path to save the figure, None means don't save
            show: Whether to display the figure
            dpi: Image resolution
        """
        if not self.phases:
            self._logger.warning("No phases to plot")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            fig.suptitle(self.title, fontsize=14, fontweight='bold')
            
            # 1. Error distribution
            ax = axes[0, 0]
            errors = [p.error for p in self.phases]
            if errors:
                x_pos = range(len(errors))
                bars = ax.bar(x_pos, errors, color='skyblue', alpha=0.7)
                ax.set_xlabel('Phase')
                ax.set_ylabel('Error')
                ax.set_title('Error Distribution')
                ax.grid(True, alpha=0.3)
                # Add value labels
                for bar, err in zip(bars, errors):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{err:.3f}', ha='center', va='bottom', fontsize=8)
            
            # 2. Confidence levels
            ax = axes[0, 1]
            confidences = [p.confidence for p in self.phases]
            if confidences:
                x_pos = range(len(confidences))
                bars = ax.bar(x_pos, confidences, color='lightgreen', alpha=0.7)
                ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='High confidence threshold')
                ax.set_xlabel('Phase')
                ax.set_ylabel('Confidence')
                ax.set_title('Confidence Levels')
                ax.legend()
                ax.grid(True, alpha=0.3)
                # Add value labels
                for bar, conf in zip(bars, confidences):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{conf:.2f}', ha='center', va='bottom', fontsize=8)
            
            # 3. Phase ranges
            ax = axes[1, 0]
            for i, phase in enumerate(self.phases):
                range_width = phase.range[1] - phase.range[0]
                ax.barh(i, range_width, 
                       left=phase.range[0], height=0.5,
                       alpha=0.7, color='lightcoral')
                # Add error label
                ax.text(phase.range[0] + range_width/2, i,
                       f'E={phase.error:.3f}', ha='center', va='center', 
                       fontsize=8, color='white', fontweight='bold')
            ax.set_xlabel('x Range')
            ax.set_ylabel('Phase')
            ax.set_title('Phase Coverage Ranges')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.5, len(self.phases) - 0.5)
            
            # 4. Summary information
            ax = axes[1, 1]
            ax.axis('off')
            
            summary = self.get_summary()
            info_text = f"""
            📊 Discovery Summary
            ====================
            
            Number of phases: {summary['n_phases']}
            
            Error Statistics:
              • Average: {summary['avg_error']:.6f}
              • Minimum: {summary['min_error']:.6f}
              • Maximum: {summary['max_error']:.6f}
            
            Confidence Statistics:
              • Average: {summary['avg_confidence']:.1%}
            
            Coverage:
              • Total range: {summary['total_range']:.2f}
              • Coverage: {summary['coverage']:.1%}
            
            Created:
              {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            ax.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
                   transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3),
                   fontfamily='monospace')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                self._logger.info(f"Figure saved to {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            self._logger.error(f"Failed to generate plot: {e}")
            plt.close()
    
    def save(self, filename: str, format: str = 'json') -> bool:
        """
        Save report to file.
        
        Args:
            filename: Output filename
            format: Format ('json', 'txt', 'latex', 'html', 'pdf')
            
        Returns:
            bool: Success status
        """
        try:
            # Ensure directory exists
            filepath = Path(filename)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'json':
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            
            elif format == 'txt':
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.to_text())
            
            elif format == 'latex':
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.to_latex())
            
            elif format == 'html':
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.to_html())
            
            elif format == 'pdf':
                # Requires reportlab
                try:
                    from reportlab.lib import colors
                    from reportlab.lib.pagesizes import letter
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    from reportlab.lib.units import inch
                    
                    doc = SimpleDocTemplate(filename, pagesize=letter)
                    styles = getSampleStyleSheet()
                    story = []
                    
                    # Title
                    story.append(Paragraph(self.title, styles['Title']))
                    story.append(Spacer(1, 0.2*inch))
                    
                    # Basic information
                    story.append(Paragraph(f"Data Source: {self.data_source}", styles['Normal']))
                    story.append(Paragraph(f"Created: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
                    story.append(Spacer(1, 0.2*inch))
                    
                    # Summary
                    summary = self.get_summary()
                    story.append(Paragraph("Discovery Summary", styles['Heading2']))
                    story.append(Paragraph(f"Number of phases: {summary['n_phases']}", styles['Normal']))
                    if summary['n_phases'] > 0:
                        story.append(Paragraph(f"Average error: {summary['avg_error']:.6f}", styles['Normal']))
                        story.append(Paragraph(f"Average confidence: {summary['avg_confidence']:.1%}", styles['Normal']))
                    story.append(Spacer(1, 0.2*inch))
                    
                    # Phases
                    if self.phases:
                        story.append(Paragraph("Discovered Patterns", styles['Heading2']))
                        for i, phase in enumerate(self.phases):
                            story.append(Paragraph(f"Phase {i+1}: {phase.formula}", styles['Heading3']))
                            story.append(Paragraph(f"Parameters: {phase.params}", styles['Normal']))
                            story.append(Paragraph(f"Range: [{phase.range[0]:.2f}, {phase.range[1]:.2f}]", styles['Normal']))
                            story.append(Paragraph(f"Error: {phase.error:.6f}, Confidence: {phase.confidence:.1%}", styles['Normal']))
                            story.append(Spacer(1, 0.1*inch))
                    else:
                        story.append(Paragraph("No patterns discovered", styles['Normal']))
                    
                    doc.build(story)
                    
                except ImportError:
                    self._logger.error("PDF export requires reportlab: pip install reportlab")
                    return False
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self._logger.info(f"Report saved to {filename}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to save report: {e}")
            return False
    
    def print(self):
        """Print report to console."""
        print(self.to_text())
    
    def __len__(self) -> int:
        """Return number of phases."""
        return len(self.phases)
    
    def __getitem__(self, index: int) -> DiscoveryPhase:
        """Support indexing."""
        return self.phases[index]
    
    def __iter__(self):
        """Support iteration."""
        return iter(self.phases)


# ============================================================
# Usage Examples
# ============================================================
"""
>>> from metathin_sci.discovery import DiscoveryReport, DiscoveryPhase
>>> 
>>> # 1. Create report
>>> report = DiscoveryReport(
...     title="Pendulum Motion Discovery Report",
...     data_source="Physics experiment data"
... )
>>> 
>>> # 2. Add discovery phases
>>> phase1 = DiscoveryPhase(
...     formula="θ₀*cos(ω*t)",
...     params={'θ₀': 0.5, 'ω': 2.0},
...     range=(0.0, 5.0),
...     error=0.0012,
...     confidence=0.95,
...     description="Small angle approximation, simple harmonic motion"
... )
>>> report.add_phase(phase1)
>>> 
>>> phase2 = DiscoveryPhase(
...     formula="θ₀*cos(ω*t) - (θ₀³/16)*cos(3ω*t)",
...     params={'θ₀': 0.5, 'ω': 2.0},
...     range=(5.0, 10.0),
...     error=0.0085,
...     confidence=0.82,
...     description="Nonlinear correction"
... )
>>> report.add_phase(phase2)
>>> 
>>> # 3. Print report
>>> report.print()
>>> 
>>> # 4. Save report
>>> report.save('discovery.json', 'json')
>>> report.save('discovery.pdf', 'pdf')
>>> 
>>> # 5. Generate visualization
>>> report.plot()
"""