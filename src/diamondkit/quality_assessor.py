"""
Quality assessment system for diamond painting kits.
Handles SSIM, DeltaE validation, rare color detection, and quality gates.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, lab2rgb
import warnings

from .dmc import DMCColor
from .grid_index_map import GridIndexMap
from .print_math import GridSpecs


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for diamond painting kit."""
    # Image quality metrics
    ssim_score: float
    delta_e_mean: float
    delta_e_max: float
    delta_e_std: float
    
    # Color distribution metrics
    rare_colors: List[str]  # DMC codes with <2% usage
    color_distribution: Dict[str, float]  # DMC code -> percentage
    
    # Grid compliance
    total_cells: int
    grid_constraints_met: bool
    
    # Quality gates
    quality_warnings: List[str]
    quality_risks: List[str]
    
    # Recommendations
    auto_fixes: List[str]
    overall_quality: str  # "EXCELLENT", "GOOD", "ACCEPTABLE", "NEEDS_IMPROVEMENT"


class QualityAssessor:
    """Assesses quality of diamond painting kits against specifications."""
    
    def __init__(self):
        """Initialize quality assessor with thresholds."""
        # Quality thresholds (as specified in requirements)
        self.SSIM_THRESHOLD = 0.75
        self.DELTA_E_MAX_THRESHOLD = 12.0
        self.RARE_COLOR_THRESHOLD = 0.02  # 2%
        self.MAX_CELLS_THRESHOLD = 10000
        
        # Color accuracy risk levels
        self.DELTA_E_RISK_LEVELS = {
            "LOW": (0.0, 6.0),
            "MEDIUM": (6.0, 12.0),
            "HIGH": (12.0, float('inf'))
        }
    
    def assess_kit_quality(self, original_rgb: np.ndarray, 
                         quantized_rgb: np.ndarray,
                         grid_map: GridIndexMap,
                         original_lab: Optional[np.ndarray] = None) -> QualityMetrics:
        """
        Comprehensive quality assessment of diamond painting kit.
        
        Args:
            original_rgb: Original image in RGB space
            quantized_rgb: Quantized image from GRID_INDEX_MAP
            grid_map: Grid index map with fixed palette
            original_lab: Original image in Lab space (optional, calculated if needed)
            
        Returns:
            Comprehensive quality metrics. The QualityMetrics struct is later surfaced
            on the PDF title page, written into kit_metadata.json, and used by the CLI
            to raise warnings about ΔE/SSIM/color-distribution issues.
        """
        print("Assessing diamond painting kit quality...")
        
        # Convert to Lab if needed
        if original_lab is None:
            original_lab = self._rgb_to_lab(original_rgb)
        
        # Calculate image quality metrics
        ssim_score = self._calculate_ssim(original_rgb, quantized_rgb)
        delta_e_stats = self._calculate_delta_e_statistics(original_lab, grid_map)
        
        # Analyze color distribution
        color_dist, rare_colors = self._analyze_color_distribution(grid_map)
        
        # Check grid compliance
        grid_compliance = self._check_grid_compliance(grid_map.grid_specs)
        
        # Generate warnings and risks
        warnings, risks, fixes = self._generate_quality_warnings(
            ssim_score, delta_e_stats, rare_colors, grid_map.grid_specs
        )
        
        # Determine overall quality
        overall_quality = self._determine_overall_quality(
            ssim_score, delta_e_stats, warnings, grid_compliance
        )
        
        return QualityMetrics(
            ssim_score=ssim_score,
            delta_e_mean=delta_e_stats["delta_e_mean"],
            delta_e_max=delta_e_stats["delta_e_max"],
            delta_e_std=delta_e_stats["delta_e_std"],
            rare_colors=rare_colors,
            color_distribution=color_dist,
            total_cells=grid_map.grid_specs.total_cells,
            grid_constraints_met=grid_compliance,
            quality_warnings=warnings,
            quality_risks=risks,
            auto_fixes=fixes,
            overall_quality=overall_quality
        )
    
    def _calculate_ssim(self, original_rgb: np.ndarray, 
                       quantized_rgb: np.ndarray) -> float:
        """Calculate Structural Similarity Index."""
        # Ensure same dimensions
        if original_rgb.shape != quantized_rgb.shape:
            raise ValueError("Images must have same dimensions for SSIM calculation")
        
        # Convert to grayscale for SSIM (more reliable for structural comparison)
        def rgb_to_gray(rgb):
            return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        
        original_gray = rgb_to_gray(original_rgb)
        quantized_gray = rgb_to_gray(quantized_rgb)
        
        # Calculate SSIM with appropriate window size based on image size
        min_dim = min(original_gray.shape)
        win_size = min(7, max(3, min_dim // 20))
        
        # Ensure window size is odd
        if win_size % 2 == 0:
            win_size += 1
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = ssim(
                original_gray, 
                quantized_gray, 
                data_range=255, 
                win_size=win_size,
                gaussian_weights=True,
                use_sample_covariance=False
            )
        
        return float(score)
    
    def _calculate_delta_e_statistics(self, original_lab: np.ndarray, 
                                 grid_map: GridIndexMap) -> Dict[str, float]:
        """Calculate DeltaE statistics between original and quantized images."""
        from .grid_index_map import ColorQuantizerFixed
        
        # Use the quantizer's method for consistency
        quantizer = ColorQuantizerFixed(grid_map.style_name)
        return quantizer.calculate_delta_e_stats(original_lab, grid_map)
    
    def _analyze_color_distribution(self, grid_map: GridIndexMap) -> Tuple[Dict[str, float], List[str]]:
        """Analyze color distribution and identify rare colors."""
        h, w = grid_map.grid_data.shape
        total_pixels = h * w
        
        # Count occurrences of each cluster index
        unique_indices, counts = np.unique(grid_map.grid_data, return_counts=True)
        
        color_distribution = {}
        rare_colors = []
        
        for idx, count in zip(unique_indices, counts):
            if idx < len(grid_map.palette_colors):
                dmc_code = grid_map.palette_colors[idx].dmc_code
                percentage = (count / total_pixels) * 100
                color_distribution[dmc_code] = percentage
                
                # Check if rare
                if percentage < (self.RARE_COLOR_THRESHOLD * 100):
                    rare_colors.append(dmc_code)
        
        return color_distribution, rare_colors
    
    def _check_grid_compliance(self, grid_specs: GridSpecs) -> bool:
        """Check if grid meets all constraints."""
        return (
            grid_specs.total_cells <= self.MAX_CELLS_THRESHOLD and
            grid_specs.cols > 0 and
            grid_specs.rows > 0
        )
    
    def _generate_quality_warnings(self, ssim_score: float, 
                               delta_e_stats: Dict[str, float],
                               rare_colors: List[str],
                               grid_specs: GridSpecs) -> Tuple[List[str], List[str], List[str]]:
        """Generate quality warnings, risks, and auto-fix suggestions."""
        warnings = []
        risks = []
        fixes = []
        
        # SSIM quality check
        if ssim_score < self.SSIM_THRESHOLD:
            warnings.append(f"Detail loss risk: SSIM {ssim_score:.3f} < {self.SSIM_THRESHOLD}")
            risks.append("Loss of fine details in quantized result")
            fixes.append("Consider larger grid size or different crop to preserve details")
        
        # DeltaE max check
        if delta_e_stats["delta_e_max"] > self.DELTA_E_MAX_THRESHOLD:
            warnings.append(f"Color accuracy risk: DeltaE_max {delta_e_stats['delta_e_max']:.1f} > {self.DELTA_E_MAX_THRESHOLD}")
            risks.append("Some colors may not accurately represent original")
            fixes.append("Consider different style palette or preprocessing adjustments")
        
        # Rare color check
        if rare_colors:
            warnings.append(f"Rare colors detected: {', '.join(rare_colors)} (each < 2%)")
            risks.append("Difficult color management during crafting")
            fixes.append("Consider different crop to balance color distribution")
        
        # Grid size warnings
        if grid_specs.total_cells > self.MAX_CELLS_THRESHOLD:
            warnings.append(f"Grid size violation: {grid_specs.total_cells:,} > {self.MAX_CELLS_THRESHOLD:,}")
            risks.append("Exceeds maximum allowed cells")
            fixes.append("Apply automatic scaling to meet 10k limit")
        
        # Color balance warnings
        if len(rare_colors) > 2:
            warnings.append(f"Multiple rare colors ({len(rare_colors)}) may indicate poor palette match")
            fixes.append("Consider different style or image crop")
        
        return warnings, risks, fixes
    
    def _determine_overall_quality(self, ssim_score: float,
                                  delta_e_stats: Dict[str, float],
                                  warnings: List[str],
                                  grid_compliance: bool) -> str:
        """Determine overall quality rating."""
        score = 0
        
        # SSIM component (40% weight)
        if ssim_score >= 0.9:
            score += 40
        elif ssim_score >= self.SSIM_THRESHOLD:
            score += 30
        elif ssim_score >= 0.6:
            score += 20
        else:
            score += 10
        
        # DeltaE component (30% weight)
        if delta_e_stats["delta_e_max"] <= 6.0:
            score += 30
        elif delta_e_stats["delta_e_max"] <= self.DELTA_E_MAX_THRESHOLD:
            score += 20
        elif delta_e_stats["delta_e_max"] <= 18.0:
            score += 10
        else:
            score += 5
        
        # Warning component (20% weight)
        warning_penalty = min(len(warnings) * 5, 20)
        score += max(0, 20 - warning_penalty)
        
        # Compliance component (10% weight)
        if grid_compliance:
            score += 10
        else:
            score += 0
        
        # Determine quality level
        if score >= 85:
            return "EXCELLENT"
        elif score >= 70:
            return "GOOD"
        elif score >= 50:
            return "ACCEPTABLE"
        else:
            return "NEEDS_IMPROVEMENT"
    
    def _rgb_to_lab(self, rgb_array: np.ndarray) -> np.ndarray:
        """Convert RGB array to Lab color space."""
        # Normalize to 0-1 range
        rgb_normalized = rgb_array.astype(np.float32) / 255.0
        
        # Convert to Lab
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lab_array = rgb2lab(rgb_normalized)
        
        return lab_array
    
    def generate_quality_report(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        return {
            "summary": {
                "overall_quality": metrics.overall_quality,
                "ssim_score": round(metrics.ssim_score, 4),
                "delta_e_mean": round(metrics.delta_e_mean, 2),
                "delta_e_max": round(metrics.delta_e_max, 2),
                "total_cells": metrics.total_cells,
                "constraints_met": metrics.grid_constraints_met
            },
            "image_quality": {
                "ssim_assessment": self._interpret_ssim(metrics.ssim_score),
                "color_accuracy": self._interpret_delta_e(metrics.delta_e_max),
                "detail_preservation": "High" if metrics.ssim_score > 0.8 else "Medium" if metrics.ssim_score > 0.6 else "Low"
            },
            "color_analysis": {
                "total_colors": len(metrics.color_distribution),
                "rare_colors": metrics.rare_colors,
                "distribution": metrics.color_distribution,
                "balance_score": self._calculate_color_balance(metrics.color_distribution)
            },
            "quality_gates": {
                "warnings": metrics.quality_warnings,
                "risks": metrics.quality_risks,
                "auto_fixes": metrics.auto_fixes,
                "critical_issues": [w for w in metrics.quality_warnings if "risk" in w.lower()]
            },
            "recommendations": self._generate_recommendations(metrics)
        }
    
    def _interpret_ssim(self, ssim_score: float) -> str:
        """Interpret SSIM score."""
        if ssim_score >= 0.9:
            return "Excellent structural similarity"
        elif ssim_score >= self.SSIM_THRESHOLD:
            return "Good structural similarity"
        elif ssim_score >= 0.6:
            return "Moderate structural similarity"
        else:
            return "Poor structural similarity"
    
    def _interpret_delta_e(self, delta_e_max: float) -> str:
        """Interpret maximum DeltaE value."""
        if delta_e_max <= 6.0:
            return "Excellent color accuracy"
        elif delta_e_max <= self.DELTA_E_MAX_THRESHOLD:
            return "Good color accuracy"
        elif delta_e_max <= 18.0:
            return "Moderate color accuracy"
        else:
            return "Poor color accuracy"
    
    def _calculate_color_balance(self, distribution: Dict[str, float]) -> float:
        """Calculate color balance score (0-100)."""
        if not distribution:
            return 0.0
        
        percentages = list(distribution.values())
        # Ideal balance is uniform distribution
        uniform_percentage = 100.0 / len(percentages)
        deviations = [abs(p - uniform_percentage) for p in percentages]
        avg_deviation = np.mean(deviations)
        
        # Convert to score (lower deviation = higher score)
        balance_score = max(0, 100 - avg_deviation * 2)
        return round(balance_score, 1)
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Generate specific recommendations based on metrics."""
        recommendations = []
        
        # SSIM-based recommendations
        if metrics.ssim_score < 0.7:
            recommendations.append("Increase grid resolution to preserve more details")
        elif metrics.ssim_score < self.SSIM_THRESHOLD:
            recommendations.append("Consider different crop to focus on important details")
        
        # DeltaE-based recommendations
        if metrics.delta_e_max > 15.0:
            recommendations.append("Current style may not match image colors - try different style")
        elif metrics.delta_e_max > self.DELTA_E_MAX_THRESHOLD:
            recommendations.append("Some colors poorly represented - consider adjusting preprocessing")
        
        # Color distribution recommendations
        if len(metrics.rare_colors) > 3:
            recommendations.append("Many rare colors suggest poor style-image compatibility")
        elif metrics.rare_colors:
            recommendations.append("Rare colors may be difficult to work with")
        
        # Grid size recommendations
        if metrics.total_cells > 8000:
            recommendations.append("Large grid size - ensure sufficient crafting time")
        elif metrics.total_cells < 1000:
            recommendations.append("Small grid size - consider larger crop for more detail")
        
        return recommendations


def assess_quality(original_rgb: np.ndarray, quantized_rgb: np.ndarray,
                 grid_map: GridIndexMap, original_lab: Optional[np.ndarray] = None) -> QualityMetrics:
    """
    Convenience function for quality assessment.
    
    Args:
        original_rgb: Original RGB image
        quantized_rgb: Quantized RGB from grid map
        grid_map: Grid index map
        original_lab: Original Lab image (optional)
        
    Returns:
        Quality metrics
    """
    assessor = QualityAssessor()
    return assessor.assess_kit_quality(original_rgb, quantized_rgb, grid_map, original_lab)


def generate_quality_report(metrics: QualityMetrics) -> Dict[str, Any]:
    """
    Convenience function to generate quality report.
    
    Args:
        metrics: Quality metrics from assessment
        
    Returns:
        Comprehensive quality report
    """
    assessor = QualityAssessor()
    return assessor.generate_quality_report(metrics)
