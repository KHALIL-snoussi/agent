"""
Quality gates validation module for QBRIX diamond painting kits.
Enforces all required constraints and provides automatic fixes where possible.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from .grid_index_map import GridIndexMap
from .print_math import PrintSpecs, PrintMathEngine, GridSpecs
from .fixed_palettes import get_fixed_palette_manager
from .dmc import DMCColor


@dataclass
class QualityGateResult:
    """Result of quality gate validation."""
    passed: bool
    warnings: List[str]
    errors: List[str]
    auto_fixes: List[str]
    metrics: Dict[str, Any]


class QualityGates:
    """Quality gates enforcement for QBRIX diamond painting kits."""
    
    def __init__(self, print_specs: PrintSpecs):
        """Initialize quality gates with print specifications."""
        self.print_specs = print_specs
        self.print_engine = PrintMathEngine(print_specs)
        self.palette_manager = get_fixed_palette_manager()
    
    def validate_all(self, grid_map: GridIndexMap, 
                    scale_factor: float = 1.0) -> QualityGateResult:
        """
        Run all quality gates validations.
        
        Args:
            grid_map: Grid index map to validate
            scale_factor: Scale factor applied to original image
            
        Returns:
            QualityGateResult with all validation results
        """
        warnings = []
        errors = []
        auto_fixes = []
        metrics = {}
        
        # 1. Palette invariance validation
        palette_result = self.validate_palette_invariance(grid_map)
        warnings.extend(palette_result.warnings)
        errors.extend(palette_result.errors)
        auto_fixes.extend(palette_result.auto_fixes)
        metrics.update(palette_result.metrics)
        
        # 2. Cell cap validation
        cell_result = self.validate_cell_cap(grid_map.grid_specs)
        warnings.extend(cell_result.warnings)
        errors.extend(cell_result.errors)
        auto_fixes.extend(cell_result.auto_fixes)
        metrics.update(cell_result.metrics)
        
        # 3. Symbol legibility validation
        symbol_result = self.validate_symbol_legibility()
        warnings.extend(symbol_result.warnings)
        errors.extend(symbol_result.errors)
        auto_fixes.extend(symbol_result.auto_fixes)
        metrics.update(symbol_result.metrics)
        
        # 4. Tiling coverage validation
        tiling_result = self.validate_tiling_coverage(grid_map.grid_specs)
        warnings.extend(tiling_result.warnings)
        errors.extend(tiling_result.errors)
        auto_fixes.extend(tiling_result.auto_fixes)
        metrics.update(tiling_result.metrics)
        
        # 5. Color distribution validation
        color_result = self.validate_color_distribution(grid_map)
        warnings.extend(color_result.warnings)
        errors.extend(color_result.errors)
        auto_fixes.extend(color_result.auto_fixes)
        metrics.update(color_result.metrics)
        
        # 6. Scale factor validation
        scale_result = self.validate_scale_factor(scale_factor)
        warnings.extend(scale_result.warnings)
        errors.extend(scale_result.errors)
        auto_fixes.extend(scale_result.auto_fixes)
        metrics.update(scale_result.metrics)
        
        # Overall result
        passed = len(errors) == 0
        return QualityGateResult(passed, warnings, errors, auto_fixes, metrics)
    
    def validate_palette_invariance(self, grid_map: GridIndexMap) -> QualityGateResult:
        """Validate that palette is exactly one of the fixed 7-color palettes."""
        warnings = []
        errors = []
        auto_fixes = []
        metrics = {}
        
        # Check palette size
        if len(grid_map.palette_colors) != 7:
            errors.append(f"Palette has {len(grid_map.palette_colors)} colors, must be exactly 7")
        
        # Check DMC codes match fixed palettes
        dmc_codes = [color.dmc_code for color in grid_map.palette_colors]
        
        # Try to match against each fixed palette
        matched_style = None
        for style_name in self.palette_manager.list_styles():
            palette = self.palette_manager.get_palette(style_name)
            if dmc_codes == palette.dmc_codes:
                matched_style = style_name
                break
        
        if matched_style is None:
            errors.append("Palette does not match any fixed style (ORIGINAL, VINTAGE, POPART)")
            errors.append(f"Current DMC codes: {dmc_codes}")
        else:
            # Verify style name matches
            if grid_map.style_name != matched_style:
                warnings.append(f"Style name mismatch: grid says '{grid_map.style_name}', palette matches '{matched_style}'")
        
        # Check cluster indices are within range
        unique_indices = np.unique(grid_map.grid_data)
        if not all(0 <= idx < 7 for idx in unique_indices):
            errors.append(f"Grid contains cluster indices outside 0-6 range: {unique_indices}")
        
        metrics.update({
            'palette_size': len(grid_map.palette_colors),
            'dmc_codes': dmc_codes,
            'style_matched': matched_style,
            'cluster_indices': unique_indices.tolist(),
            'palette_invariant': matched_style is not None
        })
        
        return QualityGateResult(
            len(errors) == 0, warnings, errors, auto_fixes, metrics
        )
    
    def validate_cell_cap(self, grid_specs: GridSpecs) -> QualityGateResult:
        """Validate that total cells <= 10,000."""
        warnings = []
        errors = []
        auto_fixes = []
        metrics = {}
        
        total_cells = grid_specs.total_cells
        
        if total_cells > 10000:
            errors.append(f"Grid exceeds 10,000 cell limit: {total_cells:,} cells")
            
            # Suggest auto-fix
            scale_factor = math.sqrt(10000 / total_cells)
            suggested_cols = max(1, int(round(grid_specs.cols * scale_factor)))
            suggested_rows = max(1, int(round(grid_specs.rows * scale_factor)))
            suggested_total = suggested_cols * suggested_rows
            
            auto_fixes.append(
                f"Auto-fix available: scale to {suggested_cols}x{suggested_rows} "
                f"({suggested_total:,} cells, {scale_factor:.3f}x scale)"
            )
        elif total_cells < 1000:
            warnings.append(f"Very small grid: {total_cells:,} cells (may lack detail)")
        else:
            # Good size range
            pass
        
        metrics.update({
            'total_cells': total_cells,
            'under_limit': total_cells <= 10000,
            'cell_utilization': total_cells / 10000,
            'grid_area_mm2': grid_specs.cols * grid_specs.rows * (self.print_specs.cell_size_mm ** 2)
        })
        
        return QualityGateResult(
            total_cells <= 10000, warnings, errors, auto_fixes, metrics
        )
    
    def validate_symbol_legibility(self) -> QualityGateResult:
        """Validate symbol legibility requirements (x-height >= 1.2mm, stroke >= 0.15mm)."""
        warnings = []
        errors = []
        auto_fixes = []
        metrics = {}
        
        cell_size_mm = self.print_specs.cell_size_mm
        
        # Calculate minimum cell size for 1.2mm x-height
        # For bold digits 1-7, x-height is ~43% of cell height
        min_cell_for_xheight = 1.2 / 0.43  # approx 2.79mm
        
        if cell_size_mm < min_cell_for_xheight:
            errors.append(
                f"Cell size {cell_size_mm:.2f}mm too small for 1.2mm x-height "
                f"(minimum {min_cell_for_xheight:.2f}mm required)"
            )
            
            # Suggest auto-fix
            suggested_size = max(2.3, min(3.0, min_cell_for_xheight))
            auto_fixes.append(f"Auto-fix: increase cell size to {suggested_size:.2f}mm")
        
        # Stroke thickness validation (minimum 0.15mm at print scale)
        # For digits, we use ~8% of cell size for stroke
        stroke_thickness_mm = cell_size_mm * 0.08
        min_stroke_mm = 0.15
        
        if stroke_thickness_mm < min_stroke_mm:
            warnings.append(
                f"Stroke thickness {stroke_thickness_mm:.2f}mm below recommended "
                f"{min_stroke_mm:.2f}mm (may affect print clarity)"
            )
        
        # Calculate actual x-height
        actual_xheight = cell_size_mm * 0.43
        
        metrics.update({
            'cell_size_mm': cell_size_mm,
            'x_height_mm': actual_xheight,
            'stroke_thickness_mm': stroke_thickness_mm,
            'x_height_adequate': actual_xheight >= 1.2,
            'stroke_adequate': stroke_thickness_mm >= 0.15,
            'min_cell_size_for_xheight': min_cell_for_xheight
        })
        
        return QualityGateResult(
            actual_xheight >= 1.2 and stroke_thickness_mm >= 0.15,
            warnings, errors, auto_fixes, metrics
        )
    
    def validate_tiling_coverage(self, grid_specs: GridSpecs) -> QualityGateResult:
        """Validate that tiling covers all cells without gaps."""
        warnings = []
        errors = []
        auto_fixes = []
        metrics = {}
        
        tiles = self.print_engine.calculate_tiling(grid_specs)
        
        # Check coverage
        covered_cells = set()
        for tile in tiles:
            for y in range(tile.y_start, tile.y_start + tile.tile_rows):
                for x in range(tile.x_start, tile.x_start + tile.tile_cols):
                    if 0 <= y < grid_specs.rows and 0 <= x < grid_specs.cols:
                        covered_cells.add((x, y))
        
        total_required_cells = grid_specs.total_cells
        total_covered_cells = len(covered_cells)
        
        if total_covered_cells < total_required_cells:
            errors.append(
                f"Tiling coverage incomplete: {total_covered_cells:,}/{total_required_cells:,} cells covered"
            )
        elif total_covered_cells > total_required_cells:
            warnings.append(
                f"Tiling over-coverage: {total_covered_cells:,} cells covered (expected {total_required_cells:,})"
            )
        
        # Check for overlaps
        overlaps = 0
        cell_coverage = {}
        for tile in tiles:
            for y in range(tile.y_start, tile.y_start + tile.tile_rows):
                for x in range(tile.x_start, tile.x_start + tile.tile_cols):
                    if 0 <= y < grid_specs.rows and 0 <= x < grid_specs.cols:
                        cell_key = (x, y)
                        if cell_key in cell_coverage:
                            overlaps += 1
                        cell_coverage[cell_key] = cell_coverage.get(cell_key, 0) + 1
        
        if overlaps > 0:
            metrics['overlap_cells'] = overlaps
            if overlaps > 100:  # Significant overlap
                warnings.append(f"High tiling overlap: {overlaps:,} cells covered multiple times")
        
        # Calculate tiling efficiency
        tile_area = sum(tile.tile_cols * tile.tile_rows for tile in tiles)
        efficiency = total_required_cells / tile_area if tile_area > 0 else 0
        
        metrics.update({
            'total_tiles': len(tiles),
            'total_covered_cells': total_covered_cells,
            'total_required_cells': total_required_cells,
            'coverage_complete': total_covered_cells >= total_required_cells,
            'overlap_cells': overlaps,
            'tiling_efficiency': efficiency,
            'cells_per_tile_avg': tile_area / len(tiles) if tiles else 0
        })
        
        return QualityGateResult(
            total_covered_cells >= total_required_cells, warnings, errors, auto_fixes, metrics
        )
    
    def validate_color_distribution(self, grid_map: GridIndexMap) -> QualityGateResult:
        """Validate color distribution and detect rare colors."""
        warnings = []
        errors = []
        auto_fixes = []
        metrics = {}
        
        # Count usage of each color
        unique_indices, counts = np.unique(grid_map.grid_data, return_counts=True)
        total_cells = grid_map.grid_specs.total_cells
        
        color_usage = {}
        for idx, count in zip(unique_indices, counts):
            if idx < len(grid_map.palette_colors):
                color = grid_map.palette_colors[idx]
                percentage = (count / total_cells) * 100
                color_usage[color.dmc_code] = {
                    'count': int(count),
                    'percentage': percentage,
                    'rare': percentage < 2.0
                }
                
                if percentage < 2.0:
                    warnings.append(
                        f"Rare color: {color.dmc_code} ({color.name}) "
                        f"only {percentage:.1f}% of grid ({count:,} cells)"
                    )
                elif percentage < 0.5:
                    warnings.append(
                        f"Very rare color: {color.dmc_code} ({color.name}) "
                        f"only {percentage:.1f}% of grid ({count:,} cells) - consider different crop"
                    )
        
        # Check for unused colors in palette
        used_indices = set(unique_indices)
        for i in range(len(grid_map.palette_colors)):
            if i not in used_indices:
                color = grid_map.palette_colors[i]
                warnings.append(f"Unused color in palette: {color.dmc_code} ({color.name})")
        
        # Calculate color balance metrics
        percentages = [count / total_cells for count in counts]
        color_balance = 1.0 - np.std(percentages)  # Higher = more balanced
        
        metrics.update({
            'unique_colors_used': len(unique_indices),
            'total_palette_colors': len(grid_map.palette_colors),
            'color_usage': color_usage,
            'rare_colors': sum(1 for usage in color_usage.values() if usage['rare']),
            'color_balance_score': color_balance,
            'most_used_percentage': max(percentages) * 100,
            'least_used_percentage': min(percentages) * 100
        })
        
        return QualityGateResult(
            len(errors) == 0, warnings, errors, auto_fixes, metrics
        )
    
    def validate_scale_factor(self, scale_factor: float) -> QualityGateResult:
        """Validate scale factor applied to original image."""
        warnings = []
        errors = []
        auto_fixes = []
        metrics = {}
        
        if scale_factor < 0.1:
            errors.append(f"Scale factor too small: {scale_factor:.3f} (minimum 0.1)")
        elif scale_factor < 0.5:
            warnings.append(f"Significant downscaling: {scale_factor:.3f} (may lose detail)")
        elif scale_factor > 1.0:
            warnings.append(f"Upscaling applied: {scale_factor:.3f} (may reduce quality)")
        
        # Quality categories
        if scale_factor >= 0.8:
            quality_category = "Excellent"
        elif scale_factor >= 0.6:
            quality_category = "Good"
        elif scale_factor >= 0.4:
            quality_category = "Acceptable"
        elif scale_factor >= 0.2:
            quality_category = "Poor"
        else:
            quality_category = "Very Poor"
        
        metrics.update({
            'scale_factor': scale_factor,
            'quality_category': quality_category,
            'detail_retention_estimate': scale_factor,
            'upscaling_applied': scale_factor > 1.0,
            'significant_downscale': scale_factor < 0.5
        })
        
        return QualityGateResult(
            0.1 <= scale_factor <= 2.0, warnings, errors, auto_fixes, metrics
        )
    
    def generate_quality_report(self, result: QualityGateResult) -> str:
        """Generate human-readable quality report."""
        lines = []
        
        lines.append("QUALITY GATES REPORT")
        lines.append("=" * 50)
        
        # Overall status
        if result.passed:
            lines.append("[OK] ALL QUALITY GATES PASSED")
        else:
            lines.append("[X] QUALITY GATES FAILED")
        
        lines.append("")
        
        # Warnings
        if result.warnings:
            lines.append("WARNINGS:")
            for warning in result.warnings:
                lines.append(f"  [WARN] {warning}")
            lines.append("")
        
        # Errors
        if result.errors:
            lines.append("ERRORS:")
            for error in result.errors:
                lines.append(f"  [X] {error}")
            lines.append("")
        
        # Auto-fixes
        if result.auto_fixes:
            lines.append("AVAILABLE AUTO-FIXES:")
            for fix in result.auto_fixes:
                lines.append(f"  [fix] {fix}")
            lines.append("")
        
        # Key metrics
        lines.append("KEY METRICS:")
        if 'total_cells' in result.metrics:
            lines.append(f"  Grid size: {result.metrics['total_cells']:,} cells")
        if 'cell_size_mm' in result.metrics:
            lines.append(f"  Cell size: {result.metrics['cell_size_mm']:.2f}mm")
        if 'x_height_mm' in result.metrics:
            lines.append(f"  Symbol x-height: {result.metrics['x_height_mm']:.2f}mm")
        if 'scale_factor' in result.metrics:
            lines.append(f"  Scale factor: {result.metrics['scale_factor']:.3f}")
        if 'unique_colors_used' in result.metrics:
            lines.append(f"  Colors used: {result.metrics['unique_colors_used']}/7")
        if 'color_balance_score' in result.metrics:
            lines.append(f"  Color balance: {result.metrics['color_balance_score']:.3f}")
        
        return "\n".join(lines)


def run_quality_gates(grid_map: GridIndexMap, print_specs: PrintSpecs,
                     scale_factor: float = 1.0) -> QualityGateResult:
    """
    Convenience function to run all quality gates.
    
    Args:
        grid_map: Grid index map to validate
        print_specs: Print specifications
        scale_factor: Scale factor applied to original image
        
    Returns:
        QualityGateResult with all validation results
    """
    gates = QualityGates(print_specs)
    return gates.validate_all(grid_map, scale_factor)
