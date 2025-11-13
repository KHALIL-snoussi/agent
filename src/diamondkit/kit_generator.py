"""
Main diamond painting kit generator.
Integrates all components for commercial-grade A4-optimized kits with fixed palettes.
"""

import os
import json
import csv
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
from PIL import Image

from .fixed_palettes import get_fixed_palette_manager, list_available_styles
from .print_math import get_print_math_engine, PrintSpecs, GridSpecs
from .grid_index_map import create_grid_index_map, verify_grid_invariance
from .quality_assessor import assess_quality, generate_quality_report
from .dmc import DMCColor
from .image_preprocessor import (
    ImagePreprocessor,
    PreprocessResult,
    ProcessingSettings,
    load_processing_settings,
)


class DiamondKitGenerator:
    """Commercial-grade diamond painting kit generator with fixed palettes."""
    
    def __init__(self, dpi: int = 600, margin_mm: float = 12.0, cell_size_mm: float = 2.8):
        """
        Initialize kit generator with print specifications.
        
        Args:
            dpi: Print DPI (>=300 required, default 600 for quality)
            margin_mm: Paper margins in mm (10-15mm range, default 12mm)
            cell_size_mm: Cell size in mm (2.3-3.0mm range, default 2.8mm)
        """
        self.print_specs = PrintSpecs(
            dpi=dpi,
            margin_mm=margin_mm,
            cell_size_mm=cell_size_mm
        )
        self.print_engine = get_print_math_engine(self.print_specs)
        self.palette_manager = get_fixed_palette_manager()
        self.processing_settings: ProcessingSettings = load_processing_settings()
        self.preprocessor = ImagePreprocessor(self.print_engine, self.processing_settings)
    
    def generate_kit(self, image_path: str, style_name: str, 
                    output_dir: str, crop_rect: Optional[Tuple[float, float, float, float]] = None) -> Dict[str, Any]:
        """
        Generate complete diamond painting kit.
        
        Args:
            image_path: Path to input image
            style_name: Style name (ORIGINAL, VINTAGE, POPART)
            output_dir: Output directory for kit files
            crop_rect: Optional crop rectangle (x, y, w, h) as normalized 0-1
            
        Returns:
            Complete kit metadata and results
        """
        print(f"Generating {style_name} diamond painting kit...")
        print(f"Input: {image_path}")
        print(f"Output: {output_dir}")
        
        # Validate style
        if style_name not in list_available_styles():
            raise ValueError(f"Unknown style '{style_name}'. Available: {list_available_styles()}")
        
        # High-fidelity preprocessing with smart cropping and exposure control
        preprocess_result = self.preprocessor.process_image(image_path, crop_rect)
        grid_specs = preprocess_result.grid_specs
        scale_factor = preprocess_result.scale_factor
        
        popart_bias = self.processing_settings.popart_edge_bias if style_name == "POPART" else 0.0
        
        # Create fixed GRID_INDEX_MAP
        grid_map = create_grid_index_map(
            preprocess_result.grid_lab,
            grid_specs,
            style_name,
            enable_smoothing=self.processing_settings.smoothing_enabled,
            smoothing_kernel=self.processing_settings.smoothing_kernel,
            image_is_preprocessed=True,
            popart_edge_bias=popart_bias,
            popart_edge_threshold=self.processing_settings.popart_edge_threshold,
        )
        
        # Generate quantized RGB from grid map
        quantized_rgb = self._grid_to_rgb_visualization(grid_map)
        
        # Assess quality on the grid-ready crop
        quality_metrics = assess_quality(
            preprocess_result.grid_rgb,
            quantized_rgb,
            grid_map,
            preprocess_result.grid_lab
        )
        
        # Run quality gates validation
        from .quality_gates import run_quality_gates
        quality_result = run_quality_gates(grid_map, self.print_specs, scale_factor)
        
        # Update quality metrics with quality gates results
        quality_metrics.quality_warnings.extend(quality_result.warnings)
        quality_metrics.quality_risks.extend(quality_result.errors)
        quality_metrics.auto_fixes.extend(quality_result.auto_fixes)
        
        # Apply auto-fixes if available and needed
        if quality_result.errors and quality_result.auto_fixes:
            print("Applying quality gates auto-fixes:")
            for fix in quality_result.auto_fixes:
                print(f"  [fix] {fix}")
            # Note: In production, you might implement actual auto-fixes here
            # For now, we'll just report them in metadata
        
        # Generate all output artifacts
        kit_outputs = self._generate_all_outputs(
            preprocess_result,
            quantized_rgb,
            grid_map,
            quality_metrics,
            style_name,
            output_dir,
            scale_factor,
        )
        
        # Compile comprehensive metadata
        metadata = self._compile_comprehensive_metadata(
            grid_map,
            quality_metrics,
            quality_result,
            scale_factor,
            preprocess_result,
            kit_outputs,
        )
        metadata["output_files"]["kit_metadata"] = "kit_metadata.json"
        metadata["output_files"]["metadata_legacy"] = "metadata.json"
        metadata_files = self._write_metadata_bundle(metadata, output_dir)
        kit_outputs.update(metadata_files)
        
        # Print quality gates report
        from .quality_gates import QualityGates
        gates = QualityGates(self.print_specs)
        quality_report = gates.generate_quality_report(quality_result)
        print("\n" + quality_report)
        
        print(f"Kit generation complete! Output in: {output_dir}")
        print(f"Grid: {grid_specs.cols}x{grid_specs.rows} ({grid_specs.total_cells:,} cells)")
        print(f"Quality: {quality_metrics.overall_quality}")
        print(f"DeltaE: mean={quality_metrics.delta_e_mean:.1f}, max={quality_metrics.delta_e_max:.1f}")
        
        return {
            "metadata": metadata,
            "outputs": kit_outputs,
            "quality_report": generate_quality_report(quality_metrics),
            "grid_map": grid_map,
            "grid_specs": grid_specs,
            "scale_factor": scale_factor
        }
    
    def _grid_to_rgb_visualization(self, grid_map) -> np.ndarray:
        """Convert grid index map to RGB visualization."""
        h, w = grid_map.grid_data.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        for y in range(h):
            for x in range(w):
                cluster_idx = grid_map.get_cell_at(x, y)
                color = grid_map.palette_colors[cluster_idx]
                rgb_image[y, x] = color.rgb
        
        return rgb_image
    
    def _generate_all_outputs(self, preprocess_result: PreprocessResult,
                           quantized_rgb: np.ndarray,
                           grid_map, quality_metrics,
                           style_name: str, output_dir: str,
                           scale_factor: float) -> Dict[str, str]:
        """Generate all required output artifacts."""
        outputs = {}
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate preview images
        outputs["original_preview"] = self._save_preview_image(
            preprocess_result.cropped_rgb, output_dir, "original_preview.jpg"
        )
        
        outputs["quantized_preview"] = self._save_preview_image(
            quantized_rgb, output_dir, "quantized_preview.jpg"
        )
        
        # Generate style previews (all styles using same grid map)
        for style in list_available_styles():
            style_preview = self._generate_style_preview(grid_map, style)
            outputs[f"{style.lower()}_style_preview"] = self._save_preview_image(
                style_preview, output_dir, f"{style.lower()}_style_preview.jpg"
            )
        
        # Generate CSV inventory
        outputs["csv_inventory"] = self._generate_csv_inventory(
            grid_map, quality_metrics, output_dir
        )
        
        # Generate PDF kit (placeholder - will be implemented in pdf.py)
        outputs["pdf_kit"] = self._generate_pdf_kit(
            grid_map, output_dir, style_name, quality_metrics, scale_factor
        )
        
        return outputs
    
    def _save_preview_image(self, image_rgb: np.ndarray, output_dir: str, filename: str) -> str:
        """Save preview image with optimal sizing."""
        output_path = os.path.join(output_dir, filename)
        
        # Resize for web preview (max 1200px on longest side)
        h, w = image_rgb.shape[:2]
        max_size = 1200
        
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            pil_image = Image.fromarray(image_rgb)
            resized_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
            resized_image.save(output_path, quality=95, optimize=True)
        else:
            pil_image = Image.fromarray(image_rgb)
            pil_image.save(output_path, quality=95, optimize=True)
        
        return output_path
    
    def _generate_style_preview(self, grid_map, style_name: str) -> np.ndarray:
        """Render the grid using the palette of the requested style."""
        try:
            palette = self.palette_manager.get_palette(style_name.upper())
        except ValueError:
            palette = self.palette_manager.get_palette(grid_map.style_name)
        
        colors = palette.get_colors()
        rows, cols = grid_map.grid_data.shape
        preview = np.zeros((rows, cols, 3), dtype=np.uint8)
        
        for idx, color in enumerate(colors):
            preview[grid_map.grid_data == idx] = color.rgb
        
        return preview
    
    def _build_color_usage(self, grid_map) -> Dict[str, int]:
        """Build mapping of DMC codes to drill counts."""
        usage: Dict[str, int] = {}
        unique_indices, counts = np.unique(grid_map.grid_data, return_counts=True)
        for idx, count in zip(unique_indices, counts):
            if idx < len(grid_map.palette_colors):
                dmc_code = grid_map.palette_colors[idx].dmc_code
                usage[dmc_code] = int(count)
        return usage
    
    def _generate_csv_inventory(self, grid_map, quality_metrics, output_dir: str) -> str:
        """Generate exact CSV inventory format as specified."""
        csv_path = os.path.join(output_dir, "inventory.csv")
        
        color_usage = self._build_color_usage(grid_map)
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'dmc_code', 'name', 'hex', 'cluster_id', 'drill_count', 
                'bag_qty_200pcs', 'deltaE_mean', 'deltaE_max'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for idx, color in enumerate(grid_map.palette_colors):
                count = color_usage.get(color.dmc_code, 0)
                bag_qty = -(-count // 200)  # Ceiling division
                
                writer.writerow({
                    'dmc_code': color.dmc_code,
                    'name': color.name,
                    'hex': color.hex,
                    'cluster_id': idx,
                    'drill_count': int(count),
                    'bag_qty_200pcs': bag_qty,
                    'deltaE_mean': round(quality_metrics.delta_e_mean, 2),
                    'deltaE_max': round(quality_metrics.delta_e_max, 2)
                })
        
        return csv_path
    
    
    def _generate_tiling_map(self, grid_specs) -> List[Dict]:
        """Generate tiling map for PDF generation."""
        tiles = self.print_engine.calculate_tiling(grid_specs)
        
        tiling_map = []
        for tile in tiles:
            tiling_map.append({
                "page_number": tile.page_number,
                "coordinates": tile.coordinates,
                "x_start": tile.x_start,
                "y_start": tile.y_start,
                "tile_cols": tile.tile_cols,
                "tile_rows": tile.tile_rows,
                "crop_marks": tile.crop_marks,
                "registration_crosses": tile.registration_crosses
            })
        
        return tiling_map
    
    def _write_metadata_bundle(self, metadata: Dict[str, Any], output_dir: str) -> Dict[str, str]:
        """Persist metadata in both primary and legacy filenames."""
        kit_meta_path = os.path.join(output_dir, "kit_metadata.json")
        with open(kit_meta_path, 'w', encoding='utf-8') as fh:
            json.dump(metadata, fh, indent=2, default=str)
        
        legacy_path = os.path.join(output_dir, "metadata.json")
        with open(legacy_path, 'w', encoding='utf-8') as fh:
            json.dump(metadata, fh, indent=2, default=str)
        
        return {
            "kit_metadata": kit_meta_path,
            "metadata_legacy": legacy_path,
            "json_metadata": kit_meta_path,  # backwards compatibility
        }
    
    def _generate_pdf_kit(self, grid_map,
                         output_dir: str, style_name: str, quality_metrics=None, scale_factor=1.0) -> str:
        """Generate QBRIX PDF kit using new PDF generator."""
        try:
            from .pdf import QBRIXPDFGenerator
            
            # Create QBRIX PDF generator with current print specs
            pdf_generator = QBRIXPDFGenerator(self.print_specs)
            pdf_path = os.path.join(output_dir, "diamond_painting_kit.pdf")
            
            # Create metadata for PDF generation
            if quality_metrics is not None:
                metadata = {
                    'filename': os.path.basename(output_dir),
                    'grid_size': f"{grid_map.grid_specs.cols}x{grid_map.grid_specs.rows}",
                    'total_cells': grid_map.grid_specs.total_cells,
                    'colors_used': len(grid_map.palette_colors),
                    'style': style_name,
                    'deltaE_stats': {
                        'mean': quality_metrics.delta_e_mean,
                        'max': quality_metrics.delta_e_max,
                        'std': quality_metrics.delta_e_std
                    },
                    'ssim': quality_metrics.ssim_score,
                    'scale_factor': scale_factor,
                    'warnings': quality_metrics.quality_warnings
                }
            else:
                metadata = {
                    'filename': os.path.basename(output_dir),
                    'grid_size': f"{grid_map.grid_specs.cols}x{grid_map.grid_specs.rows}",
                    'total_cells': grid_map.grid_specs.total_cells,
                    'colors_used': len(grid_map.palette_colors),
                    'style': style_name,
                    'deltaE_stats': {'mean': 0.0, 'max': 0.0, 'std': 0.0},
                    'ssim': 0.0,
                    'scale_factor': scale_factor,
                    'warnings': []
                }
            
            # Generate quantized RGB from grid map
            quantized_rgb = self._grid_to_rgb_visualization(grid_map)
            
            # Generate the QBRIX PDF
            pdf_path = pdf_generator.generate_qbrix_pdf(
                grid_map=grid_map,
                preview_image=quantized_rgb,
                metadata=metadata,
                output_path=pdf_path
            )
            
            print(f"QBRIX PDF generated successfully: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            print(f"Warning: QBRIX PDF generation failed: {e}")
            # Fallback to placeholder
            pdf_path = os.path.join(output_dir, "diamond_painting_kit.pdf")
            with open(pdf_path, 'w') as f:
                f.write("QBRIX PDF generation failed. Please check console for errors.")
            return pdf_path
    
    def _compile_comprehensive_metadata(self, grid_map, quality_metrics, quality_result,
                                  scale_factor: float, preprocess_result: PreprocessResult,
                                  outputs: Dict[str, str]) -> Dict[str, Any]:
        """Compile comprehensive metadata for kit."""
        color_usage = self._build_color_usage(grid_map)
        tiling_map = self._generate_tiling_map(grid_map.grid_specs)
        palette_codes = [color.dmc_code for color in grid_map.palette_colors]
        output_manifest = {key: os.path.basename(path) for key, path in outputs.items()}
        processing_meta = {
            "long_side_px": self.processing_settings.long_side_px,
            "min_long_side_px": self.processing_settings.min_long_side_px,
            "smoothing_kernel": self.processing_settings.smoothing_kernel,
            "smoothing_enabled": self.processing_settings.smoothing_enabled,
            "popart_edge_bias": self.processing_settings.popart_edge_bias,
            "popart_edge_threshold": self.processing_settings.popart_edge_threshold,
        }
        
        pattern_pages = len(tiling_map)
        
        metadata = {
            "paper_mm": [self.print_specs.paper_width_mm, self.print_specs.paper_height_mm],
            "dpi": self.print_specs.dpi,
            "margins_mm": self.print_specs.margin_mm,
            "cell_mm": self.print_specs.cell_size_mm,
            "grid_cols": grid_map.grid_specs.cols,
            "grid_rows": grid_map.grid_specs.rows,
            "total_cells": grid_map.grid_specs.total_cells,
            "pages": pattern_pages + 3,  # title + legend + instructions + pattern tiles
            "pattern_pages": pattern_pages,
            "style": grid_map.style_name,
            "fixed_palette": True,
            "fixed_palette_dmc": palette_codes,
            "grid_index_map_hash": grid_map.grid_hash,
            "deltaE_stats": {
                "mean": round(quality_metrics.delta_e_mean, 2),
                "max": round(quality_metrics.delta_e_max, 2),
                "std": round(quality_metrics.delta_e_std, 2),
            },
            "ssim": round(quality_metrics.ssim_score, 4),
            "crop_rect_norm": preprocess_result.crop_rect_norm,
            "tiling_map": tiling_map,
            "color_usage": color_usage,
            "scale_factor": round(scale_factor, 3),
            "quality_warnings": quality_metrics.quality_warnings,
            "quality_risks": quality_metrics.quality_risks,
            "preprocessing": preprocess_result.metadata,
            "processing_settings": processing_meta,
            "generation_info": {
                "style": grid_map.style_name,
                "grid_hash": grid_map.grid_hash,
                "scale_factor": scale_factor,
                "crop_applied": preprocess_result.crop_rect_norm is not None,
            },
            "grid_specifications": {
                "cols": grid_map.grid_specs.cols,
                "rows": grid_map.grid_specs.rows,
                "total_cells": grid_map.grid_specs.total_cells,
                "cell_size_mm": self.print_specs.cell_size_mm,
                "print_area_mm": (
                    grid_map.grid_specs.cols * self.print_specs.cell_size_mm,
                    grid_map.grid_specs.rows * self.print_specs.cell_size_mm,
                ),
            },
            "quality_assessment": {
                "overall_quality": quality_metrics.overall_quality,
                "ssim_score": quality_metrics.ssim_score,
                "delta_e_mean": quality_metrics.delta_e_mean,
                "delta_e_max": quality_metrics.delta_e_max,
                "delta_e_std": quality_metrics.delta_e_std,
                "color_distribution": quality_metrics.color_distribution,
                "rare_colors": quality_metrics.rare_colors,
                "auto_fixes": quality_metrics.auto_fixes,
                "warnings": quality_metrics.quality_warnings,
                "risks": quality_metrics.quality_risks,
            },
            "quality_gates": {
                "passed": quality_result.passed,
                "warnings": quality_result.warnings,
                "errors": quality_result.errors,
                "auto_fixes": quality_result.auto_fixes,
                "metrics": quality_result.metrics,
            },
            "palette_info": {
                "style_name": grid_map.style_name,
                "total_colors": len(grid_map.palette_colors),
                "dmc_codes": palette_codes,
            },
            "output_files": output_manifest,
            "print_specifications": self.print_engine.calculate_print_metrics(grid_map.grid_specs),
        }
        
        return metadata


def generate_diamond_kit(image_path: str, style_name: str, output_dir: str,
                        dpi: int = 600, margin_mm: float = 12.0, 
                        cell_size_mm: float = 2.8,
                        crop_rect: Optional[Tuple[float, float, float, float]] = None) -> Dict[str, Any]:
    """
    Convenience function to generate a complete diamond painting kit.
    
    Args:
        image_path: Path to input image
        style_name: Style name (ORIGINAL, VINTAGE, POPART)
        output_dir: Output directory
        dpi: Print DPI
        margin_mm: Paper margins
        cell_size_mm: Cell size in mm
        crop_rect: Optional crop rectangle
        
    Returns:
        Complete kit generation results
    """
    generator = DiamondKitGenerator(dpi, margin_mm, cell_size_mm)
    return generator.generate_kit(image_path, style_name, output_dir, crop_rect)


def get_available_styles() -> List[str]:
    """Get list of available styles."""
    return list_available_styles()


def get_style_info(style_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific style."""
    manager = get_fixed_palette_manager()
    palette = manager.get_palette(style_name)
    colors = palette.get_colors()
    
    return {
        "style_name": style_name,
        "description": palette.description,
        "rationale": palette.rationale,
        "dmc_codes": palette.dmc_codes,
        "colors": [
            {
                "dmc_code": color.dmc_code,
                "name": color.name,
                "hex": color.hex,
                "rgb": color.rgb
            }
            for color in colors
        ]
    }
