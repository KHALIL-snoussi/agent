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
from skimage.color import rgb2lab

from .fixed_palettes import get_fixed_palette_manager, list_available_styles
from .print_math import get_print_math_engine, PrintSpecs, GridSpecs
from .grid_index_map import create_grid_index_map, verify_grid_invariance
from .quality_assessor import assess_quality, generate_quality_report
from .dmc import DMCColor


class DiamondKitGenerator:
    """Commercial-grade diamond painting kit generator with fixed palettes."""
    
    def __init__(self, dpi: int = 600, margin_mm: float = 12.0, cell_size_mm: float = 2.8):
        """
        Initialize kit generator with print specifications.
        
        Args:
            dpi: Print DPI (≥300 required, default 600 for quality)
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
        
        # Load and validate image
        original_rgb, original_lab = self._load_and_validate_image(image_path)
        
        # Apply smart cropping if needed
        cropped_rgb = self._apply_smart_cropping(original_rgb, crop_rect)
        cropped_lab = rgb2lab(cropped_rgb / 255.0)
        
        # Calculate grid specifications with 10k cap enforcement
        grid_specs, scale_factor = self.print_engine.calculate_grid_from_image(
            cropped_rgb.shape[1], cropped_rgb.shape[0]
        )
        
        # Create fixed GRID_INDEX_MAP
        grid_map = create_grid_index_map(
            cropped_lab, grid_specs, style_name, enable_smoothing=True
        )
        
        # Generate quantized RGB from grid map
        quantized_rgb = self._grid_to_rgb_visualization(grid_map)
        
        # Resize cropped RGB to match grid dimensions for SSIM comparison
        resized_for_ssim = self._resize_image_for_comparison(cropped_rgb, grid_map.grid_specs)
        resized_lab_for_ssim = rgb2lab(resized_for_ssim / 255.0)
        
        # Assess quality
        quality_metrics = assess_quality(
            resized_for_ssim, quantized_rgb, grid_map, resized_lab_for_ssim
        )
        
        # Generate all output artifacts
        kit_outputs = self._generate_all_outputs(
            cropped_rgb, quantized_rgb, grid_map, quality_metrics, 
            style_name, output_dir, scale_factor
        )
        
        # Compile comprehensive metadata
        metadata = self._compile_comprehensive_metadata(
            grid_map, quality_metrics, scale_factor, crop_rect, kit_outputs
        )
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "kit_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Kit generation complete! Output in: {output_dir}")
        print(f"Grid: {grid_specs.cols}×{grid_specs.rows} ({grid_specs.total_cells:,} cells)")
        print(f"Quality: {quality_metrics.overall_quality}")
        print(f"ΔE: mean={quality_metrics.delta_e_mean:.1f}, max={quality_metrics.delta_e_max:.1f}")
        
        return {
            "metadata": metadata,
            "outputs": kit_outputs,
            "quality_report": generate_quality_report(quality_metrics),
            "grid_map": grid_map,
            "grid_specs": grid_specs,
            "scale_factor": scale_factor
        }
    
    def _load_and_validate_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and validate input image."""
        print(f"Loading image: {image_path}")
        
        # Load image using PIL directly (simplest approach)
        pil_image = Image.open(image_path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        original_rgb = np.array(pil_image)
        
        # Validate image properties
        h, w = original_rgb.shape[:2]
        
        if h < 800 or w < 800:
            print(f"Warning: Small image ({w}×{h}). Consider higher resolution for better results.")
        
        # Check for sufficient contrast/entropy
        gray = np.mean(original_rgb, axis=2)
        entropy = -np.sum((np.histogram(gray, 256)[0] / gray.size) * 
                        np.log2(np.histogram(gray, 256)[0] / gray.size + 1e-10))
        
        if entropy < 3.0:
            print(f"Warning: Low image entropy ({entropy:.2f}). May result in poor quantization.")
        
        # Convert to Lab
        original_lab = rgb2lab(original_rgb / 255.0)
        
        return original_rgb, original_lab
    
    def _apply_smart_cropping(self, image_rgb: np.ndarray, 
                           crop_rect: Optional[Tuple[float, float, float, float]]) -> np.ndarray:
        """Apply smart cropping or use provided crop rectangle."""
        h, w = image_rgb.shape[:2]
        
        if crop_rect is not None:
            # Use provided crop rectangle
            x, y, crop_w, crop_h = crop_rect
            
            # Convert normalized to pixel coordinates
            x_px = int(x * w)
            y_px = int(y * h)
            w_px = int(crop_w * w)
            h_px = int(crop_h * h)
            
            # Ensure bounds
            x_px = max(0, min(x_px, w - 1))
            y_px = max(0, min(y_px, h - 1))
            w_px = min(w_px, w - x_px)
            h_px = min(h_px, h - y_px)
            
            return image_rgb[y_px:y_px+h_px, x_px:x_px+w_px]
        
        else:
            # Auto-suggest crop (implement simple center crop for now)
            # Could be enhanced with saliency detection
            target_aspect = 4/3  # Standard aspect ratio
            
            current_aspect = w / h
            if abs(current_aspect - target_aspect) > 0.2:
                # Crop to match target aspect
                if current_aspect > target_aspect:
                    # Too wide - crop width
                    new_w = int(h * target_aspect)
                    x_start = (w - new_w) // 2
                    return image_rgb[:, x_start:x_start+new_w]
                else:
                    # Too tall - crop height
                    new_h = int(w / target_aspect)
                    y_start = (h - new_h) // 2
                    return image_rgb[y_start:y_start+new_h, :]
            
            return image_rgb
    
    def _resize_image_for_comparison(self, image_rgb: np.ndarray, grid_specs) -> np.ndarray:
        """Resize image to match grid dimensions for SSIM comparison."""
        from PIL import Image
        
        h, w = image_rgb.shape[:2]
        
        if w == grid_specs.cols and h == grid_specs.rows:
            return image_rgb
        
        pil_image = Image.fromarray(image_rgb)
        resized_image = pil_image.resize((grid_specs.cols, grid_specs.rows), Image.LANCZOS)
        return np.array(resized_image)
    
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
    
    def _generate_all_outputs(self, original_rgb: np.ndarray, quantized_rgb: np.ndarray,
                           grid_map, quality_metrics,
                           style_name: str, output_dir: str, 
                           scale_factor: float) -> Dict[str, str]:
        """Generate all required output artifacts."""
        outputs = {}
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate preview images
        outputs["original_preview"] = self._save_preview_image(
            original_rgb, output_dir, "original_preview.jpg"
        )
        
        outputs["quantized_preview"] = self._save_preview_image(
            quantized_rgb, output_dir, "quantized_preview.jpg"
        )
        
        # Generate style previews (all styles using same grid map)
        for style in list_available_styles():
            style_preview = self._generate_style_preview(grid_map, style)
            outputs[f"preview_{style.lower()}"] = self._save_preview_image(
                style_preview, output_dir, f"preview_{style.lower()}.jpg"
            )
        
        # Generate CSV inventory
        outputs["csv_inventory"] = self._generate_csv_inventory(
            grid_map, quality_metrics, output_dir
        )
        
        # Generate JSON metadata
        outputs["json_metadata"] = self._generate_json_metadata(
            grid_map, quality_metrics, scale_factor, output_dir
        )
        
        # Generate PDF kit (placeholder - will be implemented in pdf.py)
        outputs["pdf_kit"] = self._generate_pdf_kit(
            grid_map, original_rgb, output_dir, style_name
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
        """Generate style preview overlay (maintains same grid assignments)."""
        base_rgb = self._grid_to_rgb_visualization(grid_map)
        
        # Apply style-specific overlays (without changing grid assignments)
        if style_name == "ORIGINAL":
            # Mild contrast enhancement
            from skimage.exposure import adjust_gamma
            enhanced = adjust_gamma(base_rgb / 255.0, gamma=1.1)
            return np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        
        elif style_name == "VINTAGE":
            # Sepia/aging overlay
            sepia_filter = np.array([[0.393, 0.769, 0.189],
                                   [0.349, 0.686, 0.168],
                                   [0.272, 0.534, 0.131]])
            vintage = base_rgb @ sepia_filter.T
            vintage = np.clip(vintage, 0, 255).astype(np.uint8)
            return vintage
        
        elif style_name == "POPART":
            # High contrast with edge enhancement
            from skimage.filters import sobel
            edges = sobel(base_rgb.astype(float))
            edge_enhanced = base_rgb + (edges * 20).astype(np.uint8)
            return np.clip(edge_enhanced, 0, 255).astype(np.uint8)
        
        return base_rgb
    
    def _generate_csv_inventory(self, grid_map, quality_metrics, output_dir: str) -> str:
        """Generate exact CSV inventory format as specified."""
        csv_path = os.path.join(output_dir, "inventory.csv")
        
        # Count drill requirements
        h, w = grid_map.grid_data.shape
        unique_indices, counts = np.unique(grid_map.grid_data, return_counts=True)
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'dmc_code', 'name', 'hex', 'cluster_id', 'drill_count', 
                'bag_qty_200pcs', 'deltaE_mean', 'deltaE_max'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for idx, count in zip(unique_indices, counts):
                if idx < len(grid_map.palette_colors):
                    color = grid_map.palette_colors[idx]
                    
                    # Calculate bag quantity (200 drills per bag)
                    bag_qty = -(-count // 200)  # Ceiling division
                    
                    writer.writerow({
                        'dmc_code': color.dmc_code,
                        'name': color.name,
                        'hex': color.hex,
                        'cluster_id': idx,  # 0-6 matching fixed palette order
                        'drill_count': int(count),
                        'bag_qty_200pcs': bag_qty,
                        'deltaE_mean': round(quality_metrics.delta_e_mean, 2),
                        'deltaE_max': round(quality_metrics.delta_e_max, 2)
                    })
        
        return csv_path
    
    def _generate_json_metadata(self, grid_map, quality_metrics,
                             scale_factor: float, output_dir: str) -> str:
        """Generate exact JSON metadata format as specified."""
        json_path = os.path.join(output_dir, "metadata.json")
        
        # Count color usage for metadata
        h, w = grid_map.grid_data.shape
        unique_indices, counts = np.unique(grid_map.grid_data, return_counts=True)
        
        color_usage = {}
        for idx, count in zip(unique_indices, counts):
            if idx < len(grid_map.palette_colors):
                color = grid_map.palette_colors[idx]
                color_usage[color.dmc_code] = int(count)
        
        metadata = {
            "paper_mm": [self.print_specs.paper_width_mm, self.print_specs.paper_height_mm],
            "dpi": self.print_specs.dpi,
            "margins_mm": self.print_specs.margin_mm,
            "cell_mm": self.print_specs.cell_size_mm,
            "grid_cols": grid_map.grid_specs.cols,
            "grid_rows": grid_map.grid_specs.rows,
            "total_cells": grid_map.grid_specs.total_cells,
            "pages": len(self.print_engine.calculate_tiling(grid_map.grid_specs)),
            "style": grid_map.style_name,
            "fixed_palette": True,
            "fixed_palette_dmc": [color.dmc_code for color in grid_map.palette_colors],
            "deltaE_stats": {
                "mean": round(quality_metrics.delta_e_mean, 2),
                "max": round(quality_metrics.delta_e_max, 2),
                "std": round(quality_metrics.delta_e_std, 2)
            },
            "ssim": round(quality_metrics.ssim_score, 4),
            "crop_rect_norm": None,  # Could be populated from smart cropping
            "tiling_map": self._generate_tiling_map(grid_map.grid_specs),
            "grid_index_map_hash": grid_map.grid_hash,
            "color_usage": color_usage,
            "scale_factor": round(scale_factor, 3),
            "print_metrics": self.print_engine.calculate_print_metrics(grid_map.grid_specs)
        }
        
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return json_path
    
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
    
    def _generate_pdf_kit(self, grid_map, original_rgb: np.ndarray,
                         output_dir: str, style_name: str) -> str:
        """Generate PDF kit using the diamondkit PDF generator."""
        try:
            from .pdf import PDFGenerator
            from .config import Config
            
            # Create a basic config for PDF generation
            config = Config()
            config.export.page = "A4"
            config.export.pdf_dpi = 300
            
            # Generate PDF
            pdf_generator = PDFGenerator(config)
            pdf_path = os.path.join(output_dir, "diamond_painting_kit.pdf")
            
            # Create a simple metadata dict for PDF generation
            metadata = {
                'filename': os.path.basename(output_dir),
                'grid_size': f"{grid_map.grid_specs.cols}×{grid_map.grid_specs.rows}",
                'total_cells': grid_map.grid_specs.total_cells,
                'colors_used': len(grid_map.palette_colors),
                'style': style_name
            }
            
            # Generate quantized RGB from grid map
            quantized_rgb = self._grid_to_rgb_visualization(grid_map)
            
            # Generate the PDF
            pdf_generator.generate_complete_pdf(grid_map, quantized_rgb, metadata, pdf_path)
            
            print(f"PDF generated successfully: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            print(f"Warning: PDF generation failed: {e}")
            # Fallback to placeholder
            pdf_path = os.path.join(output_dir, "diamond_painting_kit.pdf")
            with open(pdf_path, 'w') as f:
                f.write("PDF generation failed. Please check console for errors.")
            return pdf_path
    
    def _compile_comprehensive_metadata(self, grid_map, quality_metrics,
                                  scale_factor: float, crop_rect, outputs: Dict) -> Dict[str, Any]:
        """Compile comprehensive metadata for kit."""
        return {
            "generation_info": {
                "style": grid_map.style_name,
                "grid_hash": grid_map.grid_hash,
                "scale_factor": scale_factor,
                "crop_applied": crop_rect is not None
            },
            "grid_specifications": {
                "dimensions": f"{grid_map.grid_specs.cols}×{grid_map.grid_specs.rows}",
                "total_cells": grid_map.grid_specs.total_cells,
                "cell_size_mm": self.print_specs.cell_size_mm,
                "print_area_mm": (
                    grid_map.grid_specs.cols * self.print_specs.cell_size_mm,
                    grid_map.grid_specs.rows * self.print_specs.cell_size_mm
                )
            },
            "quality_assessment": {
                "overall_quality": quality_metrics.overall_quality,
                "ssim_score": quality_metrics.ssim_score,
                "delta_e_mean": quality_metrics.delta_e_mean,
                "delta_e_max": quality_metrics.delta_e_max,
                "warnings": quality_metrics.quality_warnings,
                "risks": quality_metrics.quality_risks
            },
            "palette_info": {
                "style_name": grid_map.style_name,
                "total_colors": len(grid_map.palette_colors),
                "dmc_codes": [color.dmc_code for color in grid_map.palette_colors],
                "color_distribution": quality_metrics.color_distribution,
                "rare_colors": quality_metrics.rare_colors
            },
            "output_files": {
                key: os.path.basename(path) for key, path in outputs.items()
            },
            "print_specifications": {
                "paper_size": "A4",
                "dpi": self.print_specs.dpi,
                "margins_mm": self.print_specs.margin_mm,
                "total_pages": len(self.print_engine.calculate_tiling(grid_map.grid_specs))
            }
        }


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
