"""
Export manager for diamond painting kit files (PDF, CSV, JSON).
"""

import os
import csv
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import uuid

from .config import Config
from .grid import CanvasGrid
from .pdf import QBRIXPDFGenerator as PDFGenerator


class ExportManager:
    """Manages export of all diamond painting kit files."""
    
    def __init__(self, config: Config):
        """Initialize export manager with configuration."""
        self.config = config
        self.output_dir = config.output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.pdf_generator = PDFGenerator(config)
    
    def export_complete_kit(self, canvas_grid: CanvasGrid, 
                          preview_image: np.ndarray,
                          metadata: Dict[str, Any]) -> bool:
        """
        Export complete diamond painting kit with all files.
        
        Args:
            canvas_grid: Canvas grid with symbol assignments
            preview_image: Preview image as numpy array
            metadata: Image processing metadata
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            print("Exporting complete diamond painting kit...")
            
            # Generate unique kit ID
            kit_id = str(uuid.uuid4())[:8]
            
            # Export all components
            self._export_pdf(canvas_grid, preview_image, metadata, kit_id)
            self._export_csv_legend(canvas_grid, kit_id)
            self._export_json_manifest(canvas_grid, metadata, kit_id)
            self._export_preview(preview_image, kit_id)
            
            print(f"[OK] Kit exported successfully to {self.output_dir}")
            print(f"  Kit ID: {kit_id}")
            return True
            
        except Exception as e:
            print(f"[X] Export failed: {e}")
            return False
    
    def _export_pdf(self, canvas_grid: CanvasGrid, 
                   preview_image: np.ndarray,
                   metadata: Dict[str, Any],
                   kit_id: str):
        """Export PDF instruction booklet."""
        pdf_path = os.path.join(self.output_dir, f"diamond_kit_{kit_id}.pdf")
        
        self.pdf_generator.generate_complete_pdf(
            canvas_grid, preview_image, metadata, pdf_path
        )
        
        print(f"  PDF: {os.path.basename(pdf_path)}")
    
    def _export_csv_legend(self, canvas_grid: CanvasGrid, kit_id: str):
        """Export CSV legend with drill counts and bag information."""
        csv_path = os.path.join(self.output_dir, f"diamond_kit_{kit_id}_legend.csv")
        
        legend_items = canvas_grid.get_color_legend()
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'symbol', 'dmc_code', 'color_name', 'hex_color',
                'drill_count', 'percentage', 'bags_needed', 'spare_drills'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for item in legend_items:
                count = item['count']
                spare_drills = int(count * self.config.export.spare_ratio)
                
                writer.writerow({
                    'symbol': item['symbol'],
                    'dmc_code': item['dmc_code'],
                    'color_name': item['name'],
                    'hex_color': item['hex'],
                    'drill_count': count,
                    'percentage': f"{item['percentage']:.2f}%",
                    'bags_needed': item['bags_needed'],
                    'spare_drills': spare_drills
                })
        
        print(f"  CSV: {os.path.basename(csv_path)}")
    
    def _export_json_manifest(self, canvas_grid: CanvasGrid,
                            metadata: Dict[str, Any],
                            kit_id: str):
        """Export JSON manifest with all kit parameters."""
        json_path = os.path.join(self.output_dir, f"diamond_kit_{kit_id}_manifest.json")
        
        # Build comprehensive manifest
        manifest = {
            'kit_info': {
                'kit_id': kit_id,
                'generated_at': datetime.now().isoformat(),
                'generator_version': '1.0.0',
                'input_image': {
                    'filename': metadata.get('filename', 'unknown'),
                    'original_size': metadata.get('original_size', []),
                    'processed_size': metadata.get('processed_size', [])
                }
            },
            'canvas': {
                'width_cm': self.config.canvas.width_cm,
                'height_cm': self.config.canvas.height_cm,
                'drill_shape': self.config.canvas.drill_shape,
                'drill_size_mm': self.config.canvas.drill_size_mm,
                'cells_w': canvas_grid.cells_w,
                'cells_h': canvas_grid.cells_h,
                'total_cells': canvas_grid.total_cells
            },
            'color_palette': {
                'mode': self.config.palette.mode,
                'max_colors': self.config.palette.max_colors,
                'preserve_skin_tones': self.config.palette.preserve_skin_tones,
                'actual_colors_used': len(canvas_grid.dmc_colors),
                'colors': [
                    {
                        'dmc_code': color.dmc_code,
                        'name': color.name,
                        'rgb': color.rgb,
                        'hex': color.hex,
                        'symbol': canvas_grid.color_to_symbol.get(color.dmc_code, '?')
                    }
                    for color in canvas_grid.dmc_colors
                ]
            },
            'processing': {
                'dither_mode': self.config.dither.mode,
                'dither_strength': self.config.dither.strength,
                'seed': self.config.processing.seed,
                'color_space': self.config.processing.color_space,
                'quantization_method': self.config.processing.quantization_method
            },
            'export': {
                'page_size': self.config.export.page,
                'pdf_dpi': self.config.export.pdf_dpi,
                'spare_ratio': self.config.export.spare_ratio,
                'bag_size': self.config.export.bag_size,
                'tiling_overlap_mm': self.config.export.overlap_mm
            },
            'drill_statistics': {
                'total_drills': canvas_grid._get_total_drills(),
                'unique_colors': len(canvas_grid.dmc_colors),
                'color_breakdown': canvas_grid.color_counts
            },
            'quality_metrics': {
                'estimated_difficulty': self._calculate_difficulty(canvas_grid),
                'complexity_score': self._calculate_complexity(canvas_grid),
                'recommended_experience': self._get_experience_level(canvas_grid)
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(manifest, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"  JSON: {os.path.basename(json_path)}")
    
    def _export_preview(self, preview_image: np.ndarray, kit_id: str):
        """Export preview image."""
        preview_path = os.path.join(self.output_dir, f"diamond_kit_{kit_id}_preview.jpg")
        
        from PIL import Image
        pil_image = Image.fromarray(preview_image)
        pil_image.save(preview_path, quality=90, optimize=True)
        
        print(f"  Preview: {os.path.basename(preview_path)}")
    
    def _calculate_difficulty(self, canvas_grid: CanvasGrid) -> str:
        """Calculate estimated difficulty level."""
        total_cells = canvas_grid.total_cells
        color_count = len(canvas_grid.dmc_colors)
        
        # Simple difficulty algorithm
        if total_cells < 2000 and color_count < 15:
            return "Beginner"
        elif total_cells < 5000 and color_count < 25:
            return "Easy"
        elif total_cells < 10000 and color_count < 40:
            return "Medium"
        elif total_cells < 20000:
            return "Challenging"
        else:
            return "Expert"
    
    def _calculate_complexity(self, canvas_grid: CanvasGrid) -> float:
        """Calculate complexity score (0-100)."""
        total_cells = canvas_grid.total_cells
        color_count = len(canvas_grid.dmc_colors)
        
        # Complexity based on cell count and color diversity
        cell_complexity = min(total_cells / 500, 50)  # Max 50 points from size
        color_complexity = min(color_count * 2, 50)    # Max 50 points from colors
        
        return cell_complexity + color_complexity
    
    def _get_experience_level(self, canvas_grid: CanvasGrid) -> str:
        """Get recommended experience level."""
        difficulty = self._calculate_difficulty(canvas_grid)
        
        recommendations = {
            "Beginner": "Perfect for first-time diamond painters",
            "Easy": "Great for beginners with some experience",
            "Medium": "Suitable for intermediate diamond painters",
            "Challenging": "Recommended for experienced crafters",
            "Expert": "For advanced diamond painting enthusiasts"
        }
        
        return recommendations.get(difficulty, "All skill levels")
    
    def create_shopping_list(self, canvas_grid: CanvasGrid, 
                           kit_id: str) -> str:
        """Create shopping list for DMC drills."""
        shopping_path = os.path.join(self.output_dir, f"diamond_kit_{kit_id}_shopping.txt")
        
        legend_items = canvas_grid.get_color_legend()
        
        with open(shopping_path, 'w', encoding='utf-8') as f:
            f.write("DIAMOND PAINTING KIT - SHOPPING LIST\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Kit ID: {kit_id}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            f.write("DMC DRILLS REQUIRED:\n")
            f.write("-" * 30 + "\n")
            
            total_bags = 0
            for item in legend_items:
                bags = item['bags_needed']
                total_bags += bags
                
                f.write(f"{item['dmc_code']:>6} - {item['name']:<25} ({bags:2d} bags)\n")
            
            f.write("-" * 30 + "\n")
            f.write(f"Total bags required: {total_bags}\n\n")
            
            f.write("NOTES:\n")
            f.write("- Each bag contains {self.config.export.bag_size} drills\n")
            f.write(f"- Includes {self.config.export.spare_ratio:.0%} spare drills\n")
            f.write("- Drill shape: {self.config.canvas.drill_shape}\n")
            f.write(f"- Drill size: {self.config.canvas.drill_size_mm}mm\n")
        
        return shopping_path
    
    def export_grid_data(self, canvas_grid: CanvasGrid, kit_id: str):
        """Export raw grid data for external processing."""
        grid_path = os.path.join(self.output_dir, f"diamond_kit_{kit_id}_grid.json")
        
        if canvas_grid.grid_data is None:
            print("  [WARN] No grid data available for export")
            return
        
        # Convert grid to more serializable format
        grid_serialized = {
            'dimensions': {
                'width': canvas_grid.cells_w,
                'height': canvas_grid.cells_h
            },
            'symbols': canvas_grid.color_to_symbol,
            'grid': canvas_grid.grid_data.tolist(),
            'color_mapping': {
                str(idx): {
                    'dmc_code': color.dmc_code,
                    'symbol': canvas_grid.color_to_symbol.get(color.dmc_code, '?'),
                    'hex': color.hex
                }
                for idx, color in enumerate(canvas_grid.dmc_colors)
            }
        }
        
        with open(grid_path, 'w', encoding='utf-8') as f:
            json.dump(grid_serialized, f, indent=2)
        
        print(f"  Grid: {os.path.basename(grid_path)}")
    
    def get_export_summary(self, canvas_grid: CanvasGrid) -> Dict[str, Any]:
        """Get summary of exported files and statistics."""
        return {
            'output_directory': self.output_dir,
            'expected_files': [
                'diamond_kit_{kit_id}.pdf',
                'diamond_kit_{kit_id}_legend.csv',
                'diamond_kit_{kit_id}_manifest.json',
                'diamond_kit_{kit_id}_preview.jpg'
            ],
            'statistics': {
                'total_cells': canvas_grid.total_cells,
                'colors_used': len(canvas_grid.dmc_colors),
                'total_drills': canvas_grid._get_total_drills(),
                'total_bags': sum(item['bags_needed'] for item in canvas_grid.get_color_legend()),
                'difficulty': self._calculate_difficulty(canvas_grid),
                'complexity': self._calculate_complexity(canvas_grid)
            }
        }
