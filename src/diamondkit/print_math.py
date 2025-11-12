"""
A4 print mathematics engine for diamond painting kit generation.
Handles cell sizing, grid constraints, tiling, and print compliance.
"""

import math
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class PrintSpecs:
    """Print specifications for A4 diamond painting kits."""
    # Paper specs
    paper_width_mm: float = 210.0  # A4 width
    paper_height_mm: float = 297.0  # A4 height
    dpi: int = 600  # High DPI for print clarity (≥300 required)
    
    # Margins (within 10-15mm range)
    margin_mm: float = 12.0  # Chosen within 10-15mm range
    
    # Cell sizing (within 2.3-3.0mm range)
    cell_size_mm: float = 2.8  # Chosen within 2.3-3.0mm range
    
    # Grid constraints
    max_total_cells: int = 10000  # Hard constraint
    
    # Tiling specs
    overlap_cells: int = 2  # 2-cell overlap for tiling
    
    def __post_init__(self):
        """Validate print specifications."""
        if not (10 <= self.margin_mm <= 15):
            raise ValueError(f"Margin {self.margin_mm}mm outside 10-15mm range")
        
        if not (2.3 <= self.cell_size_mm <= 3.0):
            raise ValueError(f"Cell size {self.cell_size_mm}mm outside 2.3-3.0mm range")
        
        if self.dpi < 300:
            raise ValueError(f"DPI {self.dpi} below minimum 300")
    
    @property
    def usable_width_mm(self) -> float:
        """Calculate usable width after margins."""
        return self.paper_width_mm - 2 * self.margin_mm
    
    @property
    def usable_height_mm(self) -> float:
        """Calculate usable height after margins."""
        return self.paper_height_mm - 2 * self.margin_mm
    
    @property
    def cells_per_page_width(self) -> int:
        """Calculate how many cells fit per page width."""
        return int(self.usable_width_mm / self.cell_size_mm)
    
    @property
    def cells_per_page_height(self) -> int:
        """Calculate how many cells fit per page height."""
        return int(self.usable_height_mm / self.cell_size_mm)


@dataclass
class GridSpecs:
    """Grid specifications for the diamond painting."""
    cols: int
    rows: int
    total_cells: int
    
    def __post_init__(self):
        """Validate grid specifications."""
        if self.cols <= 0 or self.rows <= 0:
            raise ValueError("Grid dimensions must be positive")
        
        if self.total_cells != self.cols * self.rows:
            raise ValueError("Total cells doesn't match cols * rows")
        
        if self.total_cells > 10000:
            raise ValueError(f"Grid {self.total_cells} cells exceeds 10k limit")


@dataclass
class TileInfo:
    """Information about a single tile in the tiling layout."""
    x_start: int
    y_start: int
    tile_cols: int
    tile_rows: int
    page_number: int
    total_pages: int
    crop_marks: List[Tuple[float, float, float, float]]  # (x1, y1, x2, y2)
    registration_crosses: List[Tuple[float, float]]  # (x, y) positions
    
    @property
    def coordinates(self) -> str:
        """Get coordinate string for tile."""
        return f"({self.x_start},{self.y_start})-({self.x_start + self.tile_cols},{self.y_start + self.tile_rows})"


class PrintMathEngine:
    """Engine for calculating A4 print layout and constraints."""
    
    def __init__(self, specs: Optional[PrintSpecs] = None):
        """Initialize engine with print specifications."""
        self.specs = specs or PrintSpecs()
    
    def calculate_grid_from_image(self, image_width: int, image_height: int) -> Tuple[GridSpecs, float]:
        """
        Calculate optimal grid dimensions from image size.
        
        Args:
            image_width: Input image width in pixels
            image_height: Input image height in pixels
            
        Returns:
            Tuple of (grid_specs, scale_factor)
        """
        # Start with 1:1 pixel to cell mapping
        initial_cols = image_width
        initial_rows = image_height
        initial_total = initial_cols * initial_rows
        
        if initial_total <= self.specs.max_total_cells:
            # No scaling needed
            return GridSpecs(initial_cols, initial_rows, initial_total), 1.0
        
        # Calculate scaling factor to fit within 10k limit
        scale_factor = math.sqrt(self.specs.max_total_cells / initial_total)
        
        # Apply scaling and round to integers
        scaled_cols = max(1, int(round(initial_cols * scale_factor)))
        scaled_rows = max(1, int(round(initial_rows * scale_factor)))
        scaled_total = scaled_cols * scaled_rows
        
        # Ensure we're within limits (adjust if rounding pushed us over)
        while scaled_total > self.specs.max_total_cells:
            if scaled_cols > scaled_rows:
                scaled_cols -= 1
            else:
                scaled_rows -= 1
            scaled_total = scaled_cols * scaled_rows
        
        return GridSpecs(scaled_cols, scaled_rows, scaled_total), scale_factor
    
    def calculate_tiling(self, grid: GridSpecs) -> List[TileInfo]:
        """
        Calculate tiling layout for multi-page PDF generation.
        
        Args:
            grid: Grid specifications
            
        Returns:
            List of tile information
        """
        tiles = []
        page_num = 1
        
        cells_per_page_w = self.specs.cells_per_page_width
        cells_per_page_h = self.specs.cells_per_page_height
        overlap = self.specs.overlap_cells
        
        y_start = 0
        while y_start < grid.rows:
            x_start = 0
            
            while x_start < grid.cols:
                # Calculate tile dimensions
                tile_cols = min(cells_per_page_w, grid.cols - x_start)
                tile_rows = min(cells_per_page_h, grid.rows - y_start)
                
                # Create tile info
                tile = TileInfo(
                    x_start=x_start,
                    y_start=y_start,
                    tile_cols=tile_cols,
                    tile_rows=tile_rows,
                    page_number=page_num,
                    total_pages=0,  # Will be calculated after all tiles are created
                    crop_marks=self._calculate_crop_marks(tile_cols, tile_rows),
                    registration_crosses=self._calculate_registration_crosses(tile_cols, tile_rows)
                )
                
                tiles.append(tile)
                
                # Move to next tile position
                x_start += cells_per_page_w - overlap
                if x_start >= grid.cols:
                    break
                
                page_num += 1
            
            # Move to next row of tiles
            y_start += cells_per_page_h - overlap
            page_num += 1
        
        # Update total pages for all tiles
        total_pages = len(tiles)
        for tile in tiles:
            tile.total_pages = total_pages
        
        return tiles
    
    def _calculate_crop_marks(self, tile_cols: int, tile_rows: int) -> List[Tuple[float, float, float, float]]:
        """Calculate crop mark positions for a tile."""
        crop_length_mm = 10.0  # 10mm crop mark length
        crop_length_pt = self._mm_to_points(crop_length_mm)
        
        tile_width_mm = tile_cols * self.specs.cell_size_mm
        tile_height_mm = tile_rows * self.specs.cell_size_mm
        tile_width_pt = self._mm_to_points(tile_width_mm)
        tile_height_pt = self._mm_to_points(tile_height_mm)
        
        marks = []
        
        # Top-left corner
        marks.append((-crop_length_pt, 0, 0, 0))  # Left
        marks.append((0, 0, 0, crop_length_pt))   # Top
        
        # Top-right corner
        marks.append((tile_width_pt, 0, tile_width_pt + crop_length_pt, 0))  # Right
        marks.append((tile_width_pt - crop_length_pt, 0, tile_width_pt, crop_length_pt))  # Top
        
        # Bottom-left corner
        marks.append((-crop_length_pt, tile_height_pt, 0, tile_height_pt))  # Left
        marks.append((0, tile_height_pt - crop_length_pt, 0, tile_height_pt))  # Bottom
        
        # Bottom-right corner
        marks.append((tile_width_pt, tile_height_pt, tile_width_pt + crop_length_pt, tile_height_pt))  # Right
        marks.append((tile_width_pt - crop_length_pt, tile_height_pt - crop_length_pt, tile_width_pt, tile_height_pt))  # Bottom
        
        return marks
    
    def _calculate_registration_crosses(self, tile_cols: int, tile_rows: int) -> List[Tuple[float, float]]:
        """Calculate registration cross positions for a tile."""
        cross_size_mm = 5.0  # 5mm cross size
        tile_width_mm = tile_cols * self.specs.cell_size_mm
        tile_height_mm = tile_rows * self.specs.cell_size_mm
        
        # Position crosses at 1/4 and 3/4 positions
        crosses = []
        
        # Top quadrant
        crosses.append((tile_width_mm * 0.25, tile_height_mm * 0.25))
        # Bottom quadrant
        crosses.append((tile_width_mm * 0.75, tile_height_mm * 0.75))
        
        return crosses
    
    def _mm_to_points(self, mm: float) -> float:
        """Convert millimeters to points (72 points per inch)."""
        inches = mm / 25.4
        return inches * 72.0
    
    def calculate_cell_size_for_symbols(self) -> float:
        """
        Calculate optimal cell size for symbol legibility.
        
        Returns:
            Recommended cell size in mm that meets 1.2mm x-height requirement
        """
        # For bold digits 1-7, we need approximately 43% of cell height for x-height
        # To achieve 1.2mm x-height: cell_size >= 1.2mm / 0.43 = 2.79mm
        min_for_legibility = 1.2 / 0.43
        
        # Return the larger of our default or minimum for legibility
        return max(self.specs.cell_size_mm, min_for_legibility)
    
    def calculate_print_metrics(self, grid: GridSpecs) -> Dict[str, Any]:
        """Calculate comprehensive print metrics for reporting."""
        tiles = self.calculate_tiling(grid)
        
        return {
            "paper_size_mm": (self.specs.paper_width_mm, self.specs.paper_height_mm),
            "dpi": self.specs.dpi,
            "margins_mm": self.specs.margin_mm,
            "cell_size_mm": self.specs.cell_size_mm,
            "grid_dimensions": (grid.cols, grid.rows),
            "total_cells": grid.total_cells,
            "total_pages": len(tiles),
            "tiles_per_page_width": self.specs.cells_per_page_width,
            "tiles_per_page_height": self.specs.cells_per_page_height,
            "overlap_cells": self.specs.overlap_cells,
            "print_area_mm": (
                grid.cols * self.specs.cell_size_mm,
                grid.rows * self.specs.cell_size_mm
            ),
            "aspect_ratio": grid.cols / grid.rows,
            "meets_constraints": {
                "cell_size_in_range": 2.3 <= self.specs.cell_size_mm <= 3.0,
                "margins_in_range": 10 <= self.specs.margin_mm <= 15,
                "under_10k_limit": grid.total_cells <= 10000,
                "dpi_sufficient": self.specs.dpi >= 300,
                "symbols_legible": self.specs.cell_size_mm >= 2.79
            }
        }
    
    def validate_aspect_ratio_crop(self, original_aspect: float, target_aspect: float, 
                                 tolerance: float = 0.02) -> bool:
        """
        Validate if crop aspect ratio is within tolerance.
        
        Args:
            original_aspect: Original image aspect ratio
            target_aspect: Target grid aspect ratio
            tolerance: Allowed deviation (default 2% = 0.02)
            
        Returns:
            True if within tolerance
        """
        ratio = original_aspect / target_aspect
        return (1.0 - tolerance) <= ratio <= (1.0 + tolerance)
    
    def suggest_cell_size(self, grid_cols: int, grid_rows: int, 
                         target_pages: Optional[int] = None) -> float:
        """
        Suggest optimal cell size based on grid dimensions and target pages.
        
        Args:
            grid_cols: Number of grid columns
            grid_rows: Number of grid rows
            target_pages: Target number of pages (optional)
            
        Returns:
            Recommended cell size in mm
        """
        if target_pages is None:
            # Use default that fits within constraints
            return self.calculate_cell_size_for_symbols()
        
        # Calculate cell size needed to fit in target pages
        cells_per_page = (grid_cols * grid_rows) / target_pages
        cells_per_side = math.sqrt(cells_per_page)
        
        # Convert to mm based on usable area
        usable_diagonal_mm = math.sqrt(
            self.specs.usable_width_mm ** 2 + self.specs.usable_height_mm ** 2
        )
        suggested_size = usable_diagonal_mm / (cells_per_side * math.sqrt(2))
        
        # Clamp to valid range
        return max(2.3, min(3.0, suggested_size))


# Default engine instance
_default_engine: Optional[PrintMathEngine] = None


def get_print_math_engine(specs: Optional[PrintSpecs] = None) -> PrintMathEngine:
    """Get default print math engine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = PrintMathEngine(specs)
    return _default_engine
