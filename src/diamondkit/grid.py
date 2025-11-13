"""
Canvas grid generation with symbol assignment and drill counting.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import string
from dataclasses import dataclass

from .dmc import DMCColor
from .config import Config


@dataclass
class GridCell:
    """Represents a single cell in the diamond painting grid."""
    x: int
    y: int
    dmc_color: DMCColor
    symbol: str
    
    @property
    def rgb(self) -> Tuple[int, int, int]:
        """Get RGB color for this cell."""
        return self.dmc_color.rgb
    
    @property
    def hex(self) -> str:
        """Get hex color for this cell."""
        return self.dmc_color.hex


class SymbolGenerator:
    """Generates unique symbols for DMC colors."""
    
    def __init__(self):
        """Initialize symbol generator with available symbols."""
        # High-contrast symbol set: digits, letters, and special characters
        self.symbols = (
            list(string.digits) +  # 0-9
            list(string.ascii_uppercase) +  # A-Z
            list(string.ascii_lowercase) +  # a-z
            ['@', '#', '$', '%', '&', '*', '+', '-', '=', '~', '^']  # Special chars
        )
        self.next_index = 0
    
    def get_symbol(self) -> str:
        """Get next available symbol."""
        if self.next_index >= len(self.symbols):
            # Start using multi-character symbols if we run out
            symbol = f"{self.symbols[0]}{self.symbols[self.next_index % len(self.symbols)]}"
        else:
            symbol = self.symbols[self.next_index]
        
        self.next_index += 1
        return symbol
    
    def reset(self):
        """Reset symbol generator."""
        self.next_index = 0


class CanvasGrid:
    """Manages the diamond painting canvas grid."""
    
    def __init__(self, config: Config, dmc_colors: List[DMCColor]):
        """Initialize canvas grid with configuration and DMC colors."""
        self.config = config
        self.dmc_colors = dmc_colors
        self.cells_w = config.canvas.cells_w
        self.cells_h = config.canvas.cells_h
        self.total_cells = self.cells_w * self.cells_h
        
        # Initialize symbol generator
        self.symbol_gen = SymbolGenerator()
        
        # Create color to symbol mapping
        self.color_to_symbol = self._create_symbol_mapping()
        
        # Initialize grid data
        self.grid_data: Optional[np.ndarray] = None
        self.color_counts: Dict[str, int] = {}
        
        # Grid layout properties
        self.cell_size_mm = config.canvas.drill_size_mm
        self.canvas_width_mm = config.canvas.width_cm * 10
        self.canvas_height_mm = config.canvas.height_cm * 10
    
    def _create_symbol_mapping(self) -> Dict[str, str]:
        """Create mapping from DMC codes to symbols."""
        mapping = {}
        self.symbol_gen.reset()
        
        # Sort colors by expected usage (darker/more prominent colors first)
        sorted_colors = sorted(
            self.dmc_colors, 
            key=lambda c: sum(c.rgb)  # Simple heuristic: darker colors first
        )
        
        for color in sorted_colors:
            mapping[color.dmc_code] = self.symbol_gen.get_symbol()
        
        return mapping
    
    def create_grid(self, image_lab: np.ndarray) -> np.ndarray:
        """
        Create diamond painting grid from quantized Lab image.
        
        Args:
            image_lab: Quantized image in Lab color space
            
        Returns:
            Grid array with DMC color indices
        """
        print(f"Creating {self.cells_w}x{self.cells_h} diamond painting grid...")
        
        # Convert Lab image to DMC color indices
        self.grid_data = self._map_lab_to_dmc_indices(image_lab)
        
        # Calculate color counts
        self._calculate_color_counts()
        
        print(f"Grid created with {len(self.dmc_colors)} colors")
        self._print_color_statistics()
        
        return self.grid_data
    
    def _map_lab_to_dmc_indices(self, image_lab: np.ndarray) -> np.ndarray:
        """Map Lab image to DMC color indices."""
        h, w = image_lab.shape[:2]
        
        # Resize image to grid dimensions
        from PIL import Image
        lab_rgb = self._lab_to_rgb_image(image_lab)
        pil_image = Image.fromarray(lab_rgb)
        resized_image = pil_image.resize((self.cells_w, self.cells_h), Image.LANCZOS)
        resized_array = np.array(resized_image)
        
        # Map each pixel to nearest DMC color
        grid = np.zeros((self.cells_h, self.cells_w), dtype=int)
        
        for y in range(self.cells_h):
            for x in range(self.cells_w):
                rgb = tuple(resized_array[y, x])
                
                # Find nearest DMC color
                min_distance = float('inf')
                best_index = 0
                
                for i, color in enumerate(self.dmc_colors):
                    # Simple RGB distance for speed
                    distance = sum((rgb[j] - color.rgb[j])**2 for j in range(3))
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_index = i
                
                grid[y, x] = best_index
        
        return grid
    
    def _lab_to_rgb_image(self, lab_array: np.ndarray) -> np.ndarray:
        """Convert Lab array to RGB image array."""
        h, w, c = lab_array.shape
        rgb_array = np.zeros((h, w, 3), dtype=np.uint8)
        
        for y in range(h):
            for x in range(w):
                lab = lab_array[y, x]
                rgb = self._lab_to_rgb_single(lab)
                rgb_array[y, x] = rgb
        
        return rgb_array
    
    def _lab_to_rgb_single(self, lab: np.ndarray) -> Tuple[int, int, int]:
        """Convert single Lab pixel to RGB."""
        l, a, b_ = lab
        
        # Lab to XYZ
        def lab_inverse(t):
            return t**3 if t > 0.008856 else (t - 16/116) / 7.787
        
        y = (l + 16) / 116
        x = a / 500 + y
        z = y - b_ / 200
        
        x = lab_inverse(x) * 95.047 / 100
        y = lab_inverse(y) * 100.0 / 100
        z = lab_inverse(z) * 108.883 / 100
        
        # XYZ to sRGB
        r = x * 3.2406 + y * -1.5372 + z * -0.4986
        g = x * -0.9689 + y * 1.8758 + z * 0.0415
        b = x * 0.0557 + y * -0.2040 + z * 1.0570
        
        # Gamma correction
        def gamma_inverse(c):
            return 1.055 * (c ** (1/2.4)) - 0.055 if c > 0.0031308 else 12.92 * c
        
        r = gamma_inverse(r)
        g = gamma_inverse(g)
        b = gamma_inverse(b)
        
        # Clamp and convert to 0-255
        r = int(np.clip(r * 255, 0, 255))
        g = int(np.clip(g * 255, 0, 255))
        b = int(np.clip(b * 255, 0, 255))
        
        return (r, g, b)
    
    def _calculate_color_counts(self):
        """Calculate how many cells use each color."""
        self.color_counts = {}
        
        if self.grid_data is not None:
            # Count occurrences of each color index
            indices, counts = np.unique(self.grid_data, return_counts=True)
            
            for index, count in zip(indices, counts):
                if index < len(self.dmc_colors):
                    color = self.dmc_colors[index]
                    self.color_counts[color.dmc_code] = {
                        'color': color,
                        'count': int(count),
                        'percentage': (count / self.total_cells) * 100,
                        'bags_needed': self._calculate_bags(count)
                    }
    
    def _calculate_bags(self, count: int) -> int:
        """Calculate number of bags needed for given drill count."""
        # Include spare ratio
        total_count = count * (1 + self.config.export.spare_ratio)
        # Calculate bags of specified size
        return int(np.ceil(total_count / self.config.export.bag_size))
    
    def _print_color_statistics(self):
        """Print color usage statistics."""
        print("\n" + "="*60)
        print("COLOR USAGE STATISTICS")
        print("="*60)
        print(f"Grid size: {self.cells_w} x {self.cells_h} drills")
        print(f"Total cells: {self.total_cells:,}")
        print(f"Colors used: {len(self.dmc_colors)}")
        print(f"Spare ratio: {self.config.export.spare_ratio:.0%}")
        print(f"Total drills with spare: {self._get_total_drills():,}")
        print()
        
        # Sort by count (descending)
        sorted_counts = sorted(
            self.color_counts.items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        )
        
        print("Color breakdown:")
        for dmc_code, stats in sorted_counts:
            color = stats['color']
            count = stats['count']
            percentage = stats['percentage']
            bags = stats['bags_needed']
            
            print(f"  {color.name:<20} ({dmc_code:>6}): {count:>6,} drills "
                  f"({percentage:>5.1f}%) - {bags:>2} bags")
        print("="*60)
    
    def _get_total_drills(self) -> int:
        """Get total number of drills including spares."""
        if not self.color_counts:
            return 0
        
        total = sum(stats['count'] for stats in self.color_counts.values())
        return int(total * (1 + self.config.export.spare_ratio))
    
    def get_cell_at(self, x: int, y: int) -> Optional[GridCell]:
        """Get grid cell at specified coordinates."""
        if self.grid_data is None:
            return None
        
        if not (0 <= x < self.cells_w and 0 <= y < self.cells_h):
            return None
        
        color_index = self.grid_data[y, x]
        if color_index >= len(self.dmc_colors):
            return None
        
        color = self.dmc_colors[color_index]
        symbol = self.color_to_symbol.get(color.dmc_code, '?')
        
        return GridCell(x, y, color, symbol)
    
    def get_color_legend(self) -> List[Dict]:
        """Get color legend sorted by usage."""
        if not self.color_counts:
            return []
        
        # Sort by count (descending)
        sorted_items = sorted(
            self.color_counts.items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        )
        
        legend = []
        for dmc_code, stats in sorted_items:
            color = stats['color']
            symbol = self.color_to_symbol.get(dmc_code, '?')
            
            legend.append({
                'symbol': symbol,
                'dmc_code': dmc_code,
                'name': color.name,
                'rgb': color.rgb,
                'hex': color.hex,
                'count': stats['count'],
                'percentage': stats['percentage'],
                'bags_needed': stats['bags_needed']
            })
        
        return legend
    
    def get_grid_info(self) -> dict:
        """Get comprehensive grid information."""
        return {
            'dimensions': {
                'cells_w': self.cells_w,
                'cells_h': self.cells_h,
                'total_cells': self.total_cells,
                'canvas_width_mm': self.canvas_width_mm,
                'canvas_height_mm': self.canvas_height_mm,
                'cell_size_mm': self.cell_size_mm,
                'drill_shape': self.config.canvas.drill_shape
            },
            'colors': {
                'total_used': len(self.dmc_colors),
                'unique_codes': [color.dmc_code for color in self.dmc_colors],
                'color_counts': self.color_counts
            },
            'symbols': self.color_to_symbol,
            'total_drills_with_spare': self._get_total_drills()
        }
    
    def create_grid_visualization(self) -> np.ndarray:
        """Create visual representation of the grid."""
        if self.grid_data is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Create RGB image
        vis_image = np.zeros((self.cells_h, self.cells_w, 3), dtype=np.uint8)
        
        for y in range(self.cells_h):
            for x in range(self.cells_w):
                cell = self.get_cell_at(x, y)
                if cell:
                    vis_image[y, x] = cell.rgb
        
        # Scale up for visibility
        scale_factor = min(5, 1000 // max(self.cells_w, self.cells_h))
        if scale_factor > 1:
            from PIL import Image
            pil_image = Image.fromarray(vis_image)
            pil_image = pil_image.resize(
                (self.cells_w * scale_factor, self.cells_h * scale_factor),
                Image.NEAREST
            )
            vis_image = np.array(pil_image)
        
        return vis_image
    
    def get_grid_tiles(self, tile_size: Tuple[int, int], 
                      overlap: int = 0) -> List[Tuple[int, int, np.ndarray]]:
        """
        Split grid into tiles for multi-page PDF generation.
        
        Args:
            tile_size: (width, height) in cells for each tile
            overlap: Number of overlapping cells between tiles
            
        Returns:
            List of (x, y, tile_data) tuples
        """
        if self.grid_data is None:
            return []
        
        tile_w, tile_h = tile_size
        tiles = []
        
        for y_start in range(0, self.cells_h, tile_h - overlap):
            for x_start in range(0, self.cells_w, tile_w - overlap):
                # Calculate tile boundaries
                x_end = min(x_start + tile_w, self.cells_w)
                y_end = min(y_start + tile_h, self.cells_h)
                
                # Extract tile data
                tile_data = self.grid_data[y_start:y_end, x_start:x_end]
                
                tiles.append((x_start, y_start, tile_data))
        
        return tiles
