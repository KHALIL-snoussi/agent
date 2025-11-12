"""
GRID_INDEX_MAP system for fixed palette diamond painting.
Handles ΔE2000 color quantization to fixed 7-color palettes with invariance proof.
"""

import hashlib
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
from scipy import ndimage
from skimage.filters import sobel

from .dmc import DMCColor, CIEDE2000, DMCColor
from .fixed_palettes import get_fixed_palette_colors
from .print_math import GridSpecs


@dataclass
class GridIndexMap:
    """Represents fixed grid index map with hash verification."""
    grid_data: np.ndarray  # 2D array of cluster indices (0-6)
    palette_colors: List[DMCColor]  # 7 fixed colors for this style
    grid_specs: GridSpecs
    style_name: str
    grid_hash: str  # SHA-256 hash for invariance proof
    
    def __post_init__(self):
        """Calculate grid hash after initialization."""
        self.grid_hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of grid data and palette for invariance proof."""
        # Create hash input from grid dimensions, data, and palette codes
        hash_input = (
            f"{self.grid_specs.cols}x{self.grid_specs.rows}|"
            f"{self.style_name}|"
            f"{'|'.join(color.dmc_code for color in self.palette_colors)}|"
            f"{self.grid_data.tobytes().hex()}"
        )
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]  # First 16 chars
    
    def verify_invariance(self, other_hash: str) -> bool:
        """Verify that this grid matches another grid (for invariance testing)."""
        return self.grid_hash == other_hash
    
    def get_cell_at(self, x: int, y: int) -> int:
        """Get cluster index at specified coordinates."""
        if not (0 <= x < self.grid_specs.cols and 0 <= y < self.grid_specs.rows):
            raise IndexError(f"Coordinates ({x}, {y}) outside grid bounds")
        return self.grid_data[y, x]
    
    def get_color_at(self, x: int, y: int) -> DMCColor:
        """Get DMC color at specified coordinates."""
        cluster_idx = self.get_cell_at(x, y)
        return self.palette_colors[cluster_idx]


class ColorQuantizerFixed:
    """Fixed palette quantizer using ΔE2000 color distance."""
    
    def __init__(self, style_name: str):
        """Initialize quantizer for a specific style."""
        self.style_name = style_name
        self.palette_colors = get_fixed_palette_colors(style_name)
        
        if len(self.palette_colors) != 7:
            raise ValueError(f"Style {style_name} must have exactly 7 colors, got {len(self.palette_colors)}")
        
        # Pre-compute Lab values for efficiency
        self.palette_lab = np.array([color.lab for color in self.palette_colors])
        self.dmc_codes = [color.dmc_code for color in self.palette_colors]
    
    def quantize_image_to_grid(self, image_lab: np.ndarray, grid_specs: GridSpecs,
                             enable_smoothing: bool = True) -> GridIndexMap:
        """
        Quantize image to fixed palette using ΔE2000 nearest neighbor.
        
        Args:
            image_lab: Input image in Lab color space (H, W, 3)
            grid_specs: Target grid specifications
            enable_smoothing: Whether to apply spatial smoothing to reduce speckle
            
        Returns:
            GridIndexMap with fixed palette assignments
        """
        print(f"Quantizing to fixed {self.style_name} palette using ΔE2000...")
        
        # Resize image to grid dimensions
        resized_lab = self._resize_image_to_grid(image_lab, grid_specs)
        
        # Apply mild preprocessing (neutral, as specified)
        preprocessed_lab = self._apply_neutral_preprocessing(resized_lab)
        
        # Quantize each pixel to nearest palette color using ΔE2000
        grid_data = self._quantize_pixels_deltae(preprocessed_lab)
        
        # Optional spatial smoothing to reduce speckle
        if enable_smoothing:
            grid_data = self._apply_spatial_smoothing(grid_data)
        
        # Create and return grid index map
        return GridIndexMap(
            grid_data=grid_data,
            palette_colors=self.palette_colors,
            grid_specs=grid_specs,
            style_name=self.style_name,
            grid_hash=""  # Will be calculated in __post_init__
        )
    
    def _resize_image_to_grid(self, image_lab: np.ndarray, grid_specs: GridSpecs) -> np.ndarray:
        """Resize image to match grid dimensions using high-quality interpolation."""
        from PIL import Image
        
        h, w = image_lab.shape[:2]
        
        if w == grid_specs.cols and h == grid_specs.rows:
            return image_lab
        
        # Convert Lab to RGB for PIL resizing (more reliable)
        rgb_image = self._lab_to_rgb_image(image_lab)
        pil_image = Image.fromarray(rgb_image)
        
        # Resize using LANCZOS for high quality
        resized_rgb = np.array(pil_image.resize(
            (grid_specs.cols, grid_specs.rows), 
            Image.LANCZOS
        ))
        
        # Convert back to Lab
        return self._rgb_image_to_lab(resized_rgb)
    
    def _apply_neutral_preprocessing(self, image_lab: np.ndarray) -> np.ndarray:
        """Apply neutral preprocessing (light denoise + exposure/contrast fix only)."""
        # Make a copy to avoid modifying original
        processed = image_lab.copy()
        
        # Very light contrast enhancement on L channel only
        # Ensure we work within valid Lab L range (0-100)
        processed[:, :, 0] = np.clip(processed[:, :, 0] * 1.02, 0, 100)  # L channel only
        
        return processed
    
    def _quantize_pixels_deltae(self, image_lab: np.ndarray) -> np.ndarray:
        """Quantize pixels using ΔE2000 nearest neighbor assignment."""
        h, w = image_lab.shape[:2]
        grid_data = np.zeros((h, w), dtype=np.uint8)
        
        print(f"Calculating ΔE2000 distances for {h*w:,} pixels...")
        
        # Process in chunks for memory efficiency
        chunk_size = 10000
        for y in range(0, h, chunk_size):
            y_end = min(y + chunk_size, h)
            
            for x in range(0, w, chunk_size):
                x_end = min(x + chunk_size, w)
                
                # Extract chunk
                chunk = image_lab[y:y_end, x:x_end]
                chunk_h, chunk_w = chunk.shape[:2]
                
                # Find nearest color for each pixel in chunk
                for cy in range(chunk_h):
                    for cx in range(chunk_w):
                        pixel_lab = chunk[cy, cx]
                        
                        # Calculate ΔE2000 to all palette colors
                        min_distance = float('inf')
                        best_index = 0
                        
                        for i, palette_lab in enumerate(self.palette_lab):
                            distance = CIEDE2000.delta_e2000(
                                tuple(pixel_lab), 
                                tuple(palette_lab)
                            )
                            
                            if distance < min_distance:
                                min_distance = distance
                                best_index = i
                        
                        grid_data[y + cy, x + cx] = best_index
        
        print(f"ΔE2000 quantization complete")
        return grid_data
    
    def _apply_spatial_smoothing(self, grid_data: np.ndarray) -> np.ndarray:
        """Apply spatial smoothing to reduce speckle while preserving edges."""
        # Use median filter with small kernel to reduce isolated pixels
        smoothed = ndimage.median_filter(grid_data, size=3)
        
        # Only apply smoothing where it doesn't create new colors at edges
        # Detect edges using Sobel filter
        edges = sobel(grid_data.astype(float))
        edge_mask = edges > 0.1
        
        # Preserve original values at edges, use smoothed elsewhere
        result = np.where(edge_mask, grid_data, smoothed)
        
        return result.astype(np.uint8)
    
    def _lab_to_rgb_image(self, lab_array: np.ndarray) -> np.ndarray:
        """Convert Lab array to RGB image."""
        h, w, c = lab_array.shape
        rgb_array = np.zeros((h, w, 3), dtype=np.uint8)
        
        for y in range(h):
            for x in range(w):
                lab = lab_array[y, x]
                rgb = self._lab_to_rgb_single(lab)
                rgb_array[y, x] = rgb
        
        return rgb_array
    
    def _rgb_image_to_lab(self, rgb_array: np.ndarray) -> np.ndarray:
        """Convert RGB image to Lab array."""
        h, w, c = rgb_array.shape
        lab_array = np.zeros((h, w, 3), dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                rgb = tuple(rgb_array[y, x])
                lab = DMCColor._rgb_to_lab(rgb)
                lab_array[y, x] = lab
        
        return lab_array
    
    def _lab_to_rgb_single(self, lab: np.ndarray) -> Tuple[int, int, int]:
        """Convert single Lab pixel to RGB."""
        l, a, b_ = lab
        
        # Lab to XYZ
        def lab_inverse(t):
            return t**3 if t > 0.008856 else (t - 16/116) / 7.787
        
        y = (l + 16) / 116
        x = a / 500 + y
        z = y - b_ / 200
        
        # D65 white point
        x_ref, y_ref, z_ref = 95.047, 100.0, 108.883
        
        x = lab_inverse(x) * x_ref / 100
        y = lab_inverse(y) * y_ref / 100
        z = lab_inverse(z) * z_ref / 100
        
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
    
    def calculate_delta_e_stats(self, image_lab: np.ndarray, grid_map: GridIndexMap) -> Dict[str, float]:
        """Calculate ΔE statistics for quantization."""
        h, w = image_lab.shape[:2]
        distances = []
        
        # Sample pixels for efficiency (avoid processing every pixel)
        sample_step = max(1, (h * w) // 10000)  # Target ~10k samples max
        
        # Resize original image to match grid for comparison
        resized_lab = self._resize_image_to_grid(image_lab, grid_map.grid_specs)
        
        for y in range(0, h, sample_step):
            for x in range(0, w, sample_step):
                if y < grid_map.grid_specs.rows and x < grid_map.grid_specs.cols:
                    original_lab = resized_lab[y, x]
                    cluster_idx = grid_map.get_cell_at(x, y)
                    palette_lab = self.palette_lab[cluster_idx]
                    
                    distance = CIEDE2000.delta_e2000(
                        tuple(original_lab), 
                        tuple(palette_lab)
                    )
                    distances.append(distance)
        
        if not distances:
            return {"delta_e_mean": 0.0, "delta_e_max": 0.0, "delta_e_std": 0.0}
        
        return {
            "delta_e_mean": float(np.mean(distances)),
            "delta_e_max": float(np.max(distances)),
            "delta_e_std": float(np.std(distances))
        }
    
    def get_palette_info(self) -> Dict[str, Any]:
        """Get information about the fixed palette."""
        return {
            "style_name": self.style_name,
            "total_colors": len(self.palette_colors),
            "dmc_codes": self.dmc_codes,
            "colors": [
                {
                    "dmc_code": color.dmc_code,
                    "name": color.name,
                    "hex": color.hex,
                    "rgb": color.rgb,
                    "lab": color.lab
                }
                for color in self.palette_colors
            ]
        }


def create_grid_index_map(image_lab: np.ndarray, grid_specs: GridSpecs, 
                        style_name: str, enable_smoothing: bool = True) -> GridIndexMap:
    """
    Convenience function to create a grid index map.
    
    Args:
        image_lab: Input image in Lab color space
        grid_specs: Target grid specifications
        style_name: Style name (ORIGINAL, VINTAGE, POPART)
        enable_smoothing: Whether to apply spatial smoothing
        
    Returns:
        GridIndexMap with quantized assignments
    """
    quantizer = ColorQuantizerFixed(style_name)
    return quantizer.quantize_image_to_grid(image_lab, grid_specs, enable_smoothing)


def verify_grid_invariance(grid1: GridIndexMap, grid2: GridIndexMap) -> bool:
    """
    Verify that two grids have identical assignments (for invariance testing).
    
    Args:
        grid1: First grid index map
        grid2: Second grid index map
        
    Returns:
        True if grids are identical
    """
    return grid1.verify_invariance(grid2.grid_hash)
