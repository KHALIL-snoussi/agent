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
        
        print(f"Resizing image from {w}×{h} to {grid_specs.cols}×{grid_specs.rows} grid...")
        
        # Calculate scaling factors for area sampling
        scale_x = w / grid_specs.cols
        scale_y = h / grid_specs.rows
        
        # Create preprocessed Lab image for sampling
        preprocessed_lab = self._apply_neutral_preprocessing(image_lab)
        
        # AREA-AVERAGING SAMPLING to reduce aliasing
        # Each grid cell will be the average Lab of corresponding region
        grid_lab = np.zeros((grid_specs.rows, grid_specs.cols, 3), dtype=np.float32)
        
        for grid_y in range(grid_specs.rows):
            for grid_x in range(grid_specs.cols):
                # Calculate source region boundaries
                src_x_start = int(grid_x * scale_x)
                src_x_end = min(int((grid_x + 1) * scale_x), w)
                src_y_start = int(grid_y * scale_y)
                src_y_end = min(int((grid_y + 1) * scale_y), h)
                
                # Extract region from preprocessed image
                region = preprocessed_lab[src_y_start:src_y_end, src_x_start:src_x_end]
                
                if region.size > 0:
                    # Calculate average Lab values for this region
                    avg_l = np.mean(region[:, :, 0])
                    avg_a = np.mean(region[:, :, 1])
                    avg_b = np.mean(region[:, :, 2])
                    
                    grid_lab[grid_y, grid_x] = [avg_l, avg_a, avg_b]
        
        print(f"Applied area-averaging sampling with scale factors: {scale_x:.2f}x, {scale_y:.2f}x")
        return grid_lab
    
    def _apply_neutral_preprocessing(self, image_lab: np.ndarray) -> np.ndarray:
        """Apply advanced neutral preprocessing for optimal 7-color quantization."""
        # Make a copy to avoid modifying original
        processed = image_lab.copy()
        
        # Extract channels
        l_channel = processed[:, :, 0]
        a_channel = processed[:, :, 1]
        b_channel = processed[:, :, 2]
        
        print("Applying advanced neutral preprocessing...")
        
        # 1. WHITE BALANCE using grey-world assumption
        # Compute grey-world correction factors
        l_mean = np.mean(l_channel)
        a_mean = np.mean(a_channel)
        b_mean = np.mean(b_channel)
        
        # Grey world assumes average scene is neutral grey (a=0, b=0)
        grey_factor = 1.0
        if abs(a_mean) > 2.0:  # Significant color cast
            grey_factor = min(grey_factor, 2.0 / abs(a_mean))
        if abs(b_mean) > 2.0:
            grey_factor = min(grey_factor, 2.0 / abs(b_mean))
        
        # Apply white balance correction
        a_channel = a_channel * grey_factor
        b_channel = b_channel * grey_factor
        
        # 2. ADVANCED CONTRAST ENHANCEMENT using CLAHE
        try:
            from skimage.exposure import equalize_adapthist
            
            # Apply CLAHE to L channel for local contrast enhancement
            l_clahe = equalize_adapthist(l_channel / 100.0, clip_limit=0.02)
            l_enhanced = l_clahe * 100.0
            
            print("Applied CLAHE contrast enhancement")
        except ImportError:
            # Fallback to improved S-curve if skimage not available
            # Normalize to 0-1 range
            l_norm = l_channel / 100.0
            
            # Adaptive S-curve based on histogram
            hist, bins = np.histogram(l_norm, bins=256, range=(0, 1))
            cdf = hist.cumsum()
            cdf_normalized = cdf / cdf[-1]
            
            # Apply adaptive S-curve
            l_enhanced = np.interp(l_norm, bins[:-1], cdf_normalized)
            l_enhanced = np.power(l_enhanced, 1.05)  # Mild gamma boost
            
            print("Applied adaptive S-curve contrast enhancement")
        
        # 3. BILATERAL FILTER for edge-preserving noise reduction
        try:
            from skimage.restoration import denoise_bilateral
            
            # Apply bilateral filter to each channel (handle API compatibility)
            try:
                # Newer skimage versions
                l_filtered = denoise_bilateral(l_enhanced, sigma_spatial=1.0, sigma_color=0.1, channel_axis=None)
                a_filtered = denoise_bilateral(a_channel, sigma_spatial=1.0, sigma_color=0.1, channel_axis=None)
                b_filtered = denoise_bilateral(b_channel, sigma_spatial=1.0, sigma_color=0.1, channel_axis=None)
            except TypeError:
                # Older skimage versions
                l_filtered = denoise_bilateral(l_enhanced, sigma_spatial=1.0, sigma_color=0.1, multichannel=False)
                a_filtered = denoise_bilateral(a_channel, sigma_spatial=1.0, sigma_color=0.1, multichannel=False)
                b_filtered = denoise_bilateral(b_channel, sigma_spatial=1.0, sigma_color=0.1, multichannel=False)
            
            print("Applied bilateral filtering for noise reduction")
        except ImportError:
            # Fallback to simple Gaussian filter
            from scipy import ndimage
            l_filtered = ndimage.gaussian_filter(l_enhanced, sigma=0.8)
            a_filtered = ndimage.gaussian_filter(a_channel, sigma=0.8)
            b_filtered = ndimage.gaussian_filter(b_channel, sigma=0.8)
            
            print("Applied Gaussian filtering fallback")
        
        # 4. ADAPTIVE GAMMA CORRECTION based on histogram analysis
        l_hist, l_bins = np.histogram(l_filtered, bins=256, range=(0, 100))
        l_cdf = l_hist.cumsum()
        l_cdf_normalized = l_cdf / l_cdf[-1]
        
        # Calculate optimal gamma based on median brightness
        median_l = np.median(l_filtered)
        target_gamma = 1.0
        if median_l < 40:  # Dark image
            target_gamma = 0.9  # Brighten shadows
        elif median_l > 60:  # Bright image
            target_gamma = 1.1  # Slight darkening
        
        # Apply gamma correction
        l_final = np.power(l_filtered / 100.0, target_gamma) * 100.0
        
        # 5. FINAL CHANNEL COMBINATION
        processed[:, :, 0] = np.clip(l_final, 0, 100)
        processed[:, :, 1] = np.clip(a_filtered, -128, 127)
        processed[:, :, 2] = np.clip(b_filtered, -128, 127)
        
        print("Advanced neutral preprocessing complete")
        return processed
    
    def _quantize_pixels_deltae(self, image_lab: np.ndarray) -> np.ndarray:
        """Quantize pixels using improved ΔE2000 assignment with adaptive weighting."""
        h, w = image_lab.shape[:2]
        grid_data = np.zeros((h, w), dtype=np.uint8)
        
        print(f"Calculating ΔE2000 distances for {h*w:,} pixels...")
        
        # Pre-calculate palette color distributions for better weighting
        # Focus on L* (lightness) channel as it's most perceptually important
        l_weights = np.array([1.5, 1.0, 1.0, 1.2, 1.1, 1.0, 1.3])  # Emphasize key colors
        
        # Process in chunks for memory efficiency
        chunk_size = 5000  # Smaller chunks for better control
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
                        
                        # Calculate weighted ΔE2000 to all palette colors
                        min_distance = float('inf')
                        best_index = 0
                        
                        for i, palette_lab in enumerate(self.palette_lab):
                            # Standard ΔE2000 calculation
                            distance = CIEDE2000.delta_e2000(
                                tuple(pixel_lab), 
                                tuple(palette_lab)
                            )
                            
                            # Apply perceptual weighting
                            # Lightness differences are more perceptible than chroma differences
                            l_diff = abs(pixel_lab[0] - palette_lab[0])
                            if l_diff > 20:  # Large lightness difference penalty
                                distance *= 1.2
                            
                            # Apply color-specific weighting
                            weighted_distance = distance * l_weights[i]
                            
                            if weighted_distance < min_distance:
                                min_distance = weighted_distance
                                best_index = i
                        
                        grid_data[y + cy, x + cx] = best_index
        
        print(f"ΔE2000 quantization complete")
        return grid_data
    
    def _apply_spatial_smoothing(self, grid_data: np.ndarray) -> np.ndarray:
        """Apply advanced spatial smoothing with enhanced edge preservation."""
        h, w = grid_data.shape
        print("Applying advanced spatial smoothing...")
        
        # Multi-stage edge-aware smoothing approach
        result = grid_data.copy()
        
        # STAGE 1: Gradient-based edge detection for better boundary preservation
        # Calculate gradients in both directions
        grad_x = np.abs(np.diff(grid_data.astype(float), axis=1, prepend=0))
        grad_y = np.abs(np.diff(grid_data.astype(float), axis=0, prepend=0))
        
        # Pad gradients to match original size
        grad_x = np.pad(grad_x, ((0, 1)), mode='constant')
        grad_y = np.pad(grad_y, ((1, 0)), mode='constant')
        
        # Combined edge strength (Euclidean gradient magnitude)
        edge_strength = np.sqrt(grad_x**2 + grad_y**2)
        strong_edges = edge_strength > 2.0  # Threshold for strong edges
        
        # STAGE 2: Context-aware median filtering
        # Apply 3×3 majority filter for noise reduction
        for y in range(1, h-1):
            for x in range(1, w-1):
                if strong_edges[y, x]:  # Skip strong edges entirely
                    continue
                
                # Get 3×3 neighborhood
                neighborhood = grid_data[max(0, y-1):y+2, max(0, x-1):x+2]
                
                # Find majority color in neighborhood
                unique, counts = np.unique(neighborhood, return_counts=True)
                if len(unique) > 0:
                    majority_idx = unique[np.argmax(counts)]
                    
                    # Only apply majority if center is isolated and majority is dominant
                    if (counts[np.argmax(counts)] >= 5 and 
                        grid_data[y, x] != majority_idx):
                        result[y, x] = majority_idx
        
        # STAGE 3: Edge-preserving final cleanup
        # Remove single-pixel noise using enhanced detection
        for y in range(1, h-1):
            for x in range(1, w-1):
                center = result[y, x]
                
                # Get 8-connected neighborhood after majority filtering
                neighbors = [
                    result[y-1, x-1], result[y-1, x], result[y-1, x+1],
                    result[y, x-1],                     result[y, x+1],
                    result[y+1, x-1], result[y+1, x], result[y+1, x+1]
                ]
                
                # Enhanced isolation detection
                neighbor_set = set(neighbors)
                center_unique = center not in neighbor_set
                
                if (center_unique and len(neighbor_set) >= 7):
                    # If center is isolated but neighborhood is mostly uniform, fix it
                    unique_vals, counts = np.unique(neighbors, return_counts=True)
                    dominant_color = unique_vals[np.argmax(counts)]
                    result[y, x] = dominant_color
        
        print(f"Applied advanced spatial smoothing: strong edges preserved, noise reduced")
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
