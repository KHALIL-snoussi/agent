"""
GRID_INDEX_MAP system for fixed palette diamond painting.
Handles DeltaE2000 color quantization to fixed 7-color palettes with invariance proof.
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
    """Fixed palette quantizer using DeltaE2000 color distance with intelligent dominance control."""
    
    def __init__(self, style_name: str, smoothing_kernel: int = 3,
                 popart_edge_bias: float = 0.0, popart_edge_threshold: float = 0.25):
        """Initialize quantizer for a specific style."""
        self.style_name = style_name
        self.palette_colors = get_fixed_palette_colors(style_name)
        
        if len(self.palette_colors) != 7:
            raise ValueError(f"Style {style_name} must have exactly 7 colors, got {len(self.palette_colors)}")
        
        # Pre-compute Lab values for efficiency
        self.palette_lab = np.array([color.lab for color in self.palette_colors])
        self.dmc_codes = [color.dmc_code for color in self.palette_colors]
        
        # Identify lightest color for dominance control (works for all palettes)
        self.lightest_color_idx = self._find_lightest_color_index()
        self.black_color_idx = self._find_color_index("310")
        
        if smoothing_kernel is None or smoothing_kernel < 3:
            smoothing_kernel = 3
        if smoothing_kernel % 2 == 0:
            smoothing_kernel += 1
        self.smoothing_kernel = smoothing_kernel
        self.popart_edge_bias = max(0.0, popart_edge_bias)
        self.popart_edge_threshold = max(0.0, min(popart_edge_threshold, 1.0))
    
    def _find_lightest_color_index(self) -> int:
        """Find index of lightest color (highest L* value) in palette."""
        max_l = -1
        lightest_idx = 0
        for i, color in enumerate(self.palette_colors):
            if color.lab[0] > max_l:  # L* channel
                max_l = color.lab[0]
                lightest_idx = i
        return lightest_idx
    
    def _find_color_index(self, dmc_code: str) -> Optional[int]:
        """Find palette index for a specific DMC code."""
        for i, color in enumerate(self.palette_colors):
            if color.dmc_code == dmc_code:
                return i
        return None
    
    def quantize_image_to_grid(self, image_lab: np.ndarray, grid_specs: GridSpecs,
                             enable_smoothing: bool = True,
                             image_is_preprocessed: bool = False) -> GridIndexMap:
        """
        Quantize image to fixed palette using DeltaE2000 nearest neighbor.
        
        Args:
            image_lab: Input image in Lab color space (H, W, 3)
            grid_specs: Target grid specifications
            enable_smoothing: Whether to apply spatial smoothing to reduce speckle
            image_is_preprocessed: Skip internal resizing/preprocessing if True
            
        Returns:
            GridIndexMap with fixed palette assignments
        """
        print(f"Quantizing to fixed {self.style_name} palette using DeltaE2000...")
        
        if image_is_preprocessed:
            preprocessed_lab = image_lab
        else:
            resized_lab = self._resize_image_to_grid(image_lab, grid_specs)
            preprocessed_lab = self._apply_neutral_preprocessing(resized_lab)
        
        edge_map = None
        if self.style_name == "POPART" and self.popart_edge_bias > 0:
            edge_map = self._compute_popart_edge_map(preprocessed_lab)
        
        # Quantize each pixel to nearest palette color using DeltaE2000
        grid_data = self._quantize_pixels_deltae(preprocessed_lab, edge_map=edge_map)
        
        # Optional spatial smoothing to reduce speckle
        if enable_smoothing and self.smoothing_kernel >= 3:
            grid_data = self._apply_spatial_smoothing(grid_data, self.smoothing_kernel)
        
        # Check for color dominance issues
        self._check_color_dominance(grid_data)
        
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
        
        print(f"Resizing image from {w}x{h} to {grid_specs.cols}x{grid_specs.rows} grid...")
        
        # Calculate scaling factors for area sampling
        scale_x = w / grid_specs.cols
        scale_y = h / grid_specs.rows
        
        # Create preprocessed Lab image for sampling
        preprocessed_lab = self._apply_neutral_preprocessing(image_lab)
        
        # ENHANCED SAMPLING: Use bilinear with edge preservation
        # Convert to RGB for PIL processing, then back to Lab
        try:
            # Convert Lab back to RGB for high-quality PIL resize
            rgb_temp = np.zeros((h, w, 3), dtype=np.uint8)
            for y in range(h):
                for x in range(w):
                    lab = image_lab[y, x]
                    rgb = self._lab_to_rgb_single(lab)
                    rgb_temp[y, x] = rgb
            
            # Use PIL for high-quality bilinear interpolation
            pil_image = Image.fromarray(rgb_temp, mode='RGB')
            resized_rgb = pil_image.resize((grid_specs.cols, grid_specs.rows), Image.LANCZOS)
            
            # Convert back to Lab
            resized_lab = np.zeros((grid_specs.rows, grid_specs.cols, 3), dtype=np.float32)
            for y in range(grid_specs.rows):
                for x in range(grid_specs.cols):
                    rgb = resized_rgb[y, x]
                    lab = self._rgb_to_lab_single(rgb)
                    resized_lab[y, x] = lab
                    
            print(f"Applied high-quality bilinear sampling with scale factors: {scale_x:.2f}x, {scale_y:.2f}x")
            return resized_lab
            
        except Exception as e:
            print(f"Warning: High-quality resize failed, using area averaging: {e}")
            
            # Fallback to area averaging
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
            
            print(f"Applied fallback area-averaging sampling with scale factors: {scale_x:.2f}x, {scale_y:.2f}x")
            return grid_lab
    
    def _apply_neutral_preprocessing(self, image_lab: np.ndarray) -> np.ndarray:
        """Apply enhanced neutral preprocessing with better color balance."""
        # Make a copy to avoid modifying original
        processed = image_lab.copy()
        
        # Extract channels
        l_channel = processed[:, :, 0]
        a_channel = processed[:, :, 1]
        b_channel = processed[:, :, 2]
        
        print("Applying enhanced neutral preprocessing...")
        
        # 1. ENHANCED WHITE BALANCE with adaptive correction
        # Grey-world assumption with adaptive scaling
        a_mean = np.mean(a_channel)
        b_mean = np.mean(b_channel)
        
        # More aggressive white balance correction to prevent color casts
        correction_factor = 1.0
        if abs(a_mean) > 2.0:  # Lower threshold for more correction
            correction_factor = min(correction_factor, 2.0 / abs(a_mean))
        if abs(b_mean) > 2.0:
            correction_factor = min(correction_factor, 2.0 / abs(b_mean))
        
        a_channel = a_channel * correction_factor
        b_channel = b_channel * correction_factor
        
        # 2. ENHANCED CONTRAST with local adaptation - FIXED VERSION
        # Use simple contrast enhancement for better compatibility
        l_min, l_max = np.percentile(l_channel, [5, 95])
        if l_max > l_min + 2:  # Only stretch if there's reasonable contrast
            l_enhanced = np.clip((l_channel - l_min) / (l_max - l_min) * 100, 0, 100)
        else:
            l_enhanced = l_channel
        
        # 3. EDGE-PRESERVING NOISE REDUCTION
        # Use bilateral filter to preserve edges while reducing noise
        try:
            from skimage.restoration import denoise_bilateral
            l_filtered = denoise_bilateral(l_enhanced, sigma_spatial=0.8, sigma_color=0.8)
            a_filtered = denoise_bilateral(a_channel, sigma_spatial=0.6, sigma_color=0.6)
            b_filtered = denoise_bilateral(b_channel, sigma_spatial=0.6, sigma_color=0.6)
        except ImportError:
            # Fallback to gaussian
            from scipy import ndimage
            l_filtered = ndimage.gaussian_filter(l_enhanced, sigma=0.8)
            a_filtered = ndimage.gaussian_filter(a_channel, sigma=0.6)
            b_filtered = ndimage.gaussian_filter(b_channel, sigma=0.6)
        
        # 4. FINAL CHANNEL COMBINATION with range protection
        processed[:, :, 0] = np.clip(l_filtered, 0, 100)
        processed[:, :, 1] = np.clip(a_filtered, -128, 127)
        processed[:, :, 2] = np.clip(b_filtered, -128, 127)
        
        print("Enhanced neutral preprocessing complete")
        return processed
    
    def _quantize_pixels_deltae(self, image_lab: np.ndarray,
                               edge_map: Optional[np.ndarray] = None) -> np.ndarray:
        """Quantize pixels using DeltaE2000 with intelligent dominance control."""
        h, w = image_lab.shape[:2]
        grid_data = np.zeros((h, w), dtype=np.uint8)
        
        print(f"Calculating DeltaE2000 distances for {h*w:,} pixels...")
        
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
                        
                        # Calculate pure DeltaE2000 to all palette colors
                        min_distance = float('inf')
                        best_index = 0
                        best_luminance_gap = float('inf')
                        
                        for i, palette_lab in enumerate(self.palette_lab):
                            # Pure DeltaE2000 calculation
                            distance = CIEDE2000.delta_e2000(
                                tuple(pixel_lab), 
                                tuple(palette_lab)
                            )
                           
                            # Apply intelligent dominance control
                            # Works for ALL palettes by targeting the lightest color
                            if i == self.lightest_color_idx:
                                # Calculate L* and chroma for this pixel
                                l_star = pixel_lab[0]
                                a_star = pixel_lab[1]
                                b_star = pixel_lab[2]
                                chroma = np.sqrt(a_star**2 + b_star**2)
                                
                                # Penalize lightest color for midtones and shadows
                                # More aggressive penalty to prevent dominance
                                if l_star < 90:  # Only allow lightest color in highlights
                                    distance += 25  # Strong penalty
                                elif chroma > 12:  # Penalize in colorful areas
                                    distance += 15
                                else:
                                    distance += 5  # Mild penalty even in bright areas
                           
                            # Minimal perceptual adjustment for extreme lightness differences
                            l_diff = abs(pixel_lab[0] - palette_lab[0])
                            if l_diff > 30:  # Only penalize very extreme differences
                                distance *= 1.1  # Minimal penalty
                           
                            luminance_gap = abs(pixel_lab[0] - palette_lab[0])
                            
                            if (
                                edge_map is not None
                                and edge_map[y + cy, x + cx]
                            ):
                                if self.black_color_idx is not None and i == self.black_color_idx:
                                    distance -= self.popart_edge_bias
                                else:
                                    distance += self.popart_edge_bias * 0.15
                            
                            if distance < min_distance - 1e-6:
                                min_distance = distance
                                best_index = i
                                best_luminance_gap = luminance_gap
                            elif abs(distance - min_distance) <= 0.2 and luminance_gap < best_luminance_gap:
                                best_index = i
                                best_luminance_gap = luminance_gap
                        
                        grid_data[y + cy, x + cx] = best_index
        
        print(f"DeltaE2000 quantization complete with intelligent dominance control")
        return grid_data
    
    def _apply_spatial_smoothing(self, grid_data: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply advanced spatial smoothing with enhanced edge preservation."""
        h, w = grid_data.shape
        print("Applying advanced spatial smoothing...")
        
        if kernel_size < 3:
            return grid_data
        if kernel_size % 2 == 0:
            kernel_size += 1
        radius = kernel_size // 2
        if h <= kernel_size or w <= kernel_size:
            return grid_data
        
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
        # Apply 3x3 majority filter for noise reduction
        for y in range(radius, h - radius):
            for x in range(radius, w - radius):
                if strong_edges[y, x]:  # Skip strong edges entirely
                    continue
                
                neighborhood = grid_data[
                    y - radius : y + radius + 1,
                    x - radius : x + radius + 1
                ]
                
                # Find majority color in neighborhood
                unique, counts = np.unique(neighborhood, return_counts=True)
                if len(unique) > 0:
                    majority_idx = unique[np.argmax(counts)]
                    
                    # Only apply majority if center is isolated and majority is dominant
                    if (
                        counts[np.argmax(counts)] >= (kernel_size ** 2 // 2)
                        and grid_data[y, x] != majority_idx
                    ):
                        result[y, x] = majority_idx
        
        # STAGE 3: Edge-preserving final cleanup
        for y in range(radius, h - radius):
            for x in range(radius, w - radius):
                center = result[y, x]
                
                neighborhood = result[
                    y - radius : y + radius + 1,
                    x - radius : x + radius + 1
                ].flatten()
                
                neighbor_set = set(neighborhood)
                if center not in neighbor_set and len(neighbor_set) >= max(3, kernel_size):
                    unique_vals, counts = np.unique(neighborhood, return_counts=True)
                    dominant_color = unique_vals[np.argmax(counts)]
                    result[y, x] = dominant_color
        
        print("Applied advanced spatial smoothing: strong edges preserved, noise reduced")
        return result.astype(np.uint8)
    
    def _compute_popart_edge_map(self, image_lab: np.ndarray) -> Optional[np.ndarray]:
        """Compute edge map that helps reinforce dark outlines in POPART mode."""
        try:
            luminance = np.clip(image_lab[:, :, 0] / 100.0, 0.0, 1.0)
            gradient = sobel(luminance)
            threshold = max(0.05, self.popart_edge_threshold)
            return gradient >= threshold
        except Exception as exc:
            print(f"Warning: Unable to compute POPART edge map ({exc})")
            return None
    
    def _check_color_dominance(self, grid_data: np.ndarray):
        """Check for color dominance issues and warn if needed."""
        h, w = grid_data.shape
        total_cells = h * w
        
        # Count cells per color
        unique_indices, counts = np.unique(grid_data, return_counts=True)
        
        # Check for dominance
        for i, count in zip(unique_indices, counts):
            percentage = (count / total_cells) * 100
            if percentage > 85:  # Lowered threshold from 90% to 85%
                dmc_code = self.dmc_codes[i]
                color_name = self.palette_colors[i].name
                print(f"[WARN] COLOR DOMINANCE WARNING: {dmc_code} ({color_name}) uses {percentage:.1f}% of cells - detail loss risk!")
            elif percentage < 2:  # Rare color threshold
                dmc_code = self.dmc_codes[i]
                color_name = self.palette_colors[i].name
                print(f"[WARN] RARE COLOR: {dmc_code} ({color_name}) only {percentage:.1f}% of grid")
        
        # Print color distribution summary
        print(f"Color distribution: ", end="")
        for i, count in zip(unique_indices, counts):
            percentage = (count / total_cells) * 100
            dmc_code = self.dmc_codes[i]
            print(f"{dmc_code}:{percentage:.0f}% ", end="")
        print()
    
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
    
    def _rgb_to_lab_single(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert single RGB pixel to Lab."""
        r, g, b = rgb
        
        # Normalize to 0-1 range
        r, g, b = r/255.0, g/255.0, b/255.0
        
        # sRGB to XYZ
        def gamma_correct(c):
            return c ** 2.4 if c > 0.04045 else c / 12.92
        
        r = gamma_correct(r)
        g = gamma_correct(g)
        b = gamma_correct(b)
        
        # Apply sRGB transformation matrix
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
        
        # D65 white point normalization
        x_ref, y_ref, z_ref = 95.047, 100.0, 108.883
        x = x / x_ref
        y = y / y_ref
        z = z / z_ref
        
        # XYZ to Lab
        def xyz_to_lab(t):
            return 116 * t if t > 0.008856 else 903.3 * t
        
        fx = xyz_to_lab(x)
        fy = xyz_to_lab(y)
        fz = xyz_to_lab(z)
        
        l_star = fy * 116 - 16
        a_star = 500 * (fx - fy)
        b_star = 200 * (fy - fz)
        
        return (l_star, a_star, b_star)
    
    def calculate_delta_e_stats(self, image_lab: np.ndarray, grid_map: GridIndexMap) -> Dict[str, float]:
        """Calculate DeltaE statistics for quantization."""
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
            "lightest_color_idx": self.lightest_color_idx,
            "lightest_color_code": self.dmc_codes[self.lightest_color_idx],
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
                        style_name: str, enable_smoothing: bool = True,
                        smoothing_kernel: int = 3,
                        image_is_preprocessed: bool = False,
                        popart_edge_bias: float = 0.0,
                        popart_edge_threshold: float = 0.25) -> GridIndexMap:
    """
    Convenience function to create a grid index map.
    
    Args:
        image_lab: Input image in Lab color space
        grid_specs: Target grid specifications
        style_name: Style name (ORIGINAL, VINTAGE, POPART)
        enable_smoothing: Whether to apply spatial smoothing
        smoothing_kernel: Kernel size for smoothing pass
        image_is_preprocessed: Skip resizing/preprocessing if True
        popart_edge_bias: Bias strength for POPART edge reinforcement
        popart_edge_threshold: Gradient threshold for POPART edges
        
    Returns:
        GridIndexMap with quantized assignments
    """
    quantizer = ColorQuantizerFixed(
        style_name,
        smoothing_kernel=smoothing_kernel,
        popart_edge_bias=popart_edge_bias,
        popart_edge_threshold=popart_edge_threshold
    )
    return quantizer.quantize_image_to_grid(
        image_lab,
        grid_specs,
        enable_smoothing=enable_smoothing,
        image_is_preprocessed=image_is_preprocessed
    )


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
