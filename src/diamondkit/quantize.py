"""
Constrained k-means color quantization with DMC palette snapping.
"""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans
import random

from .dmc import DMCPalette, DMCColor, get_dmc_palette
from .config import Config


class ColorQuantizer:
    """Performs color quantization with DMC palette constraints."""
    
    def __init__(self, config: Config, dmc_palette: Optional[DMCPalette] = None):
        """Initialize quantizer with configuration and DMC palette."""
        self.config = config
        self.dmc_palette = dmc_palette or get_dmc_palette(config.palette.dmc_file)
        self.rng = np.random.RandomState(config.processing.seed or 42)
    
    def quantize_image(self, image_lab: np.ndarray) -> Tuple[np.ndarray, List[DMCColor]]:
        """
        Quantize image to DMC palette using constrained k-means.
        
        Args:
            image_lab: Input image in Lab color space (H, W, 3)
            
        Returns:
            Tuple of (quantized_image_rgb, used_dmc_colors)
        """
        print(f"Quantizing to {self.config.palette.max_colors} DMC colors...")
        
        # Balanced downsampling for quality and speed
        total_pixels = image_lab.shape[0] * image_lab.shape[1]
        
        # Target 15k pixels for better quality k-means
        target_pixels = 15000  # Increased for better quality
        downsample_factor = max(1, int(np.sqrt(total_pixels / target_pixels)))
        downsample_factor = min(downsample_factor, 8)  # Reduced cap for better quality
        
        if downsample_factor > 1:
            print(f"ULTRA-aggressive downsampling by factor of {downsample_factor} for speed...")
            downsampled = image_lab[::downsample_factor, ::downsample_factor]
            new_pixels = downsampled.shape[0] * downsampled.shape[1]
            print(f"Reduced from {total_pixels:,} to {new_pixels:,} pixels ({new_pixels/total_pixels:.1%})")
        else:
            downsampled = image_lab
            print(f"No downsampling needed: {total_pixels:,} pixels")
        
        # Reshape downsampled image for clustering
        pixels = downsampled.reshape(-1, 3)
        original_shape = image_lab.shape
        
        # Detect skin regions if preservation is enabled
        skin_mask = None
        if self.config.palette.preserve_skin_tones:
            skin_mask = self._detect_skin_regions(image_lab)
        
        # Step 1: Run k-means clustering
        print("Running k-means clustering...")
        centroids = self._run_kmeans(pixels, self.config.palette.max_colors)
        
        # Step 2: Snap centroids to nearest DMC colors
        print("Snapping centroids to DMC palette...")
        dmc_colors = self._snap_to_dmc(centroids, skin_mask)
        
        # Step 3: Map all pixels to DMC colors
        print("Mapping pixels to DMC colors...")
        # For very large images, we can also downsample the mapping step
        if total_pixels > 1000000:  # If over 1M pixels
            mapping_downsample = min(downsample_factor, 4)  # Additional downsample for mapping
            if mapping_downsample > 1:
                print(f"  Also downmapping by {mapping_downsample}x for speed...")
                # Map on smaller image, then upscale
                small_image = image_lab[::mapping_downsample, ::mapping_downsample]
                mapped_small = self._map_pixels_to_dmc(small_image, dmc_colors, skin_mask)
                # Upsample back to original size using nearest neighbor
                from scipy import ndimage
                zoom_factors = [mapping_downsample, mapping_downsample, 1]
                quantized_image = ndimage.zoom(mapped_small, zoom_factors, order=0)
                # Trim to exact original size if needed
                quantized_image = quantized_image[:original_shape[0], :original_shape[1], :]
            else:
                quantized_image = self._map_pixels_to_dmc(image_lab, dmc_colors, skin_mask)
        else:
            quantized_image = self._map_pixels_to_dmc(image_lab, dmc_colors, skin_mask)
        
        # Reshape back to image format and ensure float32
        quantized_image = quantized_image.reshape(original_shape).astype(np.float32)
        
        print(f"Quantization complete. Used {len(dmc_colors)} DMC colors.")
        return quantized_image, dmc_colors
    
    def _run_kmeans(self, pixels: np.ndarray, k: int) -> np.ndarray:
        """Run k-means clustering on Lab pixels."""
        # Use k-means++ initialization for better results
        # Balanced for quality and speed
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=3,  # Increased for better convergence
            max_iter=100,  # Increased for better quality
            random_state=self.config.processing.seed,
            algorithm='elkan',
            tol=1e-4   # Tighter tolerance for better quality
        )
        
        print(f"  K-means on {len(pixels):,} pixels for {k} colors...")
        kmeans.fit(pixels)
        print(f"  Converged in {kmeans.n_iter_} iterations")
        return kmeans.cluster_centers_
    
    def _snap_to_dmc(self, centroids: np.ndarray, 
                     skin_mask: Optional[np.ndarray]) -> List[DMCColor]:
        """Snap k-means centroids to nearest DMC colors."""
        dmc_colors = []
        used_codes = set()
        
        for i, centroid in enumerate(centroids):
            # Convert Lab centroid to RGB for DMC matching
            rgb_centroid = self._lab_to_rgb(centroid)
            
            # Check if this centroid corresponds to skin regions
            is_skin_centroid = False
            if skin_mask is not None:
                # Find pixels assigned to this centroid
                pixel_distances = np.linalg.norm(
                    centroids - centroid, axis=1
                )
                closest_centroid_idx = np.argmin(pixel_distances)
                
                # Check if significant portion of this centroid's pixels are in skin regions
                if skin_mask is not None:
                    skin_pixels_for_centroid = np.sum(skin_mask == closest_centroid_idx)
                    total_pixels_for_centroid = np.sum(np.argmin(
                        np.linalg.norm(centroids - centroid, axis=1)
                    ) == closest_centroid_idx)
                    
                    if total_pixels_for_centroid > 0:
                        skin_ratio = skin_pixels_for_centroid / total_pixels_for_centroid
                        is_skin_centroid = skin_ratio > 0.5
            
            # Find nearest DMC color with skin tone preservation
            dmc_color = self.dmc_palette.find_nearest_color(
                rgb_centroid,
                preserve_skin_tones=self.config.palette.preserve_skin_tones,
                is_skin_region=is_skin_centroid
            )
            
            # Avoid duplicate DMC codes
            if dmc_color.dmc_code not in used_codes:
                dmc_colors.append(dmc_color)
                used_codes.add(dmc_color.dmc_code)
            else:
                # Find alternative DMC color
                alternative_color = self._find_alternative_dmc(
                    rgb_centroid, used_codes, is_skin_centroid
                )
                dmc_colors.append(alternative_color)
                used_codes.add(alternative_color.dmc_code)
        
        # Sort by usage frequency (will be updated after mapping)
        return dmc_colors
    
    def _find_alternative_dmc(self, rgb: Tuple[int, int, int], 
                             used_codes: set, 
                             is_skin_region: bool = False) -> DMCColor:
        """Find alternative DMC color avoiding used codes."""
        candidates = []
        
        for color in self.dmc_palette.colors:
            if color.dmc_code not in used_codes:
                # Calculate distance
                target_lab = DMCColor._rgb_to_lab(rgb)
                distance = self._calculate_ciede2000(target_lab, color.lab)
                
                # Apply skin tone preservation
                if self.config.palette.preserve_skin_tones and is_skin_region:
                    if not self.dmc_palette._is_skin_tone_color(color):
                        distance *= 2.0  # Higher penalty
                
                candidates.append((distance, color))
        
        # Sort by distance and return best candidate
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1] if candidates else self.dmc_palette.colors[0]
    
    def _map_pixels_to_dmc(self, pixels: np.ndarray, 
                          dmc_colors: List[DMCColor],
                          skin_mask: Optional[np.ndarray]) -> np.ndarray:
        """Map all pixels to selected DMC colors."""
        # Create Lab array for DMC colors
        dmc_lab_array = np.array([color.lab for color in dmc_colors])
        
        # Reshape pixels to 2D if it's 3D (for batch processing)
        if len(pixels.shape) == 3:
            original_shape = pixels.shape
            pixels = pixels.reshape(-1, 3)
        else:
            original_shape = None
        
        # Map each pixel to nearest DMC color
        quantized_pixels = np.zeros_like(pixels)
        
        # Use faster approach with vectorized operations
        # Calculate all distances at once for speed
        print(f"  Mapping {len(pixels):,} pixels to {len(dmc_colors)} DMC colors...")
        
        # Process in larger batches for better performance
        batch_size = 50000  # Increased from 10000 for speed
        for i in range(0, len(pixels), batch_size):
            batch = pixels[i:i + batch_size]
            
            # Use squared distances to avoid sqrt (faster for comparison)
            diff = batch[:, np.newaxis, :] - dmc_lab_array[np.newaxis, :, :]
            distances = np.sum(diff * diff, axis=2)  # Squared Euclidean distance
            
            # Find nearest DMC color for each pixel
            nearest_indices = np.argmin(distances, axis=1)
            quantized_pixels[i:i + batch_size] = dmc_lab_array[nearest_indices]
        
        # Reshape back to original shape if needed
        if original_shape is not None:
            quantized_pixels = quantized_pixels.reshape(original_shape)
        
        return quantized_pixels
    
    def _detect_skin_regions(self, image_lab: np.ndarray) -> Optional[np.ndarray]:
        """Detect skin regions in the image."""
        h, w = image_lab.shape[:2]
        skin_mask = np.zeros((h * w,), dtype=int)
        
        # Simple skin detection in Lab space
        l, a, b = image_lab[:, :, 0], image_lab[:, :, 1], image_lab[:, :, 2]
        
        # Typical skin ranges in Lab (approximate)
        skin_conditions = (
            (l >= 50) & (l <= 90) &  # Lightness
            (a >= 10) & (a <= 30) &  # Red-green axis  
            (b >= 10) & (b <= 25)    # Yellow-blue axis
        )
        
        # Assign skin region indices
        skin_regions = skin_conditions.flatten()
        skin_mask[skin_regions] = 1
        
        return skin_mask
    
    def _lab_to_rgb(self, lab: np.ndarray) -> Tuple[int, int, int]:
        """Convert Lab coordinates back to RGB."""
        l, a, b_ = lab
        
        # Lab to XYZ
        def lab_inverse(t):
            return t**3 if t > 0.008856 else (t - 16/116) / 7.787
        
        # D65 white point
        x_ref, y_ref, z_ref = 95.047, 100.0, 108.883
        
        y = (l + 16) / 116
        x = a / 500 + y
        z = y - b_ / 200
        
        x = lab_inverse(x) * x_ref / 100
        y = lab_inverse(y) * y_ref / 100  
        z = lab_inverse(z) * z_ref / 100
        
        # XYZ to sRGB (D65)
        r = x * 3.2406 + y * -1.5372 + z * -0.4986
        g = x * -0.9689 + y * 1.8758 + z * 0.0415
        b = x * 0.0557 + y * -0.2040 + z * 1.0570
        
        # Apply gamma correction
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
    
    def _calculate_ciede2000(self, lab1: np.ndarray, 
                           lab2: Tuple[float, float, float]) -> float:
        """Simple CIEDE2000 approximation for performance."""
        # For speed, use Euclidean distance in Lab space as approximation
        # Full CIEDE2000 is implemented in dmc.py for exact matching
        return np.linalg.norm(lab1 - np.array(lab2))
    
    def get_color_statistics(self, image_lab: np.ndarray, 
                           dmc_colors: List[DMCColor]) -> dict:
        """Calculate color usage statistics."""
        # Map pixels to DMC colors
        pixels = image_lab.reshape(-1, 3)
        dmc_lab_array = np.array([color.lab for color in dmc_colors])
        
        # Calculate distances and assignments
        distances = np.linalg.norm(
            pixels[:, np.newaxis, :] - dmc_lab_array[np.newaxis, :, :], 
            axis=2
        )
        assignments = np.argmin(distances, axis=1)
        
        # Count occurrences
        counts = np.bincount(assignments, minlength=len(dmc_colors))
        
        # Create statistics
        stats = {}
        total_pixels = len(pixels)
        
        for i, (color, count) in enumerate(zip(dmc_colors, counts)):
            percentage = (count / total_pixels) * 100
            stats[color.dmc_code] = {
                'color': color,
                'count': int(count),
                'percentage': percentage,
                'bags_needed': self._calculate_bags(count)
            }
        
        return stats
    
    def _calculate_bags(self, count: int) -> int:
        """Calculate number of bags needed for given drill count."""
        # Include spare ratio
        total_count = count * (1 + self.config.export.spare_ratio)
        # Calculate bags of specified size
        return int(np.ceil(total_count / self.config.export.bag_size))
