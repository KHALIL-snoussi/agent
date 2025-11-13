"""
Dithering engine with Bayer ordered dithering and Floyd-Steinberg error diffusion.
"""

import numpy as np
from typing import Optional, Tuple
from .config import Config


class DitherEngine:
    """Handles various dithering algorithms for color quantization."""
    
    # Bayer 8x8 matrix for ordered dithering
    BAYER_MATRIX_8x8 = np.array([
        [0, 48, 12, 60, 3, 51, 15, 63],
        [32, 16, 44, 28, 35, 19, 47, 31],
        [8, 56, 4, 52, 11, 59, 7, 55],
        [40, 24, 36, 20, 43, 27, 39, 23],
        [2, 50, 14, 62, 1, 49, 13, 61],
        [34, 18, 46, 30, 33, 17, 45, 29],
        [10, 58, 6, 54, 9, 57, 5, 53],
        [42, 26, 38, 22, 41, 25, 37, 21]
    ], dtype=np.float32) / 64.0
    
    def __init__(self, config: Config):
        """Initialize dither engine with configuration."""
        self.config = config
        self.rng = np.random.RandomState(config.processing.seed or 42)
    
    def apply_dithering(self, original_lab: np.ndarray, 
                      quantized_lab: np.ndarray) -> np.ndarray:
        """
        Apply dithering to reduce banding artifacts.
        
        Args:
            original_lab: Original image in Lab color space
            quantized_lab: Quantized image in Lab color space
            
        Returns:
            Dithered image in Lab color space
        """
        if self.config.dither.mode == "none":
            return quantized_lab
        
        print(f"Applying {self.config.dither.mode} dithering...")
        
        if self.config.dither.mode == "ordered":
            return self._apply_ordered_dithering(original_lab, quantized_lab)
        elif self.config.dither.mode == "fs":
            return self._apply_floyd_steinberg(original_lab, quantized_lab)
        else:
            return quantized_lab
    
    def _apply_ordered_dithering(self, original_lab: np.ndarray,
                               quantized_lab: np.ndarray) -> np.ndarray:
        """Apply Bayer ordered dithering with strength control - OPTIMIZED."""
        h, w, c = original_lab.shape
        
        # Calculate error between original and quantized - ensure float32
        original_f32 = original_lab.astype(np.float32)
        quantized_f32 = quantized_lab.astype(np.float32)
        error = original_f32 - quantized_f32
        
        # Create Bayer pattern matrix tiled to image size - VECTORIZED
        bayer_tiled = np.tile(self.BAYER_MATRIX_8x8, 
                            (h // 8 + 1, w // 8 + 1))[:h, :w]
        
        # Apply dithering with strength control
        strength = self.config.dither.strength
        
        # Skip expensive flat region detection for speed
        if self.config.dither.auto_disable_flat and (h * w) < 500000:
            flat_mask = self._detect_flat_regions(original_lab)
            dither_strength = strength * (1 - flat_mask)
            # Expand to 3 channels for broadcasting - make 3D array
            dither_strength = np.stack([dither_strength] * c, axis=2)  # Shape: (h, w, c)
        else:
            # Create 3D array for scalar broadcasting
            dither_strength = np.full((h, w, c), strength)  # Shape: (h, w, c)
        
        # VECTORIZED ordered dithering - no loops!
        # Expand bayer_tiled to 3D for broadcasting with dither_strength
        bayer_3d = np.stack([bayer_tiled] * c, axis=2)  # Shape: (h, w, c)
        dither_pattern = (bayer_3d - 0.5) * dither_strength
        # Multiply all channels at once
        dither_adjustment = error * dither_pattern
        dithered = quantized_lab.astype(np.float32) + dither_adjustment
        
        # Clamp values to valid Lab range
        dithered = self._clamp_lab(dithered)
        
        return dithered
    
    def _apply_floyd_steinberg(self, original_lab: np.ndarray,
                              quantized_lab: np.ndarray) -> np.ndarray:
        """Apply Floyd-Steinberg error diffusion dithering - OPTIMIZED."""
        h, w, c = original_lab.shape
        
        # For large images, skip expensive Floyd-Steinberg
        if h * w > 1000000:  # Over 1M pixels
            print("  Image too large for Floyd-Steinberg, using ordered dithering instead...")
            return self._apply_ordered_dithering(original_lab, quantized_lab)
        
        dithered = np.zeros_like(original_lab)
        
        # Work with float for error diffusion
        original_float = original_lab.astype(np.float32)
        quantized_float = quantized_lab.astype(np.float32)
        
        # Copy quantized as starting point
        dithered[:, :, :] = quantized_float
        
        # Apply Floyd-Steinberg for each channel
        for channel in range(c):
            # Calculate initial error
            error = original_float[:, :, channel] - quantized_float[:, :, channel]
            
            # Skip expensive flat region checking for large images
            check_flat = self.config.dither.auto_disable_flat and (h * w) < 200000
            
            # Error diffusion kernel - OPTIMIZED
            for y in range(h):
                for x in range(w):
                    # Skip flat region check for speed
                    if check_flat and self._is_flat_pixel(original_float, y, x):
                        continue
                    
                    # Get current error at this pixel
                    current_error = error[y, x]
                    
                    # Apply error to neighbors - combine checks for speed
                    if x < w - 1:
                        error[y, x + 1] += current_error * 7/16
                    
                    if y < h - 1:
                        error[y + 1, x] += current_error * 5/16
                        if x > 0:
                            error[y + 1, x - 1] += current_error * 3/16
                        if x < w - 1:
                            error[y + 1, x + 1] += current_error * 1/16
                    
                    # Update dithered pixel with accumulated error
                    dithered[y, x, channel] = quantized_float[y, x, channel] + current_error
        
        # Clamp to valid Lab range
        dithered = self._clamp_lab(dithered)
        
        return dithered
    
    def _detect_flat_regions(self, image_lab: np.ndarray) -> np.ndarray:
        """Detect regions with low variance to reduce dithering artifacts."""
        h, w, c = image_lab.shape
        
        # Calculate local variance using a small window
        window_size = 3
        flat_mask = np.zeros((h, w), dtype=np.float32)
        
        # Convert to grayscale for variance calculation
        gray = image_lab[:, :, 0]  # Use L channel for variance
        
        # Pad image for edge handling
        padded = np.pad(gray, window_size//2, mode='reflect')
        
        for y in range(h):
            for x in range(w):
                # Extract local window
                window = padded[y:y+window_size, x:x+window_size]
                
                # Calculate variance
                local_variance = np.var(window)
                
                # Mark as flat if variance is below threshold
                if local_variance < self.config.dither.variance_threshold:
                    flat_mask[y, x] = 1.0
                else:
                    flat_mask[y, x] = 0.0
        
        # Apply Gaussian blur to smooth the mask
        from scipy.ndimage import gaussian_filter
        flat_mask = gaussian_filter(flat_mask, sigma=1.0)
        
        return flat_mask
    
    def _is_flat_pixel(self, image_float: np.ndarray, y: int, x: int) -> bool:
        """Check if a pixel is in a flat region."""
        h, w, c = image_float.shape
        
        # Check local variance around pixel
        y_start = max(0, y - 1)
        y_end = min(h, y + 2)
        x_start = max(0, x - 1)
        x_end = min(w, x + 2)
        
        local_window = image_float[y_start:y_end, x_start:x_end, 0]  # L channel
        local_variance = np.var(local_window)
        
        return local_variance < self.config.dither.variance_threshold
    
    def _clamp_lab(self, lab_image: np.ndarray) -> np.ndarray:
        """Clamp Lab values to valid ranges."""
        # Lab ranges (approximate)
        l_min, l_max = 0, 100
        a_min, a_max = -128, 127
        b_min, b_max = -128, 127
        
        # Ensure float32 for consistency
        clamped = lab_image.astype(np.float32)
        clamped[:, :, 0] = np.clip(clamped[:, :, 0], l_min, l_max)
        clamped[:, :, 1] = np.clip(clamped[:, :, 1], a_min, a_max)
        clamped[:, :, 2] = np.clip(clamped[:, :, 2], b_min, b_max)
        
        return clamped
    
    def create_dither_preview(self, original_lab: np.ndarray,
                           quantized_lab: np.ndarray,
                           patch_size: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """
        Create a preview showing dithering effects on a sample patch.
        
        Args:
            original_lab: Original image in Lab
            quantized_lab: Quantized image in Lab
            patch_size: Size of preview patch (height, width)
            
        Returns:
            RGB preview image showing different dithering modes
        """
        h, w = original_lab.shape[:2]
        
        # Extract center patch
        y_start = (h - patch_size[0]) // 2
        x_start = (w - patch_size[1]) // 2
        
        original_patch = original_lab[y_start:y_start+patch_size[0], 
                                   x_start:x_start+patch_size[1]]
        quantized_patch = quantized_lab[y_start:y_start+patch_size[0], 
                                     x_start:x_start+patch_size[1]]
        
        # Create comparison images
        images = [
            ("Original", original_patch),
            ("No Dither", quantized_patch),
            ("Ordered", self._apply_ordered_dithering(original_patch, quantized_patch)),
            ("Floyd-Steinberg", self._apply_floyd_steinberg(original_patch, quantized_patch))
        ]
        
        # Combine into single preview image
        preview_height = patch_size[0]
        preview_width = patch_size[1] * len(images)
        preview_lab = np.zeros((preview_height, preview_width, 3))
        
        for i, (name, img) in enumerate(images):
            x_start = i * patch_size[1]
            preview_lab[:, x_start:x_start+patch_size[1]] = img
        
        # Convert to RGB
        preview_rgb = self._lab_to_rgb_batch(preview_lab)
        
        return preview_rgb
    
    def _lab_to_rgb_batch(self, lab_image: np.ndarray) -> np.ndarray:
        """Convert Lab image to RGB in batch."""
        h, w, c = lab_image.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        for y in range(h):
            for x in range(w):
                lab = lab_image[y, x]
                rgb = self._lab_to_rgb_single(lab)
                rgb_image[y, x] = rgb
        
        return rgb_image
    
    def _lab_to_rgb_single(self, lab: np.ndarray) -> np.ndarray:
        """Convert single Lab pixel to RGB."""
        l, a, b_ = lab
        
        # Lab to XYZ (simplified version for preview)
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
        
        return np.array([r, g, b], dtype=np.uint8)
    
    def get_dithering_info(self) -> dict:
        """Get information about current dithering configuration."""
        return {
            'mode': self.config.dither.mode,
            'strength': self.config.dither.strength,
            'auto_disable_flat': self.config.dither.auto_disable_flat,
            'variance_threshold': self.config.dither.variance_threshold,
            'description': self._get_dither_description()
        }
    
    def _get_dither_description(self) -> str:
        """Get human-readable description of dithering mode."""
        if self.config.dither.mode == "none":
            return "No dithering - pure color quantization"
        elif self.config.dither.mode == "ordered":
            return f"Ordered dithering (Bayer 8x8) with {self.config.dither.strength:.0%} strength"
        elif self.config.dither.mode == "fs":
            return "Floyd-Steinberg error diffusion dithering"
        else:
            return "Unknown dithering mode"
